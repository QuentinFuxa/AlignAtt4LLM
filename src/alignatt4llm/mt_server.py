"""Text-in MT sidecar server for external ASR frontends.

Serves the AlignAtt MT policy over WebSocket to clients that run their own
ASR (WhisperLiveKit is the reference client). The client streams committed
source words (plus, optionally, the unstable hypothesis tail) with word
timestamps and a monotone stream clock; the server answers with append-only
translation deltas per utterance and a final quality pass at utterance
boundaries. Protocol v1 is specified in ``docs/mt_server_protocol.md``.

Frontier semantics (the smart part): committed words carry real timestamps
and the unstable tail carries none, so ``build_source_accessibility_frontier``
marks exactly the committed prefix accessible. The MT model drafts over the
full prompt, tail included; AlignAtt holds target tokens whose attention
lands on the tail. When the upstream ASR later commits those tail words the
prompt is unchanged, and with ``translation_alignatt_commit_fast_path`` the
held tokens release from the cached draft without a new MT engine call.

Run: ``alignatt-mt-server --preset gemma_low_latency --port 8765``
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from alignatt4llm.presets import get_runtime_preset
from alignatt4llm.runtime import (
    LANGUAGE_CODE_TO_NAME,
    SAMPLE_RATE,
    CascadeRuntimeConfig,
    CascadeSession,
    LoadedModelBundle,
    PartialTranslationState,
    alignatt_heads_path_for,
    target_lang_code_for,
)
from alignatt4llm.source_frontier import iter_source_word_spans
from alignatt4llm.text_surface import (
    join_public_emission_units,
    split_public_emission_units,
)

try:  # optional at import time so unit tests can stub the transport
    import websockets.asyncio.server as _ws_server
except ImportError:  # pragma: no cover - exercised only without websockets
    _ws_server = None

LOGGER = logging.getLogger("alignatt_mt_server")

PROTOCOL_VERSION = 1

# Per-session knobs a client may override at init. Everything else is the
# server operator's business (CLI --override) or stays preset-internal.
CLIENT_OVERRIDE_WHITELIST = (
    "translation_alignatt_inaccessible_ms",
    "translation_alignatt_border_margin",
    "translation_alignatt_hold_back_target_units",
    "translation_alignatt_min_emit_target_units",
    "translation_alignatt_commit_fast_path",
    "translation_alignatt_source_lcp_stability",
    "partial_max_new_tokens",
    "max_history_utterances",
)


def supported_target_codes(
    source_code: str,
    *,
    mt_backend_name: str,
) -> list[str]:
    """Target language codes whose calibrated heads file ships in data/."""
    source_name = LANGUAGE_CODE_TO_NAME.get(source_code, source_code)
    supported = []
    for target_code, target_name in sorted(LANGUAGE_CODE_TO_NAME.items()):
        if target_code == source_code:
            continue
        heads = alignatt_heads_path_for(
            source_name,
            target_name,
            mt_backend_name=mt_backend_name,
        )
        if Path(heads).is_file():
            supported.append(target_code)
    return supported


def _word_has_lexical_span(token: str) -> bool:
    for _ in iter_source_word_spans(token):
        return True
    return False


def compose_source_words(
    committed_words: Sequence[Sequence[Any]],
    tail_words: Sequence[Sequence[Any]],
) -> tuple[str, list[tuple[float | None, float | None]], float]:
    """Compose the utterance hypothesis text and span-aligned timestamps.

    Committed words keep their real timestamps (missing ones are backfilled
    from the previous word so a commit never becomes inaccessible by
    accident); tail words are stamped ``(None, None)``, which is what makes
    the frontier stop the accessible prefix exactly at the commit boundary.
    Timestamps are emitted once per lexical span (``iter_source_word_spans``
    drops punctuation-only tokens), keeping them index-aligned with the
    normalized source units downstream.
    """
    parts: list[str] = []
    stamps: list[tuple[float | None, float | None]] = []
    last_end_ms = 0.0
    for row in committed_words:
        text = str(row[0]).strip()
        start_ms = row[1] if len(row) > 1 else None
        end_ms = row[2] if len(row) > 2 else None
        if not text:
            continue
        parts.append(text)
        for piece in text.split():
            if not _word_has_lexical_span(piece):
                continue
            piece_end = last_end_ms if end_ms is None else float(end_ms)
            piece_start = piece_end if start_ms is None else float(start_ms)
            last_end_ms = max(last_end_ms, piece_end)
            stamps.append((piece_start, piece_end))
    for row in tail_words:
        text = str(row[0]).strip()
        if not text:
            continue
        parts.append(text)
        for piece in text.split():
            if _word_has_lexical_span(piece):
                stamps.append((None, None))
    return " ".join(parts), stamps, last_end_ms


def append_only_advance(
    previous_text: str,
    candidate_text: str,
    *,
    target_lang_code: str,
) -> tuple[str, str]:
    """Wire-level append-only guard over public emission units.

    Returns ``(new_emitted_text, appended_delta)``; when the candidate does
    not extend the previous emission unit-wise, the previous text is kept and
    the delta is empty. Mirrors the cascade processor's incremental-output
    rule without depending on the simulstream package.
    """
    previous_units = split_public_emission_units(
        previous_text, target_lang_code=target_lang_code
    )
    candidate_units = split_public_emission_units(
        candidate_text, target_lang_code=target_lang_code
    )
    if len(candidate_units) < len(previous_units):
        return previous_text, ""
    if candidate_units[: len(previous_units)] != previous_units:
        return previous_text, ""
    added_units = candidate_units[len(previous_units) :]
    if not added_units:
        return previous_text, ""
    return (
        join_public_emission_units(candidate_units, target_lang_code=target_lang_code),
        join_public_emission_units(added_units, target_lang_code=target_lang_code),
    )


class TextMTSession(CascadeSession):
    """CascadeSession driven by text updates instead of audio chunks.

    The audio buffer stays empty; the stream clock is the client's
    ``clock_ms`` and the ASR-owned state fields (``asr_hypotheses``,
    ``partial_word_timestamps_ms``, ``utt_sources``) are written from
    protocol updates. Everything MT-side (frontier, scheduler, monotone
    acceptance, finals with partial prefill) is reused unchanged.
    """

    def __init__(self, bundle: LoadedModelBundle, config: CascadeRuntimeConfig):
        super().__init__(bundle)
        self.config = config  # per-session direction/knobs, not bundle.config
        self.external_clock_ms = 0.0
        self.static_context_text = ""

    def current_audio_seconds(self) -> float:  # clock override, no audio
        return float(self.external_clock_ms) / 1000.0

    def _render_paper_context_block(self, **kwargs):
        if self.static_context_text.strip():
            from alignatt4llm.paper_context import PaperContextBlock

            return PaperContextBlock(
                text=f"[Context]\n{self.static_context_text.strip()}",
                mode="static",
            )
        return super()._render_paper_context_block(**kwargs)

    def ingest_update(
        self,
        *,
        committed_words: Sequence[Sequence[Any]],
        tail_words: Sequence[Sequence[Any]],
        clock_ms: float,
    ) -> None:
        text, stamps, last_committed_end_ms = compose_source_words(
            committed_words, tail_words
        )
        self.external_clock_ms = max(
            self.external_clock_ms, float(clock_ms), last_committed_end_ms
        )
        self.state.asr_hypotheses.append(text)
        # Only the last two hypotheses are ever read (LCP stability).
        if len(self.state.asr_hypotheses) > 4:
            self.state.asr_hypotheses = self.state.asr_hypotheses[-4:]
        self.state.partial_word_timestamps_ms = stamps

    def finalize_utterance(self) -> None:
        utterance_text = (
            self.state.asr_hypotheses[-1] if self.state.asr_hypotheses else ""
        ).strip()
        end_sample = int(self.current_audio_seconds() * SAMPLE_RATE)
        self.state.utt_timestamps.append(
            max(end_sample, self.state.utt_timestamps[-1])
        )
        self.state.utt_sources.append(utterance_text)
        self.state.asr_hypotheses = [""]
        self.state.partial_word_timestamps_ms = []

    def seed_accepted_prefix(self, accepted_target_prefix: str) -> None:
        """Reconnect support: continue an utterance whose partial target was
        already shown to the user. The prefix becomes the assistant prefill
        for the next drafts; the wire guard keeps emission append-only."""
        prefix = accepted_target_prefix.strip()
        if not prefix:
            return
        self.state.partial_translation = PartialTranslationState(
            accepted_target=prefix,
        )

    def seed_history(self, history: Sequence[Sequence[str]]) -> None:
        for pair in history:
            if not pair:
                continue
            source = str(pair[0]).strip()
            target = str(pair[1]).strip() if len(pair) > 1 else ""
            self.state.utt_timestamps.append(self.state.utt_timestamps[-1])
            self.state.utt_sources.append(source)
            self.state.utt_translations.append(target)


@dataclass
class SessionOutputs:
    """Per-connection emission bookkeeping (wire append-only guard)."""

    utterance_id: int = 0
    emitted_partial: str = ""


def build_session_config(
    *,
    preset_name: str,
    mt_backend_name: str,
    source_code: str,
    target_code: str,
    server_overrides: dict[str, Any],
    client_overrides: dict[str, Any],
) -> CascadeRuntimeConfig:
    preset = get_runtime_preset(preset_name)
    source_name = LANGUAGE_CODE_TO_NAME.get(source_code, source_code)
    target_name = LANGUAGE_CODE_TO_NAME.get(target_code, target_code)
    config = CascadeRuntimeConfig(
        source_lang=source_name,
        target_lang=target_name,
        mt_backend_name=mt_backend_name,
        min_start_seconds=0.0,
        max_history_utterances=preset.max_history_utterances,
        partial_max_new_tokens=preset.partial_max_new_tokens,
        translation_alignatt_top_k_heads=preset.translation_alignatt_top_k_heads,
        translation_alignatt_border_margin=preset.translation_alignatt_border_margin,
        translation_alignatt_min_source_mass=preset.translation_alignatt_min_source_mass,
        translation_alignatt_inaccessible_ms=preset.translation_alignatt_inaccessible_ms,
        translation_alignatt_argmax_mass_threshold=preset.translation_alignatt_argmax_mass_threshold,
        translation_alignatt_frontier_min_inaccessible_mass=preset.translation_alignatt_frontier_min_inaccessible_mass,
        translation_alignatt_max_inaccessible_source_mass=preset.translation_alignatt_max_inaccessible_source_mass,
        translation_alignatt_min_accessible_inaccessible_margin=preset.translation_alignatt_min_accessible_inaccessible_margin,
        translation_acceptance_policy=preset.translation_acceptance_policy,
        mt_vllm_enforce_eager=preset.mt_vllm_enforce_eager,
        mt_vllm_gpu_memory_utilization=preset.mt_vllm_gpu_memory_utilization,
        # The text path exists precisely for external-commit release.
        translation_alignatt_commit_fast_path=True,
    )
    config.translation_alignatt_heads_path = alignatt_heads_path_for(
        source_name,
        target_name,
        mt_backend_name=mt_backend_name,
    )
    merged = {**server_overrides, **client_overrides}
    if merged:
        config.apply_overrides(**merged)
    return config


class TextMTEngine:
    """One loaded MT model shared by all sessions, strictly serialized.

    The vLLM attention observer supports a single in-flight request, so every
    MT call runs on a one-worker executor. Sessions may use different
    directions on the same model: the bundle config is swapped under that
    serialization before each call (a heads-path change triggers a cheap
    artifact refresh, not an engine reload).
    """

    def __init__(
        self,
        *,
        preset_name: str,
        mt_backend_name: str = "gemma_vllm_alignatt",
        source_code: str = "en",
        server_overrides: dict[str, Any] | None = None,
        mt_backend_factory=None,
    ):
        self.preset_name = preset_name
        self.mt_backend_name = mt_backend_name
        self.source_code = source_code
        self.server_overrides = dict(server_overrides or {})
        self._mt_backend_factory = mt_backend_factory
        self._bundle: LoadedModelBundle | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="alignatt-mt"
        )

    def supported_targets(self) -> list[str]:
        return supported_target_codes(
            self.source_code, mt_backend_name=self.mt_backend_name
        )

    def _ensure_bundle(self, config: CascadeRuntimeConfig) -> LoadedModelBundle:
        if self._bundle is None:
            bundle = LoadedModelBundle(config)
            if self._mt_backend_factory is not None:
                bundle.mt_backend = self._mt_backend_factory(config)
                bundle._mt_backend_fp = config.mt_backend_fingerprint()
                bundle._mt_heads_path = config.translation_alignatt_heads_path
            else:
                bundle.ensure_mt_backend()
            self._bundle = bundle
        return self._bundle

    def new_session(self, config: CascadeRuntimeConfig) -> TextMTSession:
        bundle = self._ensure_bundle(config)
        return TextMTSession(bundle, config)

    def warmup(self, target_code: str) -> None:
        config = build_session_config(
            preset_name=self.preset_name,
            mt_backend_name=self.mt_backend_name,
            source_code=self.source_code,
            target_code=target_code,
            server_overrides=self.server_overrides,
            client_overrides={},
        )
        session = self.new_session(config)
        session.ingest_update(
            committed_words=[("Hello", 0.0, 400.0), ("world.", 500.0, 900.0)],
            tail_words=[],
            clock_ms=10_000.0,
        )
        self.run_serialized_sync(session, lambda: session.render_translation())
        LOGGER.info("warmup complete for %s-%s", self.source_code, target_code)

    def run_serialized_sync(self, session: TextMTSession, fn):
        """Run one MT-touching callable with the bundle pointed at this
        session's config. Callers must go through the single-worker executor
        (or, at startup, this synchronous door) so calls never interleave."""
        bundle = self._ensure_bundle(session.config)
        bundle.config = session.config
        bundle.ensure_mt_backend()
        return fn()

    async def run_serialized(self, session: TextMTSession, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.run_serialized_sync(session, fn)
        )


def _partial_buffer_text(session: TextMTSession) -> str:
    partial = session.state.partial_translation
    draft = (partial.draft_target or "").strip()
    accepted = (partial.accepted_target or "").strip()
    if draft and draft.startswith(accepted):
        return draft[len(accepted) :].strip()
    return ""


def _covered_source_units(session: TextMTSession) -> int | None:
    partial = session.state.partial_translation
    metadata = partial.last_alignatt_metadata or {}
    indices = metadata.get("aligned_source_unit_indices")
    if not isinstance(indices, list):
        return None
    accepted_count = len(partial.accepted_generated_token_ids)
    covered = None
    for index in indices[:accepted_count]:
        if index is None:
            continue
        covered = index if covered is None else max(covered, int(index))
    return None if covered is None else int(covered) + 1


class MTServerConnection:
    """Protocol state machine for one client connection."""

    def __init__(self, engine: TextMTEngine, *, max_utterance_words: int):
        self.engine = engine
        self.max_utterance_words = max_utterance_words
        self.session: TextMTSession | None = None
        self.outputs = SessionOutputs()
        self.target_code = "de"

    def handle_init(self, message: dict[str, Any]) -> dict[str, Any]:
        if int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
            return {
                "type": "error",
                "code": "bad_protocol",
                "message": (
                    f"protocol_version must be {PROTOCOL_VERSION}"
                ),
            }
        source_code = str(message.get("source_lang", self.engine.source_code)).lower()
        target_code = str(message.get("target_lang", "")).lower()
        supported = self.engine.supported_targets()
        if source_code != self.engine.source_code or target_code not in supported:
            return {
                "type": "error",
                "code": "unsupported_direction",
                "message": (
                    f"direction {source_code}-{target_code} is not available on "
                    f"this server (model {self.engine.mt_backend_name})"
                ),
                "supported": [f"{self.engine.source_code}-{t}" for t in supported],
            }
        client_overrides = {
            key: value
            for key, value in dict(message.get("overrides") or {}).items()
            if key in CLIENT_OVERRIDE_WHITELIST
        }
        config = build_session_config(
            preset_name=str(message.get("preset") or self.engine.preset_name),
            mt_backend_name=self.engine.mt_backend_name,
            source_code=source_code,
            target_code=target_code,
            server_overrides=self.engine.server_overrides,
            client_overrides=client_overrides,
        )
        self.session = self.engine.new_session(config)
        self.target_code = target_code
        context_text = str(message.get("context_text") or "")
        if context_text.strip():
            self.session.static_context_text = context_text
        history = message.get("history") or []
        if history:
            self.session.seed_history(history)
        accepted_prefix = str(message.get("accepted_target_prefix") or "")
        if accepted_prefix.strip():
            self.session.seed_accepted_prefix(accepted_prefix)
            self.outputs.emitted_partial = accepted_prefix.strip()
        return {
            "type": "init_ok",
            "protocol_version": PROTOCOL_VERSION,
            "direction": f"{source_code}-{target_code}",
            "preset": str(message.get("preset") or self.engine.preset_name),
            "model": self.engine.mt_backend_name,
        }

    def process_update_sync(self, update: dict[str, Any]) -> dict[str, Any]:
        """Runs on the engine executor: ingest + render + wire guard."""
        session = self.session
        assert session is not None
        committed_words = update.get("words") or []
        tail = update.get("tail") or {}
        tail_words = tail.get("words") or []
        clock_ms = float(update.get("clock_ms") or 0.0)
        is_final = bool(update.get("is_final"))
        forced_final = False
        if (
            not is_final
            and self.max_utterance_words > 0
            and len(committed_words) > self.max_utterance_words
        ):
            is_final = True
            forced_final = True
            tail_words = []

        session.ingest_update(
            committed_words=committed_words,
            tail_words=tail_words,
            clock_ms=clock_ms,
        )

        if is_final:
            session.finalize_utterance()
            session.translation_units.sync_committed_translations()
            final_text = (
                session.state.utt_translations[-1]
                if len(session.state.utt_translations) > 1
                else ""
            ).strip()
            response = {
                "type": "translation",
                "seq": update.get("seq"),
                "utterance_id": self.outputs.utterance_id,
                "final": True,
                "forced_final": forced_final,
                "committed_text": final_text,
                "committed_delta": "",
                "buffer_text": "",
                "covered_source_units": None,
            }
            self.outputs.utterance_id += 1
            self.outputs.emitted_partial = ""
            return response

        _, result = session.translation_units.render_translation()
        candidate = session.state.partial_translation.accepted_target.strip()
        emitted, delta = append_only_advance(
            self.outputs.emitted_partial,
            candidate,
            target_lang_code=self.target_code,
        )
        self.outputs.emitted_partial = emitted
        return {
            "type": "translation",
            "seq": update.get("seq"),
            "utterance_id": self.outputs.utterance_id,
            "final": False,
            "committed_text": emitted,
            "committed_delta": delta,
            "buffer_text": _partial_buffer_text(session),
            "covered_source_units": _covered_source_units(session),
            "stop_reason": (result.stop_reason if result is not None else None),
        }


class UpdateMailbox:
    """Latest-wins mailbox for partial updates; finals are sticky and ordered."""

    def __init__(self):
        self._finals: list[dict[str, Any]] = []
        self._latest_partial: dict[str, Any] | None = None
        self._event = asyncio.Event()
        self.closed = False

    def put(self, update: dict[str, Any]) -> None:
        if bool(update.get("is_final")):
            self._finals.append(update)
        else:
            self._latest_partial = update
        self._event.set()

    def close(self) -> None:
        self.closed = True
        self._event.set()

    async def get(self) -> dict[str, Any] | None:
        while True:
            if self._finals:
                return self._finals.pop(0)
            if self._latest_partial is not None:
                update = self._latest_partial
                self._latest_partial = None
                return update
            if self.closed:
                return None
            self._event.clear()
            await self._event.wait()


async def handle_connection(websocket, engine: TextMTEngine, *, max_utterance_words: int):
    connection = MTServerConnection(engine, max_utterance_words=max_utterance_words)
    mailbox = UpdateMailbox()

    async def consumer():
        while True:
            update = await mailbox.get()
            if update is None:
                return
            try:
                response = await engine.run_serialized(
                    connection.session, lambda: connection.process_update_sync(update)
                )
            except Exception as exc:  # a bad update must not kill the session
                LOGGER.exception("update processing failed")
                response = {
                    "type": "error",
                    "code": "processing_failed",
                    "seq": update.get("seq"),
                    "message": str(exc),
                }
            await websocket.send(json.dumps(response, ensure_ascii=False))

    consumer_task: asyncio.Task | None = None
    try:
        async for raw in websocket:
            if isinstance(raw, bytes):
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps(
                    {"type": "error", "code": "bad_protocol", "message": "invalid JSON"}
                ))
                continue
            message_type = message.get("type")
            if message_type == "init":
                if connection.session is not None:
                    await websocket.send(json.dumps(
                        {"type": "error", "code": "bad_protocol",
                         "message": "session already initialized"}
                    ))
                    continue
                response = connection.handle_init(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
                if response.get("type") == "error":
                    await websocket.close()
                    return
                consumer_task = asyncio.create_task(consumer())
            elif message_type == "update":
                if connection.session is None:
                    await websocket.send(json.dumps(
                        {"type": "error", "code": "bad_protocol",
                         "message": "update before init"}
                    ))
                    continue
                mailbox.put(message)
            else:
                await websocket.send(json.dumps(
                    {"type": "error", "code": "bad_protocol",
                     "message": f"unknown message type: {message_type!r}"}
                ))
    finally:
        mailbox.close()
        if consumer_task is not None:
            await consumer_task


async def serve_mt(
    *,
    engine: TextMTEngine,
    host: str,
    port: int,
    max_sessions: int,
    max_utterance_words: int,
) -> None:
    if _ws_server is None:
        raise ImportError(
            "The MT server requires the websockets package: pip install websockets"
        )
    active = 0

    async def handler(websocket):
        nonlocal active
        if active >= max_sessions:
            await websocket.send(json.dumps(
                {"type": "error", "code": "busy",
                 "message": f"server is at capacity ({max_sessions} sessions)"}
            ))
            await websocket.close()
            return
        active += 1
        try:
            await handle_connection(
                websocket, engine, max_utterance_words=max_utterance_words
            )
        finally:
            active -= 1

    LOGGER.info("serving AlignAtt MT at ws://%s:%s", host, port)
    async with _ws_server.serve(handler, host, port, ping_timeout=None) as server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preset", default="gemma_low_latency")
    parser.add_argument(
        "--mt-backend",
        default="gemma_vllm_alignatt",
        choices=["gemma_vllm_alignatt", "milmmt_vllm_alignatt", "qwen_vllm_alignatt"],
    )
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument(
        "--max-utterance-words",
        type=int,
        default=120,
        help="Force-finalize utterances beyond this many committed words "
        "(bounds prompt growth for clients that never punctuate).",
    )
    parser.add_argument(
        "--warmup-target",
        default=None,
        help="Run one warmup translation for this target code at startup "
        "(kills first-token latency). Default: first supported target.",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="CascadeRuntimeConfig override applied to every session.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    server_overrides: dict[str, Any] = {}
    for item in args.override:
        key, _, value = item.partition("=")
        try:
            server_overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            server_overrides[key] = value

    engine = TextMTEngine(
        preset_name=args.preset,
        mt_backend_name=args.mt_backend,
        source_code=args.source_lang,
        server_overrides=server_overrides,
    )
    supported = engine.supported_targets()
    if not supported:
        raise SystemExit(
            f"no calibrated heads found for source {args.source_lang!r} and "
            f"backend {args.mt_backend!r}; check data/alignatt_heads/"
        )
    LOGGER.info("supported directions: %s", ", ".join(f"en-{t}" for t in supported))
    if not args.no_warmup:
        engine.warmup(args.warmup_target or supported[0])

    asyncio.run(serve_mt(
        engine=engine,
        host=args.host,
        port=args.port,
        max_sessions=args.max_sessions,
        max_utterance_words=args.max_utterance_words,
    ))


if __name__ == "__main__":
    main()
