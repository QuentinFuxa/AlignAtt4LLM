"""Protocol tests for the text-in MT sidecar (no GPU, no vLLM).

A deterministic FakeMTBackend replaces the vLLM engine: it uppercases
source units one-for-one and aligns target token i to source unit i, so
frontier cuts, fast-path releases, and append-only guarantees are all
checkable exactly.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from alignatt4llm.mt.base import MTBackendResult
from alignatt4llm.mt_server import (
    CLIENT_OVERRIDE_WHITELIST,
    MTServerConnection,
    TextMTEngine,
    UpdateMailbox,
    append_only_advance,
    build_session_config,
    compose_source_words,
    handle_connection,
)


class FakeMTBackend:
    """Uppercase word-mapper with exact source-unit alignment.

    Target token i translates source unit i. Acceptance on partials is the
    accessible-unit prefix of the frontier; on finals it is the full draft.
    """

    def __init__(self, config):
        self.runtime_config = config
        self.calls: list[dict] = []
        self._word_by_id: dict[int, str] = {}
        self._id_by_word: dict[str, int] = {}

    def _token_id(self, word: str) -> int:
        if word not in self._id_by_word:
            token_id = len(self._id_by_word) + 1
            self._id_by_word[word] = token_id
            self._word_by_id[token_id] = word
        return self._id_by_word[word]

    def refresh_alignatt_artifacts(self) -> None:
        pass

    def encode_semantic_target_token_ids(self, text: str):
        return tuple(self._token_id(word) for word in text.split())

    def decode_candidate_text(self, *, generated_ids, assistant_prefill, variant, is_partial):
        return " ".join(self._word_by_id[int(i)] for i in generated_ids)

    def translate(self, *, rendered_prompt, variant, is_partial, prompt_cache_state):
        frontier = rendered_prompt.source_frontier
        units = [unit.text for unit in frontier.units]
        accessible = frontier.accessible_unit_count if not frontier.is_final else len(units)
        self.calls.append(
            {
                "source_text": rendered_prompt.source_text,
                "is_partial": is_partial,
                "accessible": accessible,
            }
        )
        draft_words = [unit.upper() for unit in units]
        ids = tuple(self._token_id(word) for word in draft_words)
        accepted_words = draft_words[:accessible]
        return MTBackendResult(
            draft_text=" ".join(draft_words),
            acceptance_text=" ".join(accepted_words),
            draft_generated_token_ids=ids,
            accepted_generated_token_ids=ids[:accessible],
            draft_token_ids=ids,
            accepted_token_ids=ids[:accessible],
            stop_reason="fake",
            alignatt_metadata={
                "aligned_source_unit_indices": list(range(len(units))),
            },
        )


def make_engine() -> TextMTEngine:
    return TextMTEngine(
        preset_name="gemma_low_latency",
        mt_backend_name="gemma_vllm_alignatt",
        source_code="en",
        mt_backend_factory=FakeMTBackend,
    )


def make_connection(engine: TextMTEngine | None = None, **init_extra):
    engine = engine or make_engine()
    connection = MTServerConnection(engine, max_utterance_words=50)
    response = connection.handle_init(
        {
            "type": "init",
            "protocol_version": 1,
            "source_lang": "en",
            "target_lang": "de",
            **init_extra,
        }
    )
    assert response["type"] == "init_ok", response
    return engine, connection


def run_update(engine, connection, update):
    return engine.run_serialized_sync(
        connection.session, lambda: connection.process_update_sync(update)
    )


def fake_backend(engine) -> FakeMTBackend:
    return engine._bundle.mt_backend


# ---------------------------------------------------------------------------
# compose / guard helpers
# ---------------------------------------------------------------------------


def test_compose_source_words_backfills_missing_committed_timestamps():
    text, stamps, last_end = compose_source_words(
        [("Hello", 0.0, 400.0), ("world", None, None)],
        [("tail", None, None)],
    )
    assert text == "Hello world tail"
    assert stamps == [(0.0, 400.0), (400.0, 400.0), (None, None)]
    assert last_end == 400.0


def test_compose_source_words_splits_multiword_tokens():
    text, stamps, _ = compose_source_words([("good morning", 0.0, 800.0)], [])
    assert text == "good morning"
    assert len(stamps) == 2


def test_append_only_advance_rejects_rewrites():
    emitted, delta = append_only_advance("HELLO WORLD", "HELLO EARTH", target_lang_code="de")
    assert emitted == "HELLO WORLD"
    assert delta == ""
    emitted, delta = append_only_advance("HELLO", "HELLO WORLD", target_lang_code="de")
    assert emitted == "HELLO WORLD"
    # deltas are verbatim appendable: previous + delta == new emitted text
    assert delta == " WORLD"
    assert "HELLO" + delta == emitted


# ---------------------------------------------------------------------------
# protocol: partials, frontier, fast path
# ---------------------------------------------------------------------------


def test_partial_holds_tail_and_commits_accessible_words():
    engine, connection = make_connection()
    response = run_update(engine, connection, {
        "seq": 1,
        "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0]],
        "tail": {"words": [["world", None, None]]},
        "clock_ms": 1200.0,
        "is_final": False,
    })
    assert response["final"] is False
    assert response["committed_text"] == "HELLO"
    assert response["committed_delta"] == "HELLO"
    # the tail word was drafted over but held by the frontier
    assert response["buffer_text"] == "WORLD"
    assert response["covered_source_units"] == 1


def test_commit_release_uses_fast_path_without_engine_call():
    engine, connection = make_connection()
    run_update(engine, connection, {
        "seq": 1, "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0]],
        "tail": {"words": [["world.", None, None]]},
        "clock_ms": 1200.0, "is_final": False,
    })
    backend = fake_backend(engine)
    calls_before = len(backend.calls)
    # The ASR now commits "world." -- the source text (after trailing
    # punctuation strip) is unchanged, only the frontier advanced.
    response = run_update(engine, connection, {
        "seq": 2, "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0], ["world.", 500.0, 900.0]],
        "tail": {"words": []},
        "clock_ms": 1500.0, "is_final": False,
    })
    assert response["committed_text"] == "HELLO WORLD"
    assert "HELLO" + response["committed_delta"] == response["committed_text"]
    assert response["stop_reason"] == "alignatt:commit_fast_path"
    assert len(backend.calls) == calls_before, "fast path must not call the engine"


def test_partial_emission_is_append_only_across_updates():
    engine, connection = make_connection()
    emitted = []
    words = []
    vocabulary = ["The", "quick", "brown", "fox", "jumps"]
    for index, word in enumerate(vocabulary):
        words.append([word, index * 500.0, index * 500.0 + 400.0])
        response = run_update(engine, connection, {
            "seq": index, "utterance_id": 0,
            "words": [list(w) for w in words],
            "tail": {"words": []},
            "clock_ms": (index + 1) * 500.0 + 400.0,
            "is_final": False,
        })
        emitted.append(response["committed_text"])
    for previous, current in zip(emitted, emitted[1:]):
        assert current.startswith(previous), (previous, current)


def test_final_returns_quality_pass_and_rolls_utterance():
    engine, connection = make_connection()
    run_update(engine, connection, {
        "seq": 1, "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0]],
        "tail": {"words": [["world.", None, None]]},
        "clock_ms": 1000.0, "is_final": False,
    })
    response = run_update(engine, connection, {
        "seq": 2, "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0], ["world.", 500.0, 900.0]],
        "tail": {"words": []},
        "clock_ms": 1500.0, "is_final": True,
    })
    assert response["final"] is True
    assert response["utterance_id"] == 0
    assert response["committed_text"] == "HELLO WORLD"
    # next utterance starts clean
    response2 = run_update(engine, connection, {
        "seq": 3, "utterance_id": 1,
        "words": [["Again", 2000.0, 2400.0], ["here", 2500.0, 2900.0]],
        "tail": {"words": []},
        "clock_ms": 3000.0, "is_final": False,
    })
    assert response2["utterance_id"] == 1
    assert response2["committed_text"] == "AGAIN HERE"


def test_forced_final_on_max_utterance_words():
    engine = make_engine()
    connection = MTServerConnection(engine, max_utterance_words=3)
    init = connection.handle_init({
        "type": "init", "protocol_version": 1,
        "source_lang": "en", "target_lang": "de",
    })
    assert init["type"] == "init_ok"
    response = run_update(engine, connection, {
        "seq": 1, "utterance_id": 0,
        "words": [[w, i * 500.0, i * 500.0 + 400.0] for i, w in
                  enumerate(["one", "two", "three", "four"])],
        "tail": {"words": []},
        "clock_ms": 3000.0, "is_final": False,
    })
    assert response["final"] is True
    assert response["forced_final"] is True


def test_reconnect_resumes_from_accepted_prefix():
    engine, connection = make_connection(accepted_target_prefix="HELLO")
    response = run_update(engine, connection, {
        "seq": 1, "utterance_id": 0,
        "words": [["Hello", 0.0, 400.0], ["world", 500.0, 900.0]],
        "tail": {"words": []},
        "clock_ms": 1200.0, "is_final": False,
    })
    # emission resumes append-only from the prefix shown before reconnect
    assert response["committed_text"] == "HELLO WORLD"
    assert "HELLO" + response["committed_delta"] == response["committed_text"]


def test_history_seeds_translation_context():
    engine, connection = make_connection(history=[["First sentence.", "ERSTER SATZ."]])
    session = connection.session
    assert session.state.utt_sources[-1] == "First sentence."
    assert session.state.utt_translations[-1] == "ERSTER SATZ."


# ---------------------------------------------------------------------------
# init validation
# ---------------------------------------------------------------------------


def test_unsupported_direction_lists_available():
    engine = make_engine()
    connection = MTServerConnection(engine, max_utterance_words=50)
    response = connection.handle_init({
        "type": "init", "protocol_version": 1,
        "source_lang": "en", "target_lang": "xx",
    })
    assert response["type"] == "error"
    assert response["code"] == "unsupported_direction"
    assert "en-de" in response["supported"]


def test_bad_protocol_version_rejected():
    engine = make_engine()
    connection = MTServerConnection(engine, max_utterance_words=50)
    response = connection.handle_init({"type": "init", "protocol_version": 99})
    assert response["type"] == "error"
    assert response["code"] == "bad_protocol"


def test_client_overrides_apply_only_whitelisted():
    config = build_session_config(
        preset_name="gemma_low_latency",
        mt_backend_name="gemma_vllm_alignatt",
        source_code="en",
        target_code="de",
        server_overrides={},
        client_overrides={"translation_alignatt_hold_back_target_units": 2},
    )
    assert config.translation_alignatt_hold_back_target_units == 2
    assert "translation_alignatt_hold_back_target_units" in CLIENT_OVERRIDE_WHITELIST
    assert "mt_vllm_gpu_memory_utilization" not in CLIENT_OVERRIDE_WHITELIST


# ---------------------------------------------------------------------------
# mailbox
# ---------------------------------------------------------------------------


def test_mailbox_latest_partial_wins_and_finals_are_sticky():
    async def scenario():
        mailbox = UpdateMailbox()
        mailbox.put({"seq": 1, "is_final": False})
        mailbox.put({"seq": 2, "is_final": False})
        mailbox.put({"seq": 3, "is_final": True})
        mailbox.put({"seq": 4, "is_final": False})
        first = await mailbox.get()
        second = await mailbox.get()
        mailbox.close()
        third = await mailbox.get()
        return first, second, third

    first, second, third = asyncio.run(scenario())
    assert first["seq"] == 3, "finals are processed before pending partials"
    assert second["seq"] == 4, "latest partial wins, seq 1-2 coalesced away"
    assert third is None


# ---------------------------------------------------------------------------
# WebSocket end to end (real sockets, fake backend)
# ---------------------------------------------------------------------------


def test_ws_end_to_end_session():
    ws_server = pytest.importorskip("websockets.asyncio.server")
    ws_client = pytest.importorskip("websockets.asyncio.client")

    async def scenario():
        engine = make_engine()

        async def handler(websocket):
            await handle_connection(websocket, engine, max_utterance_words=50)

        async with ws_server.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with ws_client.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({
                    "type": "init", "protocol_version": 1,
                    "source_lang": "en", "target_lang": "de",
                }))
                init_ok = json.loads(await ws.recv())
                assert init_ok["type"] == "init_ok"
                assert init_ok["direction"] == "en-de"

                await ws.send(json.dumps({
                    "type": "update", "seq": 1, "utterance_id": 0,
                    "words": [["Hello", 0.0, 400.0]],
                    "tail": {"words": [["there", None, None]]},
                    "clock_ms": 1000.0, "is_final": False,
                }))
                partial = json.loads(await ws.recv())
                assert partial["type"] == "translation"
                assert partial["committed_text"] == "HELLO"
                assert partial["buffer_text"] == "THERE"

                await ws.send(json.dumps({
                    "type": "update", "seq": 2, "utterance_id": 0,
                    "words": [["Hello", 0.0, 400.0], ["there.", 500.0, 900.0]],
                    "tail": {"words": []},
                    "clock_ms": 1500.0, "is_final": True,
                }))
                final = json.loads(await ws.recv())
                assert final["final"] is True
                assert final["committed_text"] == "HELLO THERE"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# prefill glue regression (word boundary lost at the acceptance seam)
# ---------------------------------------------------------------------------


def test_normalize_output_preserves_prefill_boundary_space():
    from alignatt4llm.runtime import CascadeRuntimeConfig, get_translation_variant

    variant = get_translation_variant(CascadeRuntimeConfig())
    # tokenizer decode of the continuation carries a leading space
    glued = variant.normalize_output(
        generated_text=" kausale Encoder",
        assistant_prefill="Der",
        is_partial=True,
    )
    assert glued == "Der kausale Encoder"
    # intra-word continuation (no leading space) must stay glued
    compound = variant.normalize_output(
        generated_text="verarbeitung",
        assistant_prefill="Entwurfs",
        is_partial=True,
    )
    assert compound == "Entwurfsverarbeitung"
    # model restart repeating the accepted suffix is still trimmed
    trimmed = variant.normalize_output(
        generated_text="Der kausale Encoder",
        assistant_prefill="Der kausale",
        is_partial=True,
    )
    assert trimmed == "Der kausale Encoder"
