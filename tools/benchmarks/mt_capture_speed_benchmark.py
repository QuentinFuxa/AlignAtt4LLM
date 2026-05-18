#!/usr/bin/env python3
"""MT capture-seam speed benchmark.

Measures the per-prompt and per-generated-token latency of the three MT
attention-capture seams on a fixed 16-prompt en->de suite (short/medium/long,
with and without an accepted-target prefill), greedy decode, Gemma-4-E4B-it:

- ``transformers_eager``: HF Transformers reference with ``output_attentions``
  and a selected-head attention recorder (full attention output).
- ``transformers_qk_fast``: HF Transformers reference that reconstructs the
  selected source rows from captured layer inputs plus the KV cache
  (Q/K recompute, no full attention output).
- ``vllm_qk_fast``: the shipped vLLM backend with engine-native runtime Q/K
  capture (``backend.translate`` end to end).

Each seam runs in an isolated subprocess (one unmeasured warmup pass, then
``BENCHMARK_REPEATS`` hot repeats over the suite) so engine state never leaks
across seams. The parent aggregates per-seam medians/means, per-length-bin and
per-spec breakdowns, and cross-seam draft/stop-reason divergences into a single
JSON artifact shaped like ``docs/benchmarks/mt_capture_speed_benchmark_20260420.json``
plus a top-level ``config`` block recording the measurement conditions.

Run from the repo root on a GPU machine with the ``.venv-inference``
environment:

    .venv-inference/bin/python tools/benchmarks/mt_capture_speed_benchmark.py \\
        --seams transformers_eager,transformers_qk_fast,vllm_qk_fast \\
        --vllm-mode eager \\
        --output outputs/mt_capture_speed_benchmark.json

``--vllm-mode eager`` (the default) is the capture-safe configuration that
matches all scored quality/latency runs (``mt_vllm_enforce_eager=True``).
``--vllm-mode cudagraph`` reproduces the historical 2026-04-20 measurement
conditions (``mt_vllm_enforce_eager=False``, ``mt_vllm_cudagraph_mode="full"``);
see docs/benchmarks/README.md for why that mode is a ceiling measurement with
known capture-integrity issues.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_JSON_BEGIN = "__BENCHMARK_JSON_BEGIN__"
BENCHMARK_JSON_END = "__BENCHMARK_JSON_END__"
BENCHMARK_REPEATS = 3
KNOWN_SEAMS = ("transformers_eager", "transformers_qk_fast", "vllm_qk_fast")

BENCHMARK_SUITE = [
    {
        "id": "short_01",
        "length_bin": "short",
        "source_text": "the model gives itself more context before it answers",
        "assistant_prefill": "",
        "accessible_units": 6,
    },
    {
        "id": "short_01_prefill",
        "length_bin": "short",
        "source_text": "the model gives itself more context before it answers",
        "assistant_prefill": "Das Modell gibt sich",
        "accessible_units": 6,
    },
    {
        "id": "short_02",
        "length_bin": "short",
        "source_text": "we carefully keep only the useful rows",
        "assistant_prefill": "",
        "accessible_units": 5,
    },
    {
        "id": "short_02_prefill",
        "length_bin": "short",
        "source_text": "we carefully keep only the useful rows",
        "assistant_prefill": "Wir behalten nur",
        "accessible_units": 5,
    },
    {
        "id": "short_03",
        "length_bin": "short",
        "source_text": "the policy now rejects the risky word",
        "assistant_prefill": "",
        "accessible_units": 5,
    },
    {
        "id": "short_03_prefill",
        "length_bin": "short",
        "source_text": "the policy now rejects the risky word",
        "assistant_prefill": "Die Policy weist",
        "accessible_units": 5,
    },
    {
        "id": "medium_01",
        "length_bin": "medium",
        "source_text": "we selectively reconstruct only the source slice that drives the acceptance rule",
        "assistant_prefill": "",
        "accessible_units": 9,
    },
    {
        "id": "medium_01_prefill",
        "length_bin": "medium",
        "source_text": "we selectively reconstruct only the source slice that drives the acceptance rule",
        "assistant_prefill": "Wir rekonstruieren selektiv nur",
        "accessible_units": 9,
    },
    {
        "id": "medium_02",
        "length_bin": "medium",
        "source_text": "the observer returns a compact and engine native capture of query and key tensors",
        "assistant_prefill": "",
        "accessible_units": 10,
    },
    {
        "id": "medium_02_prefill",
        "length_bin": "medium",
        "source_text": "the observer returns a compact and engine native capture of query and key tensors",
        "assistant_prefill": "Der Beobachter liefert eine",
        "accessible_units": 10,
    },
    {
        "id": "medium_03",
        "length_bin": "medium",
        "source_text": "we explicitly compare only the seams that expose a real capture path",
        "assistant_prefill": "",
        "accessible_units": 8,
    },
    {
        "id": "medium_03_prefill",
        "length_bin": "medium",
        "source_text": "we explicitly compare only the seams that expose a real capture path",
        "assistant_prefill": "Wir vergleichen ausdrücklich nur",
        "accessible_units": 8,
    },
    {
        "id": "long_01",
        "length_bin": "long",
        "source_text": "the paper reports a clear and reproducible comparison of the three capture seams under compiled inference",
        "assistant_prefill": "",
        "accessible_units": 11,
    },
    {
        "id": "long_01_prefill",
        "length_bin": "long",
        "source_text": "the paper reports a clear and reproducible comparison of the three capture seams under compiled inference",
        "assistant_prefill": "Der Beitrag berichtet über einen",
        "accessible_units": 11,
    },
    {
        "id": "long_02",
        "length_bin": "long",
        "source_text": "we intentionally place the accepted prefix before we continue the translation with a new draft",
        "assistant_prefill": "",
        "accessible_units": 11,
    },
    {
        "id": "long_02_prefill",
        "length_bin": "long",
        "source_text": "we intentionally place the accepted prefix before we continue the translation with a new draft",
        "assistant_prefill": "Wir stellen das akzeptierte Präfix",
        "accessible_units": 11,
    },
]


def ensure_repo_imports() -> None:
    src_root = str(REPO_ROOT / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def word_count(text: str) -> int:
    return len([piece for piece in text.strip().split() if piece])


def stable_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def german_heads_path() -> Path:
    ensure_repo_imports()
    from alignatt4llm.runtime import alignatt_heads_path_for

    return Path(alignatt_heads_path_for("English", "German"))


def build_runtime_config(
    *,
    target_lang: str,
    heads_path: Path,
    max_new_tokens: int = 16,
    vllm_mode: str = "eager",
    gpu_memory_utilization: float = 0.5,
) -> Any:
    ensure_repo_imports()
    from alignatt4llm.runtime import CascadeRuntimeConfig, temporary_runtime_config

    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang=target_lang,
        translation_alignatt_heads_path=str(heads_path),
        translation_alignatt_top_k_heads=8,
        translation_alignatt_filter_width=7,
        translation_alignatt_probe_mode="qk_fast",
        translation_alignatt_inaccessible_ms=0.0,
        translation_alignatt_border_margin=0,
        translation_alignatt_min_source_mass=0.0,
        partial_max_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.0,
        # ``mt_vllm_cudagraph_mode`` only takes effect when eager mode is
        # explicitly disabled, so "full" is inert under --vllm-mode eager and
        # reproduces the historical 2026-04-20 conditions under cudagraph.
        mt_vllm_enforce_eager=(vllm_mode == "eager"),
        mt_vllm_enable_prefix_caching=False,
        mt_vllm_cudagraph_mode="full",
        mt_vllm_gpu_memory_utilization=gpu_memory_utilization,
        gemma_max_model_len=1024,
    )
    return config, temporary_runtime_config


def build_frontier_with_accessible_units(source_text: str, accessible_units: int) -> Any:
    ensure_repo_imports()
    from alignatt4llm.source_frontier import build_source_accessibility_frontier

    total_units = word_count(source_text)
    unit_ms = 320.0
    timestamps = [
        (idx * unit_ms, (idx + 1) * unit_ms)
        for idx in range(total_units)
    ]
    current_audio_ms = max(0.0, float(accessible_units) * unit_ms)
    return build_source_accessibility_frontier(
        source_text=source_text,
        word_timestamps_ms=timestamps,
        current_audio_ms=current_audio_ms,
        inaccessible_ms=0.0,
        is_final=False,
    )


@dataclass
class ProbeExecution:
    target_lang: str
    heads_path: Path
    backend: Any
    variant: Any


def load_vllm_execution(
    target_lang: str,
    max_new_tokens: int = 16,
    *,
    vllm_mode: str = "eager",
    gpu_memory_utilization: float = 0.5,
) -> ProbeExecution:
    ensure_repo_imports()
    from alignatt4llm.mt.base import build_mt_backend
    from alignatt4llm.runtime import gemma_model_name
    from alignatt4llm.translation_variants import TRANSLATION_VARIANTS, DEFAULT_TRANSLATION_VARIANT_ID

    heads_path = german_heads_path()
    config, temp_cfg = build_runtime_config(
        target_lang=target_lang,
        heads_path=heads_path,
        max_new_tokens=max_new_tokens,
        vllm_mode=vllm_mode,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    with temp_cfg(
        config,
        source_lang="English",
        target_lang=target_lang,
        translation_alignatt_heads_path=str(heads_path),
    ):
        backend = build_mt_backend(model_name=gemma_model_name, runtime_config=config)
        backend.load()
    variant = TRANSLATION_VARIANTS[DEFAULT_TRANSLATION_VARIANT_ID]
    return ProbeExecution(target_lang=target_lang, heads_path=heads_path, backend=backend, variant=variant)


def snapshot_prompt_kv_from_past_key_values(past_key_values) -> list[tuple[int, Any, Any, int]]:
    snapshot: list[tuple[int, Any, Any, int]] = []
    if past_key_values is None:
        return snapshot
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx, (key, value) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            seq_length = int(key.shape[2])
            snapshot.append((int(layer_idx), key.detach(), value.detach(), seq_length))
        return snapshot
    if isinstance(past_key_values, (list, tuple)):
        for layer_idx, layer_kv in enumerate(past_key_values):
            if not isinstance(layer_kv, (list, tuple)) or len(layer_kv) < 2:
                continue
            key, value = layer_kv[:2]
            seq_length = int(key.shape[2])
            snapshot.append((int(layer_idx), key.detach(), value.detach(), seq_length))
    return snapshot


def prompt_for_benchmark(spec: dict[str, Any], *, target_lang: str) -> Any:
    ensure_repo_imports()
    from alignatt4llm.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    frontier = build_frontier_with_accessible_units(spec["source_text"], int(spec["accessible_units"]))
    return ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
        source_lang="English",
        target_lang=target_lang,
        text=spec["source_text"],
        source_frontier=frontier,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill=str(spec["assistant_prefill"]),
    )


class TransformersReferenceRunner:
    def __init__(self, *, heads_path: Path, attn_impl: str, seam_name: str, max_new_tokens: int) -> None:
        ensure_repo_imports()
        from alignatt4llm.mt.base import AlignAttDecoderPolicy, BaseMTBackend, load_alignatt_heads
        from alignatt4llm.runtime import gemma_model_name

        class _PromptOnlyBackend(BaseMTBackend):
            def load(self) -> None:
                raise NotImplementedError

            def translate(self, *, rendered_prompt, variant, is_partial, prompt_cache_state=None):
                raise NotImplementedError

        self._backend = _PromptOnlyBackend(model_name=gemma_model_name, runtime_config=type("Cfg", (), {
            "translation_alignatt_filter_width": 7,
            "translation_alignatt_border_margin": 0,
            "gemma_max_model_len": 1024,
            "partial_translation_min_new_tokens": 4,
            "partial_translation_token_budget_ratio": 1.0,
            "partial_translation_token_budget_buffer": 8,
            "partial_max_new_tokens": max_new_tokens,
            "max_new_tokens": max_new_tokens,
            "translation_min_new_tokens": 4,
            "translation_token_budget_ratio": 1.0,
            "translation_token_budget_buffer": 8,
            "translation_generation_margin": 8,
            "repetition_penalty": 1.0,
        })())
        self.model_name = gemma_model_name
        self.attn_impl = attn_impl
        self.seam_name = seam_name
        self.max_new_tokens = int(max_new_tokens)
        self.alignatt_heads = load_alignatt_heads(str(heads_path), top_k=8)
        self.policy = None
        self.model = None
        self.tokenizer = None
        self._AlignAttDecoderPolicy = AlignAttDecoderPolicy

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
        self._backend.tokenizer = self.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype="auto",
            device_map="cuda:0",
            attn_implementation=self.attn_impl,
        )
        self.model.eval()
        self.policy = self._AlignAttDecoderPolicy(tokenizer=self.tokenizer, runtime_config=self._backend.runtime_config)

    def render_prompt_package(self, rendered_prompt):
        return self._backend.render_prompt_package(rendered_prompt)

    def resolve_generation_stop_token_ids(self) -> tuple[int, ...]:
        return self._backend.resolve_generation_stop_token_ids()

    def compute_max_tokens(self, *, prompt_tokens: int, source_text: str, assistant_prefill: str) -> int:
        return self._backend.compute_max_tokens(
            prompt_tokens=prompt_tokens,
            source_text=source_text,
            is_partial=True,
            assistant_prefill=assistant_prefill,
        )

    def decode_candidate_text(self, *, generated_ids: list[int], assistant_prefill: str, variant) -> str:
        return self._backend.decode_candidate_text(
            generated_ids=generated_ids,
            assistant_prefill=assistant_prefill,
            variant=variant,
            is_partial=True,
        )

    def run_prompt(self, rendered_prompt, variant) -> dict[str, Any]:
        import torch
        from alignatt4llm.mt.base import (
            SelectedAttentionRecorder,
            SelectedLayerInputRecorder,
            extract_source_attention_rows_per_token,
            extract_source_attention_rows_per_token_from_fast_path,
        )

        prompt_package = self.render_prompt_package(rendered_prompt)
        source_map = prompt_package.source_map
        if source_map is None:
            raise RuntimeError("Benchmark prompt package did not produce a source map.")
        prompt_ids = list(prompt_package.prompt_token_ids)
        stop_ids = set(self.resolve_generation_stop_token_ids())
        max_new_tokens = self.compute_max_tokens(
            prompt_tokens=len(prompt_ids),
            source_text=rendered_prompt.source_text,
            assistant_prefill=rendered_prompt.assistant_prefill,
        )

        attention_recorder = None
        layer_input_recorder = None
        if self.seam_name == "transformers_eager":
            attention_recorder = SelectedAttentionRecorder(model=self.model, alignatt_heads=self.alignatt_heads)
        else:
            layer_input_recorder = SelectedLayerInputRecorder(model=self.model, alignatt_heads=self.alignatt_heads)

        input_ids = torch.tensor([prompt_ids], device=self.model.device, dtype=torch.long)
        with torch.no_grad():
            prompt_outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
        past = prompt_outputs.past_key_values
        prompt_snapshot = snapshot_prompt_kv_from_past_key_values(past)
        next_token = int(torch.argmax(prompt_outputs.logits[0, -1]).item())

        generated_ids: list[int] = []
        captured_rows = 0
        row_counts: list[int] = []
        total_start = perf_counter()
        step_times_ms: list[float] = []
        stop_reason = "max_new_tokens"
        while len(generated_ids) < max_new_tokens:
            generated_ids.append(next_token)
            current_input = torch.tensor([[next_token]], device=self.model.device, dtype=torch.long)
            step_start = perf_counter()
            if self.seam_name == "transformers_eager":
                assert attention_recorder is not None
                with attention_recorder.capture() as captured_attn, torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                        output_attentions=True,
                    )
                rows = extract_source_attention_rows_per_token(
                    layer_attentions_by_layer=captured_attn,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=source_map.source_token_positions,
                )
            else:
                assert layer_input_recorder is not None
                with layer_input_recorder.capture() as captured_inputs, torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                        output_attentions=False,
                    )
                rows, _ = extract_source_attention_rows_per_token_from_fast_path(
                    layer_inputs_by_layer=captured_inputs,
                    prompt_kv_snapshot=prompt_snapshot,
                    runtime_past_key_values=outputs.past_key_values,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=source_map.source_token_positions,
                    accessible_source_token_count=source_map.accessible_source_token_count,
                )
            step_times_ms.append((perf_counter() - step_start) * 1000.0)
            row_counts.append(len(rows))
            if rows:
                captured_rows += len(rows)
            past = outputs.past_key_values
            next_token = int(torch.argmax(outputs.logits[0, -1]).item())
            if next_token in stop_ids:
                generated_ids.append(next_token)
                stop_reason = "stop_token"
                break

        special_ids = set(int(token_id) for token_id in getattr(self.tokenizer, "all_special_ids", []) or [])
        trimmed_ids = list(generated_ids)
        while trimmed_ids and trimmed_ids[-1] in special_ids:
            trimmed_ids.pop()
        total_ms = (perf_counter() - total_start) * 1000.0
        draft_text = self.decode_candidate_text(
            generated_ids=trimmed_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
        )
        return {
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(trimmed_ids),
            "draft_text": draft_text,
            "stop_reason": stop_reason,
            "total_ms": total_ms,
            "median_step_ms": statistics.median(step_times_ms) if step_times_ms else 0.0,
            "max_step_ms": max(step_times_ms) if step_times_ms else 0.0,
            "row_counts": row_counts,
            "captured_rows": captured_rows,
        }


def run_vllm_benchmark_worker(
    specs: list[dict[str, Any]],
    *,
    vllm_mode: str,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    ensure_repo_imports()
    from alignatt4llm.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    execution = load_vllm_execution(
        "German",
        max_new_tokens=20,
        vllm_mode=vllm_mode,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    backend = execution.backend
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    results = []
    if specs:
        warmup_rendered = prompt_for_benchmark(specs[0], target_lang="German")
        backend.translate(rendered_prompt=warmup_rendered, variant=variant, is_partial=True)
    for repeat_index in range(BENCHMARK_REPEATS):
        for spec in specs:
            rendered = prompt_for_benchmark(spec, target_lang="German")
            result = backend.translate(rendered_prompt=rendered, variant=variant, is_partial=True)
            timings = result.timings_ms or {}
            generated_token_count = len(result.draft_generated_token_ids)
            total_ms = float(timings.get("total", 0.0))
            results.append(
                {
                    "id": spec["id"],
                    "repeat_index": repeat_index,
                    "length_bin": spec["length_bin"],
                    "assistant_prefill": spec["assistant_prefill"],
                    "prompt_num_tokens": result.prompt_num_tokens,
                    "generated_token_count": generated_token_count,
                    "draft_text": result.draft_text,
                    "acceptance_text": result.acceptance_text,
                    "stop_reason": result.stop_reason,
                    "total_ms": total_ms,
                    "per_generated_token_ms": total_ms / max(1, generated_token_count),
                    "generate_ms": float(timings.get("generate", 0.0)),
                    "prepare_observer_ms": float(timings.get("prepare_observer", 0.0)),
                    "fetch_observer_ms": float(timings.get("fetch_observer", 0.0)),
                    "reconstruct_ms": float(timings.get("reconstruct", 0.0)),
                }
            )
    return {"seam": "vllm_qk_fast", "results": results}


def run_transformers_benchmark_worker(seam: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_repo_imports()
    from alignatt4llm.translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT

    runner = TransformersReferenceRunner(
        heads_path=german_heads_path(),
        attn_impl="eager" if seam == "transformers_eager" else "sdpa",
        seam_name=seam,
        max_new_tokens=20,
    )
    runner.load()
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    if specs:
        warmup_rendered = prompt_for_benchmark(specs[0], target_lang="German")
        runner.run_prompt(warmup_rendered, variant)
    results = []
    for repeat_index in range(BENCHMARK_REPEATS):
        for spec in specs:
            rendered = prompt_for_benchmark(spec, target_lang="German")
            prompt_result = runner.run_prompt(rendered, variant)
            total_ms = float(prompt_result["total_ms"])
            generated_token_count = int(prompt_result["generated_token_count"])
            results.append(
                {
                    "id": spec["id"],
                    "repeat_index": repeat_index,
                    "length_bin": spec["length_bin"],
                    "assistant_prefill": spec["assistant_prefill"],
                    "per_generated_token_ms": total_ms / max(1, generated_token_count),
                    **prompt_result,
                }
            )
    return {"seam": seam, "results": results}


def benchmark_worker_main() -> None:
    payload = json.loads(sys.stdin.read())
    seam = payload["seam"]
    specs = payload["specs"]
    if seam == "vllm_qk_fast":
        response = run_vllm_benchmark_worker(
            specs,
            vllm_mode=str(payload.get("vllm_mode", "eager")),
            gpu_memory_utilization=float(payload.get("mt_gpu_memory_utilization", 0.5)),
        )
    else:
        response = run_transformers_benchmark_worker(seam, specs)
    sys.stdout.write(
        f"{BENCHMARK_JSON_BEGIN}{json.dumps(response)}{BENCHMARK_JSON_END}"
    )


def extract_benchmark_json(stdout_text: str) -> dict[str, Any]:
    start = stdout_text.rfind(BENCHMARK_JSON_BEGIN)
    end = stdout_text.rfind(BENCHMARK_JSON_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(
            "Benchmark worker did not return a framed JSON payload.\n"
            f"stdout tail:\n{stdout_text[-4000:]}"
        )
    payload = stdout_text[start + len(BENCHMARK_JSON_BEGIN):end]
    return json.loads(payload)


def dominant_value(values: list[Any]) -> Any:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def summarize_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_ms = [float(row["total_ms"]) for row in rows]
    per_token_ms = [float(row["per_generated_token_ms"]) for row in rows]
    by_length: dict[str, dict[str, float]] = {}
    grouped_by_length: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_spec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_length[str(row["length_bin"])].append(row)
        grouped_by_spec[str(row["id"])].append(row)
    for length_bin, group in grouped_by_length.items():
        length_total_ms = [float(row["total_ms"]) for row in group]
        length_per_token_ms = [float(row["per_generated_token_ms"]) for row in group]
        by_length[length_bin] = {
            "median_total_ms": statistics.median(length_total_ms) if length_total_ms else 0.0,
            "median_per_generated_token_ms": statistics.median(length_per_token_ms) if length_per_token_ms else 0.0,
            "row_count": len(group),
        }
    by_spec = {
        spec_id: {
            "modal_draft_text": dominant_value([str(row["draft_text"]) for row in group]),
            "modal_stop_reason": dominant_value([str(row.get("stop_reason")) for row in group]),
            "median_total_ms": statistics.median([float(row["total_ms"]) for row in group]),
            "median_per_generated_token_ms": statistics.median(
                [float(row["per_generated_token_ms"]) for row in group]
            ),
        }
        for spec_id, group in grouped_by_spec.items()
    }
    return {
        "median_total_ms": statistics.median(total_ms) if total_ms else 0.0,
        "mean_total_ms": stable_mean(total_ms),
        "max_total_ms": max(total_ms) if total_ms else 0.0,
        "median_per_generated_token_ms": statistics.median(per_token_ms) if per_token_ms else 0.0,
        "mean_per_generated_token_ms": stable_mean(per_token_ms),
        "by_length_bin": by_length,
        "by_spec": by_spec,
        "rows": rows,
    }


def summarize_benchmark(results_by_seam: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"seams": {}, "divergences": []}
    all_spec_ids: set[str] = set()
    for seam, payload in results_by_seam.items():
        seam_summary = summarize_benchmark_rows(payload["results"])
        summary["seams"][seam] = seam_summary
        all_spec_ids.update(seam_summary["by_spec"].keys())

    for spec_id in sorted(all_spec_ids):
        drafts_by_seam: dict[str, Any] = {}
        stops_by_seam: dict[str, Any] = {}
        for seam, seam_summary in summary["seams"].items():
            by_spec = seam_summary["by_spec"]
            if spec_id not in by_spec:
                continue
            drafts_by_seam[seam] = by_spec[spec_id]["modal_draft_text"]
            stops_by_seam[seam] = by_spec[spec_id]["modal_stop_reason"]
        if len(set(drafts_by_seam.values())) > 1 or len(set(stops_by_seam.values())) > 1:
            summary["divergences"].append(
                {
                    "id": spec_id,
                    "drafts_by_seam": drafts_by_seam,
                    "stop_reasons_by_seam": stops_by_seam,
                }
            )
    return summary


def benchmark_config_metadata(*, vllm_mode: str, seams_run: list[str]) -> dict[str, Any]:
    ensure_repo_imports()
    import torch
    from alignatt4llm.runtime import gemma_model_name

    try:
        import vllm

        vllm_version = str(getattr(vllm, "__version__", "unknown"))
    except Exception:
        vllm_version = "unavailable"
    return {
        "vllm_mode": vllm_mode,
        "seams_run": list(seams_run),
        "model": Path(gemma_model_name).name,
        "vllm_version": vllm_version,
        "torch_version": str(torch.__version__),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_benchmark_speed(
    *,
    seams: list[str],
    vllm_mode: str,
    mt_gpu_memory_utilization: float,
    output_path: Path,
) -> None:
    results_by_seam: dict[str, dict[str, Any]] = {}
    for seam in seams:
        payload = {
            "seam": seam,
            "specs": BENCHMARK_SUITE,
            "vllm_mode": vllm_mode,
            "mt_gpu_memory_utilization": mt_gpu_memory_utilization,
        }
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            cwd=str(REPO_ROOT),
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr.decode("utf-8"))
        results_by_seam[seam] = extract_benchmark_json(
            proc.stdout.decode("utf-8", errors="replace")
        )
    summary = summarize_benchmark(results_by_seam)
    summary["config"] = benchmark_config_metadata(vllm_mode=vllm_mode, seams_run=seams)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stderr.write(f"Wrote benchmark summary to {output_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seams",
        default=",".join(KNOWN_SEAMS),
        help=(
            "Comma-separated list of seams to benchmark "
            f"(default: {','.join(KNOWN_SEAMS)})."
        ),
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("eager", "cudagraph"),
        default="eager",
        help=(
            "vLLM engine mode for the vllm_qk_fast seam: 'eager' is the "
            "capture-safe mode matching all scored runs; 'cudagraph' "
            "reproduces the historical 2026-04-20 measurement conditions "
            "(default: eager)."
        ),
    )
    parser.add_argument(
        "--mt-gpu-memory-utilization",
        type=float,
        default=0.5,
        help="vLLM GPU memory utilization for the MT engine (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the aggregated benchmark JSON (required in parent mode).",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=(
            "Internal: read a framed JSON job from stdin, run one seam, and "
            "print framed JSON to stdout."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        benchmark_worker_main()
        return
    if args.output is None:
        raise SystemExit("--output is required (parent mode writes the aggregated benchmark JSON there).")
    seams = [seam.strip() for seam in str(args.seams).split(",") if seam.strip()]
    unknown = [seam for seam in seams if seam not in KNOWN_SEAMS]
    if unknown:
        raise SystemExit(
            f"Unknown seams: {', '.join(unknown)}. Choose from: {', '.join(KNOWN_SEAMS)}."
        )
    if not seams:
        raise SystemExit("--seams resolved to an empty list.")
    run_benchmark_speed(
        seams=seams,
        vllm_mode=args.vllm_mode,
        mt_gpu_memory_utilization=args.mt_gpu_memory_utilization,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
