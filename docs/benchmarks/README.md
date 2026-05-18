# Benchmarks

MT latency while monitoring alignment heads: how much does reconstructing
target-to-source attention from captured q/k cost, per generated token, in
each serving stack? Reproduce with:

```bash
.venv-inference/bin/python tools/benchmarks/mt_capture_speed_benchmark.py \
  --vllm-mode eager --output outputs/mt_capture_speed_benchmark.json
```

Methodology (identical across all artifacts here): Gemma-4-E4B-it, en→de, 16
fixed prompts (short/medium/long, with and without accepted-target prefill),
greedy decode, `max_new_tokens=20`, one unmeasured warmup pass then 3 hot
repeats per seam, each seam in an isolated subprocess. Reported number:
median per-generated-token milliseconds over the 48 measured rows per seam.

## The honest speed story (two numbers)

| measurement | HF eager | HF qk_fast | vLLM capture | speedup vs HF eager |
| --- | --- | --- | --- | --- |
| 2026-07-02, **eager (capture-safe)** — A100-40GB, vLLM 0.22.1rc1 | 72.41 | 60.17 | **49.93** | **1.45x** |
| 2026-04-20, **full CUDA graphs** (historical paper figure) | 63.75 | 58.99 | **25.43** | **2.51x** |

- The **eager** row is the configuration that produced every scored result in
  [results.md](../results.md): `mt_vllm_enforce_eager=True`, the shipped
  default, because CUDA-graph replay corrupts the q/k capture (see
  [limitations.md](../limitations.md)). The 1.45x is a same-day, same-GPU,
  same-harness eager-vs-eager comparison and is the number to quote for the
  capture-safe system.
- The **cudagraph** row is the paper's published figure. It is real, but it
  is the *ceiling* of the capture path: the configuration it measures is not
  usable for scored runs until the replay corruption is fixed. Cross-day
  rows are not directly comparable (different host, torch build).
- Root-causing the cudagraph corruption is the open engineering item that
  would let the 2.5x-class speed and the scored quality coexist in one
  configuration.

## Artifacts

- `mt_capture_speed_benchmark_eager_20260702.json` — capture-safe eager
  measurement (all three seams), A100-PCIE-40GB, vLLM
  `0.22.1rc1.dev316+g3d119f78f`, torch 2.11.0+cu129; produced with the
  command above on a fresh checkout.
- `mt_capture_speed_benchmark_cudagraph_20260702.json` — same-day vLLM seam
  re-run with `--vllm-mode cudagraph` (timing only; capture payload is
  corrupt in this mode).
- `mt_capture_speed_benchmark_20260420.json` — the historical artifact
  behind the paper figure, recovered verbatim from pre-release history
  (`paper/generated/` at commit `3d108e4`); vLLM seam ran with
  `mt_vllm_enforce_eager=False, cudagraph_mode="full"`.
