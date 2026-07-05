# Benchmarks

MT latency while monitoring alignment heads: how much does reconstructing
target-to-source attention from captured q/k cost, per generated token, in
each serving stack? Reproduce with:

```bash
.venv-inference/bin/python tools/benchmarks/mt_capture_speed_benchmark.py \
  --vllm-mode cudagraph --output outputs/mt_capture_speed_benchmark.json
```

Methodology (identical across all artifacts here): Gemma-4-E4B-it, en→de, 16
fixed prompts (short/medium/long, with and without accepted-target prefill),
greedy decode, `max_new_tokens=20`, one unmeasured warmup pass then 3 hot
repeats per seam, each seam in an isolated subprocess. Reported number:
median per-generated-token milliseconds over the 48 measured rows per seam.

## Results

Engine-native q/k capture keeps vLLM's speed advantage while the alignment
heads are monitored. The paper reported **2.5x** over HF eager; re-measured
on 2026-07-02 (A100-PCIE-40GB, pinned vLLM 0.22.1rc1), the same harness
reaches **5.35x**:

| measurement | HF eager | HF qk_fast | vLLM capture | speedup vs HF eager |
| --- | --- | --- | --- | --- |
| 2026-07-02, full CUDA graphs — A100-40GB | 72.41 | 60.17 | **13.54** | **5.35x** |
| 2026-04-20, full CUDA graphs (paper figure) | 63.75 | 58.99 | **25.43** | **2.51x** |
| 2026-07-02, eager (capture-safe config) | 72.41 | 60.17 | 49.93 | 1.45x |

Cross-day rows are not directly comparable (different host and torch build);
within a row the comparison is same-day, same-GPU, same-harness.

## Configuration note

The scored quality/latency results in [results.md](../results.md) run the MT
engine with `mt_vllm_enforce_eager=True` (the shipped default, third row):
CUDA-graph replay currently corrupts the captured q/k payload, so graphed
mode is a timing measurement, not a scoring configuration — see
[limitations.md](../limitations.md). Root-causing that replay corruption is
the open engineering item that unlocks the 5x-class row for scored runs.

## Artifacts

- `mt_capture_speed_benchmark_cudagraph_20260702.json` — full-CUDA-graph
  vLLM seam, A100-PCIE-40GB, vLLM `0.22.1rc1.dev316+g3d119f78f`,
  torch 2.11.0+cu129 (5.35x row).
- `mt_capture_speed_benchmark_20260420.json` — the historical artifact
  behind the paper figure, recovered verbatim from pre-release history
  (`paper/generated/` at commit `3d108e4`).
- `mt_capture_speed_benchmark_eager_20260702.json` — capture-safe eager
  measurement (all three seams), same day and hardware as the 5.35x row.

## MT server smoke (A100 40GB, 2026-07-09)

`tools/smoke_mt_server_gpu.py` against a live `alignatt-mt-server`
(`gemma_low_latency`, en-de, eager): timed replay of a 6-sentence stream
where words arrive as unstable tail first and commit 2 s later.
Artifact: `mt_server_smoke_a100_20260709.json`.

- 14 updates, 6 utterance finals, 0 retractions (wire append-only held).
- Partial draft RTT: 730 ms median, 989 ms p95 (16-token drafts, eager).
- Commit fast path (source unchanged, frontier advanced): held target
  tokens released in 0.7 ms with zero MT engine calls, vs 626 ms for the
  preceding draft. This is the release path a WhisperLiveKit client hits
  on every ASR commit when tail-feeding is on.
