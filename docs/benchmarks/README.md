# Benchmarks

## MT capture-seam speed benchmark (2026-04-20)

`mt_capture_speed_benchmark_20260420.json` is the artifact behind the paper's
MT-latency-while-monitoring-alignment-heads figure (HF Transformers eager
63.7 ms/token vs vLLM 25.4 ms/token). It is recovered verbatim from the
pre-release research history (`paper/generated/mt_capture_speed_benchmark.json`
at commit `3d108e4`, 2026-04-20); the generating harness was
`paper/build_paper_artifacts.py` at that commit.

Methodology: Gemma-4-E4B-it, en→de, 16 fixed prompts (short/medium/long, with
and without accepted-target prefill), greedy decode, `max_new_tokens=20`,
`gemma_max_model_len=1024`, one unmeasured warmup pass then 3 hot repeats per
seam, each seam run in an isolated subprocess, on a 40 GB A100 with the pinned
vLLM 0.22.1rc1 cu129 nightly. Reported numbers are the median
per-generated-token milliseconds across the 48 measured rows per seam:

| seam | median ms/token |
| --- | --- |
| `transformers_eager` (HF, full attention output) | 63.75 |
| `transformers_qk_fast` (HF, Q/K recompute) | 58.99 |
| `vllm_qk_fast` (vLLM, runtime Q/K capture) | 25.43 |

### Caveat: the vLLM seam was measured with CUDA graphs enabled

The `vllm_qk_fast` seam in this artifact ran with
`mt_vllm_enforce_eager=False` and `mt_vllm_cudagraph_mode="full"`. After this
benchmark was recorded, runtime Q/K capture under cudagraph replay was found
to corrupt the captured payload on a large fraction of chunks (see
`docs/limitations.md`), and the shipped default became
`mt_vllm_enforce_eager=True` on every surface (pinned by
`tests/test_capture_safe_default_invariants.py`). All quality/latency results
in `docs/results.md` were produced in the capture-safe eager configuration.

Consequently the 25.4 ms/token figure is the **cudagraph ceiling** of the
capture path, not the speed of the configuration that produced the quality
numbers. A capture-safe (`enforce_eager=True`) measurement with the same
16-prompt harness is tracked as follow-up work; until it lands, quote the
speedup as "2.5x with full CUDA graphs (capture integrity under repair);
scored runs use the eager configuration".
