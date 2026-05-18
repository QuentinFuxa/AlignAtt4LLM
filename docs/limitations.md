# Known Limitations

Honest constraints of the current implementation, with the evidence behind
them. Code comments and tests that cite "capture corruption" point here.

## CUDA-graph replay corrupts the q/k capture (2026-06-09 evidence)

Under the pinned vLLM 0.22.1rc1 cu129 nightly, running the MT engine with
CUDA graphs (both `cudagraph_mode="full"` and `"piecewise"`) corrupts the
attention observer's captured q/k payload:

- ~58% of streaming chunks produced non-finite provenance rows
  (`alignatt:provenance_nonfinite`), and a further ~12% returned an empty
  observer payload (`alignatt:observer_empty`);
- mechanism: the `alignatt::capture_mt_qk` custom op sits inside graphed
  pieces, and graph replay scatters padded garbage rows into the prompt-K
  buffer;
- consequence: affected draft tokens are withheld by the policy, so a
  corrupted run silently degrades toward "the system never emits". Since
  2026-07 the backend logs an error the first time either signature appears
  in a process.

Mitigation: the MT engine runs eagerly. `mt_vllm_enforce_eager=True` is the
default on every surface (runtime config, presets, backends, CLI runners) and
is pinned by `tests/test_capture_safe_default_invariants.py`. The
`mt_vllm_cudagraph_mode` knob only applies when eager is explicitly disabled,
which is not a supported configuration for scored runs.

Implication for the paper's speed figure: the published 2.5x vLLM-vs-HF-eager
ms/token comparison was recorded with full CUDA graphs *before* this failure
mode was understood, while all quality results ran eagerly. See
[docs/benchmarks/README.md](benchmarks/README.md) for the artifact and the
honest framing. Root-causing the corruption (buffer aliasing across replays
vs. the custom-op mutation contract vs. profiler-phase capture) is open
follow-up work; fixing it would let the speed and quality configurations
coincide.

## vLLM version sensitivity

The capture layer patches vLLM-internal attention classes
(`vllm.model_executor.models.*`) and runs a custom `worker_cls` inside the
engine. These are not stable vLLM APIs:

- validated stacks: the pinned `0.22.1rc1.dev316+g3d119f78f.cu129` nightly
  (installed by `tools/bootstrap/setup_inference_qwen_asr_vllm.sh` from
  vLLM's per-commit wheel index) and `0.23.1rc1` (needs
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` plus adjusted GPU-memory
  splits; see [reproducibility.md](reproducibility.md));
- on other versions, `assert_supported_attention_module` fails loudly if an
  attention class renames the attributes the patched forward reads; behavior
  beyond that check is untested.

## Other constraints

- **Single vLLM worker only.** Tensor parallelism (`tp > 1`) is unsupported;
  the observer buffers and the capture-op registry live in one worker
  process.
- **Grouped-query attention mapping.** The observer assumes the standard
  contiguous query-head → KV-head grouping
  (`map_attention_head_to_key_value_head`); exotic KV layouts need care.
- **Gemma4 KV-shared layers are not capturable.** Layers with
  `is_kv_shared_layer=True` compute no K of their own; the observer rejects
  head selections that include them at configure time. Calibrate heads on
  non-shared layers.
- **Hardware envelope.** Validated on a single 40 GB A100 (Linux, CUDA 12.9,
  Python 3.13). The dual ASR+MT engine layout needs the GPU-memory splits
  documented in [reproducibility.md](reproducibility.md).
- **Head calibration dependencies.** `detect_translation_heads.py` uses an
  LLM word aligner that requires `OPENAI_API_KEY` for new language
  directions, and `google/gemma-4-E4B-it` is a gated model (accept the
  license on Hugging Face before downloading).
