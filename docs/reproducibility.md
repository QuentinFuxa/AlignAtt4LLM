# Reproducibility

## Hardware and validated stack

The cascade is validated on a single 40 GB A100 (Linux, CUDA 12.9,
Python 3.13, [uv](https://docs.astral.sh/uv/)). The ASR and MT engines run
side by side on that one GPU.

Two vLLM stacks are known-good:

- **Pinned (paper stack):** `vllm==0.22.1rc1.dev316+g3d119f78f.cu129`,
  installed by the bootstrap script from vLLM's per-commit wheel index (the
  nightly index rotates old builds out; the per-commit index does not). This
  is the stack behind the published results.
- **vLLM 0.23.1rc:** works, but its CUDA-graph memory profiler reserves extra
  memory, so the second engine can fail with
  `No available memory for the cache blocks`. If that happens, set
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` and/or raise
  `--mt-vllm-gpu-memory-utilization` (validated on a 40 GB A100 at
  `--mt-vllm-gpu-memory-utilization 0.72`).

See [limitations.md](limitations.md) for why the MT engine always runs
eagerly (`mt_vllm_enforce_eager=True`).

## Environments

Use separate environments for inference and evaluation (their dependency
sets conflict; both are declared as uv dependency groups in
`pyproject.toml`):

```bash
tools/bootstrap/setup_inference_qwen_asr_vllm.sh
uv venv .venv-evaluation --python 3.13
UV_PROJECT_ENVIRONMENT=.venv-evaluation uv sync --group evaluation
```

The inference bootstrap pins the vLLM/CUDA stack used by this project and
patches the Qwen ASR package for the validated Transformers version.

## Model downloads

The runtime resolves pinned model snapshots from the local Hugging Face
cache and does not download them automatically. Pre-download the models for
the routes you run:

```bash
# ASR route (qwen_forced, the default)
huggingface-cli download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --revision c7cbfc2048c462b0d63a45797104fc9db3ad62b7

# Stable MT route (gemma_vllm_alignatt) - gated: accept the license on
# huggingface.co and `huggingface-cli login` first
huggingface-cli download google/gemma-4-E4B-it --revision 83df0a889143b1dbfc61b591bbc639540fd9ce4c

# EN->ZH research MT route (milmmt_vllm_alignatt)
huggingface-cli download xiaomi-research/MiLMMT-46-4B-v0.1 --revision 1341209df846d7b2f6a077090ca957e28656e3de
```

The reference bring-your-own-LLM route (`qwen_vllm_alignatt`) uses the plain
HF id `Qwen/Qwen3-1.7B`, which vLLM downloads on first use.

To point a route at a different local snapshot, set the matching
`CASCADE_QWEN_ASR_SNAPSHOT`, `CASCADE_QWEN_ALIGNER_SNAPSHOT`,
`CASCADE_GEMMA_SNAPSHOT`, `CASCADE_MILMMT_SNAPSHOT`, or
`CASCADE_QWEN_MT_SNAPSHOT` environment variable.

## Data paths

Calibrated attention-head payloads ship in `data/alignatt_heads/` and are
resolved relative to the repository checkout, so commands work from any
working directory. If the package is installed away from its `data/` tree,
set `ALIGNATT4LLM_DATA_ROOT` to the checkout path. Development audio is not
redistributed with the repo; see [data.md](data.md).

## Smoke Run

```bash
.venv-inference/bin/alignatt-compare --wav <local.wav>
```

## Batch Run

```bash
.venv-inference/bin/alignatt-batch \
  --inputs <local.wav> \
  --target zh \
  --mt-backend-name milmmt_vllm_alignatt \
  --output-dir outputs/milmmt_zh_smoke
```

## Scoring

```bash
.venv-evaluation/bin/alignatt-eval \
  --output-dir outputs/milmmt_zh_smoke
```

Claims should cite the output directory, `manifest.json`, `evaluation.json`,
and the exact command used to produce them. The historical anchors behind
the paper's numbers are recorded in [results.md](results.md), the parsed
official baseline scores in [baselines/](baselines/), and the speed-figure
artifact in [benchmarks/](benchmarks/).

## GPU-free verification

The policy/decision layer is fully exercisable without a GPU: 216 pytest
tests run in about one second against tiny synthetic tensors and recorded
run events.

```bash
UV_PROJECT_ENVIRONMENT=.venv-evaluation uv sync --group evaluation --group dev
.venv-evaluation/bin/python -m pytest -q
```
