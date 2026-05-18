#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${1:-${UV_PROJECT_ENVIRONMENT:-.venv-inference}}"
PYTHON_BIN="$ENV_DIR/bin/python"

cd "$ROOT_DIR"

if [ -d "$ENV_DIR" ]; then
  echo "Reusing existing inference environment at $ENV_DIR"
else
  echo "Creating inference environment at $ENV_DIR"
  uv venv "$ENV_DIR" --python 3.13
fi

echo "Syncing inference dependencies from pyproject.toml"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --group inference

# qwen_asr's published [vllm] extra is pinned to an older torch/vllm stack.
# We install qwen_asr via the inference group, then layer a validated vLLM
# build used by the ASR path on top for this repo's runtime.
echo "Upgrading to a vLLM build compatible with the CUDA 12.9 stack"
# Pin the exact cu129 wheel. An unpinned `-U vllm` resolves to the newest
# version across all indexes, and PyPI stable wheels are CUDA 13 builds that
# fail at runtime with `libcudart.so.13 not found` on this cu129 torch stack
# (observed 2026-06-09 with vllm==0.22.1).
#
# The nightly index (https://wheels.vllm.ai/nightly/cu129/vllm/) rotates old
# builds out (the pin below rotated out by 2026-07-02), so we install from
# vLLM's per-commit index, which is keyed by the full commit SHA embedded in
# the version's `+g<short-sha>` suffix and does not rotate. To move the pin,
# update both VLLM_CU129_VERSION and VLLM_COMMIT_SHA to a matching pair.
VLLM_CU129_VERSION="${VLLM_CU129_VERSION:-0.22.1rc1.dev316+g3d119f78f.cu129}"
VLLM_COMMIT_SHA="${VLLM_COMMIT_SHA:-3d119f78f77cd460c39a1c8ba8303724ad1f88bc}"
uv pip install --python "$PYTHON_BIN" "vllm==${VLLM_CU129_VERSION}" --pre \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT_SHA}/cu129" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

echo "Installing the validated transformers 5.x version"
uv pip install --python "$PYTHON_BIN" transformers==5.5.0

echo "Patching qwen_asr for transformers 5.x compatibility"
"$PYTHON_BIN" tools/bootstrap/patch_qwen_asr_for_transformers5.py

echo "Verifying installation"
"$PYTHON_BIN" -c "import qwen_asr, transformers; print('qwen_asr:', qwen_asr.__file__); print('transformers:', transformers.__version__)"
