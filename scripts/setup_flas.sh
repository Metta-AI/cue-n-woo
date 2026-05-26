#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ or add it to PATH." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv .venv --python 3.11
fi

uv pip install -e .

mkdir -p checkpoints/flas-gemma-2-9b-it
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="flas-ai/flas-gemma-2-9b-it",
    local_dir="checkpoints/flas-gemma-2-9b-it",
)
PY

.venv/bin/python - <<'PY'
import torch
import flas
import transformers
print("flas import ok")
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete. Run ./scripts/run_server.sh"
