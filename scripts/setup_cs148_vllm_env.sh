#!/usr/bin/env bash
# Install the vLLM evaluation stack via uv (declarative deps in vllm-eval/pyproject.toml).
# Do not use /tmp for the venv — use this repo + shared home so Slurm node changes still work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_PROJ="${REPO_ROOT}/vllm-eval"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "${1:-}" == "--recreate" ]]; then
  echo "Removing vllm-eval/.venv"
  rm -rf "${VLLM_PROJ}/.venv"
  shift
fi

cd "${REPO_ROOT}"
echo "Syncing uv project: ${VLLM_PROJ}"
uv sync --project "${VLLM_PROJ}" "$@"

echo
echo "Done. From repo root run e.g.:"
echo "  bash scripts/run_alignment_eval_vllm.sh --split train --limit 1 --use-cot"
echo "or:"
echo "  export PYTHONPATH=\"\${PWD}\""
echo "  uv run --project vllm-eval python -m alignment.eval --help"
