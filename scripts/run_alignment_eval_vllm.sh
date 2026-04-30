#!/usr/bin/env bash
# Run alignment.eval with the vLLM uv subproject (see vllm-eval/).
# Falls back to CS148_VLLM_VENV or legacy .venv-vllm if uv / vllm-eval is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_PROJ="${REPO_ROOT}/vllm-eval"

_ssl_cert_fixup() {
  if [[ -n "${SSL_CERT_DIR:-}" && ! -d "${SSL_CERT_DIR}" ]]; then
    unset SSL_CERT_DIR
  fi
}

_export_certifi_from_python() {
  local py="$1"
  if [[ -z "${SSL_CERT_FILE:-}" && -x "${py}" ]]; then
    SSL_CERT_FILE="$("${py}" -c "import certifi; print(certifi.where())" 2>/dev/null || true)"
    [[ -n "${SSL_CERT_FILE}" ]] && export SSL_CERT_FILE
  fi
}

_ssl_cert_fixup
export PYTHONPATH="${REPO_ROOT}"

if [[ -n "${CS148_VLLM_VENV:-}" ]]; then
  PY="${CS148_VLLM_VENV}/bin/python"
  if [[ ! -x "${PY}" ]]; then
    echo "error: CS148_VLLM_VENV=${CS148_VLLM_VENV} has no bin/python" >&2
    exit 1
  fi
  _export_certifi_from_python "${PY}"
  exec "${PY}" -m alignment.eval "$@"
fi

if command -v uv >/dev/null 2>&1 && [[ -f "${VLLM_PROJ}/pyproject.toml" ]]; then
  if [[ -z "${SSL_CERT_FILE:-}" ]]; then
    SSL_CERT_FILE="$(cd "${REPO_ROOT}" && uv run --project "${VLLM_PROJ}" python -c "import certifi; print(certifi.where())")"
    export SSL_CERT_FILE
  fi
  cd "${REPO_ROOT}"
  exec uv run --project "${VLLM_PROJ}" python -m alignment.eval "$@"
fi

LEGACY="${REPO_ROOT}/.venv-vllm/bin/python"
if [[ -x "${LEGACY}" ]]; then
  _export_certifi_from_python "${LEGACY}"
  exec "${LEGACY}" -m alignment.eval "$@"
fi

echo "error: no vLLM environment found." >&2
echo "  Install uv, then from repo root:  uv sync --project vllm-eval" >&2
echo "  Or set CS148_VLLM_VENV to a venv that has vllm installed." >&2
exit 1
