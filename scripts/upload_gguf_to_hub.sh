#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO_ID="${MODEL_REPO_ID:-kingkw1/minicpm-phonetic-evaluator}"
GGUF_PATH="${GGUF_PATH:-models/gguf/minicpm-phonetic-evaluator-q4_k_m.gguf}"
DEST_PATH="${DEST_PATH:-minicpm-phonetic-evaluator-q4_k_m.gguf}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload MiniCPM phonetic evaluator Q4 GGUF}"
ASSUME_YES="0"

if [[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]]; then
  ASSUME_YES="1"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

HF_CLI="${HF_CLI:-hf}"
if [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
  HF_CLI="${REPO_ROOT}/.venv/bin/hf"
fi

if ! command -v "${HF_CLI}" >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install huggingface_hub or activate the project environment." >&2
  exit 1
fi

if [[ ! -f "${GGUF_PATH}" ]]; then
  echo "Missing GGUF file: ${GGUF_PATH}" >&2
  exit 1
fi

echo "Prepared GGUF upload:"
du -h "${GGUF_PATH}"
echo "Destination: ${MODEL_REPO_ID}/${DEST_PATH}"

if [[ "${ASSUME_YES}" != "1" ]]; then
  printf "\nUpload this GGUF to %s? [y/N] " "${MODEL_REPO_ID}"
  read -r answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

"${HF_CLI}" upload "${MODEL_REPO_ID}" "${GGUF_PATH}" "${DEST_PATH}" \
  --repo-type model \
  --commit-message "${COMMIT_MESSAGE}"
