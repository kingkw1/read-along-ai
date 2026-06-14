#!/usr/bin/env bash
set -euo pipefail

SPACE_REPO_ID="${SPACE_REPO_ID:-build-small-hackathon/read-along-ai}"
STAGING_DIR="${STAGING_DIR:-/tmp/read-along-ai-space-upload}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Deploy Read-Along AI Space app}"
INCLUDE_LOCAL_GGUF="${INCLUDE_LOCAL_GGUF:-0}"
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

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

cp README.md LICENSE app.py local_inference.py modal_inference.py requirements.txt packages.txt "${STAGING_DIR}/"
cp -R docs data "${STAGING_DIR}/"
mkdir -p "${STAGING_DIR}/scripts/manual"
cp scripts/upload_gguf_to_hub.sh "${STAGING_DIR}/scripts/"
cp scripts/manual/local_smoke.py "${STAGING_DIR}/scripts/manual/"

if [[ "${INCLUDE_LOCAL_GGUF}" == "1" ]]; then
  Q4_GGUF_PATH="models/gguf/minicpm-phonetic-evaluator-q4_k_m.gguf"
  if [[ ! -f "${Q4_GGUF_PATH}" ]]; then
    echo "Missing required local MiniCPM Q4 GGUF: ${Q4_GGUF_PATH}" >&2
    exit 1
  fi
  mkdir -p "${STAGING_DIR}/models/gguf"
  cp "${Q4_GGUF_PATH}" "${STAGING_DIR}/${Q4_GGUF_PATH}"
else
  echo "Skipping local GGUF in Space payload; local_inference.py will download it from the Hub cache at runtime."
fi

# Keep the public Space payload tight. The app does not need local raw audio,
# processed child-voice clips, notebook outputs, model artifacts, or caches.
rm -rf \
  "${STAGING_DIR}/data/raw_audio" \
  "${STAGING_DIR}/data/processed_audio" \
  "${STAGING_DIR}/data/baseline_results.csv"

echo "Prepared Hugging Face Space payload:"
du -sh "${STAGING_DIR}"
find "${STAGING_DIR}" -maxdepth 4 -type f | sed "s#${STAGING_DIR}/##" | sort

if [[ "${ASSUME_YES}" != "1" ]]; then
  printf "\nUpload this payload to %s? [y/N] " "${SPACE_REPO_ID}"
  read -r answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
fi

"${HF_CLI}" upload "${SPACE_REPO_ID}" "${STAGING_DIR}" . \
  --repo-type space \
  --commit-message "${COMMIT_MESSAGE}"
