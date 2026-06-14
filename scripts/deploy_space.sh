#!/usr/bin/env bash
set -euo pipefail

SPACE_REPO_ID="${SPACE_REPO_ID:-build-small-hackathon/read-along-ai}"
STAGING_DIR="${STAGING_DIR:-/tmp/read-along-ai-space-upload}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Deploy Read-Along AI Space app}"
ASSUME_YES="0"

if [[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]]; then
  ASSUME_YES="1"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install huggingface_hub or activate the project environment." >&2
  exit 1
fi

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

cp README.md LICENSE app.py local_inference.py modal_inference.py requirements.txt packages.txt "${STAGING_DIR}/"
cp -R docs data "${STAGING_DIR}/"

Q4_GGUF_PATH="models/gguf/minicpm-phonetic-evaluator-q4_k_m.gguf"
if [[ ! -f "${Q4_GGUF_PATH}" ]]; then
  echo "Missing required local MiniCPM Q4 GGUF: ${Q4_GGUF_PATH}" >&2
  exit 1
fi
mkdir -p "${STAGING_DIR}/models/gguf"
cp "${Q4_GGUF_PATH}" "${STAGING_DIR}/${Q4_GGUF_PATH}"

# Keep the public Space payload tight. The app does not need local raw audio,
# processed child-voice clips, notebook outputs, f16 model artifacts, or caches.
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

hf upload "${SPACE_REPO_ID}" "${STAGING_DIR}" . \
  --repo-type space \
  --commit-message "${COMMIT_MESSAGE}"
