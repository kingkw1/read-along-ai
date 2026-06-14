#!/usr/bin/env bash
set -euo pipefail

# Common workflows:
#   Fast UI/CSS iteration Space:
#     COMMIT_MESSAGE="UI polish" ./scripts/deploy_space.sh --target ui --yes
#
#   Main submission Space with full local Off the Grid dependencies:
#     COMMIT_MESSAGE="Enable local Off the Grid inference" ./scripts/deploy_space.sh --target main --yes
#
#   Preview a payload before uploading:
#     ./scripts/deploy_space.sh --target ui
#     ./scripts/deploy_space.sh --target main
#
MAIN_SPACE_REPO_ID="${MAIN_SPACE_REPO_ID:-build-small-hackathon/read-along-ai}"
UI_SPACE_REPO_ID="${UI_SPACE_REPO_ID:-build-small-hackathon/read-along-ai-ui}"
DEPLOY_TARGET="${DEPLOY_TARGET:-main}"
SPACE_REPO_ID="${SPACE_REPO_ID:-}"
STAGING_DIR="${STAGING_DIR:-/tmp/read-along-ai-space-upload}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Deploy Read-Along AI Space app}"
SPACE_PROFILE="${SPACE_PROFILE:-}"
INCLUDE_LOCAL_GGUF="${INCLUDE_LOCAL_GGUF:-0}"
ASSUME_YES="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)
      ASSUME_YES="1"
      shift
      ;;
    --target)
      DEPLOY_TARGET="${2:-}"
      if [[ -z "${DEPLOY_TARGET}" ]]; then
        echo "--target requires a value: main or ui" >&2
        exit 1
      fi
      shift 2
      ;;
    --profile)
      SPACE_PROFILE="${2:-}"
      if [[ -z "${SPACE_PROFILE}" ]]; then
        echo "--profile requires a value: local or ui" >&2
        exit 1
      fi
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: ./scripts/deploy_space.sh [--target main|ui] [--profile local|ui] [--yes]

Targets:
  main  build-small-hackathon/read-along-ai     default, full local-inference profile
  ui    build-small-hackathon/read-along-ai-ui  fast UI profile

Environment overrides:
  MAIN_SPACE_REPO_ID, UI_SPACE_REPO_ID, SPACE_REPO_ID, SPACE_PROFILE,
  COMMIT_MESSAGE, STAGING_DIR, INCLUDE_LOCAL_GGUF, HF_CLI

Examples:
  ./scripts/deploy_space.sh --target ui --yes
  ./scripts/deploy_space.sh --target main --yes
  SPACE_PROFILE=ui ./scripts/deploy_space.sh --target main --yes
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Run ./scripts/deploy_space.sh --help for usage." >&2
      exit 1
      ;;
  esac
done

case "${DEPLOY_TARGET}" in
  main)
    SPACE_REPO_ID="${SPACE_REPO_ID:-${MAIN_SPACE_REPO_ID}}"
    SPACE_PROFILE="${SPACE_PROFILE:-local}"
    ;;
  ui)
    SPACE_REPO_ID="${SPACE_REPO_ID:-${UI_SPACE_REPO_ID}}"
    SPACE_PROFILE="${SPACE_PROFILE:-ui}"
    ;;
  *)
    echo "Unsupported DEPLOY_TARGET=${DEPLOY_TARGET}. Use 'main' or 'ui'." >&2
    exit 1
    ;;
esac

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

case "${SPACE_PROFILE}" in
  local|ui) ;;
  *)
    echo "Unsupported SPACE_PROFILE=${SPACE_PROFILE}. Use 'local' or 'ui'." >&2
    exit 1
    ;;
esac

cp README.md LICENSE app.py local_inference.py modal_inference.py packages.txt "${STAGING_DIR}/"
if [[ "${SPACE_PROFILE}" == "ui" ]]; then
  cat > "${STAGING_DIR}/requirements.txt" <<'EOF'
gradio
modal
rich
EOF
  echo "Using SPACE_PROFILE=ui; staging lightweight requirements for fast UI iteration."
else
  cp requirements.txt "${STAGING_DIR}/"
  echo "Using SPACE_PROFILE=local; staging full local-inference requirements."
fi
cp -R docs data "${STAGING_DIR}/"
mkdir -p "${STAGING_DIR}/scripts/manual"
cp scripts/upload_gguf_to_hub.sh "${STAGING_DIR}/scripts/"
cp scripts/manual/local_smoke.py "${STAGING_DIR}/scripts/manual/"

if [[ "${SPACE_PROFILE}" == "ui" ]]; then
  echo "Skipping local GGUF in UI profile."
elif [[ "${INCLUDE_LOCAL_GGUF}" == "1" ]]; then
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
echo "Target: ${DEPLOY_TARGET} (${SPACE_REPO_ID})"
echo "Profile: ${SPACE_PROFILE}"
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
