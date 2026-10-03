#!/usr/bin/env bash
# Run the complete WhatsApp-to-fine-tuning-dataset workflow.
# Usage: ./scripts/run_from_raw.sh /absolute/path/to/whatsapp_chat.txt.txt [output_directory]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_FILE="${1:-}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/outputs}"
SELF_SENDER="${SELF_SENDER:-Ahmed}"

if [[ -z "${RAW_FILE}" ]]; then
  echo "Usage: $0 /absolute/path/to/whatsapp_chat.txt.txt [output_directory]" >&2
  echo "Optional: SELF_SENDER='Your WhatsApp Name' $0 /path/to/chat.txt" >&2
  exit 2
fi

if [[ ! -f "${RAW_FILE}" ]]; then
  echo "Error: raw WhatsApp export was not found: ${RAW_FILE}" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required. Install Python 3.10 or newer and run this command again." >&2
  exit 2
fi

PYTHON_MAJOR_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[1/4] Using Python ${PYTHON_MAJOR_MINOR}"
echo "[2/4] Running the dependency-free test suite"
(
  cd "${PROJECT_ROOT}"
  python3 -m unittest discover -s tests -v
)

echo "[3/4] Building the local dataset"
python3 "${PROJECT_ROOT}/src/twin_dataset_builder.py" \
  --input "${RAW_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --self-sender "${SELF_SENDER}" \
  --context-turns 5 \
  --max-gap-hours 4 \
  --burst-gap-seconds 300 \
  --min-response-chars 2 \
  --max-duplicate-responses 3 \
  --validation-ratio 0.10

echo "[4/4] Running the post-build quality audit"
python3 "${PROJECT_ROOT}/scripts/audit_outputs.py" "${OUTPUT_DIR}"

echo
printf 'Finished. Review this file before training: %s\n' "${OUTPUT_DIR}/audit_report.json"
printf 'Primary training file: %s\n' "${OUTPUT_DIR}/train.jsonl"
printf 'Validation file: %s\n' "${OUTPUT_DIR}/validation.jsonl"
