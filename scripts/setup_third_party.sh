#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOCK_FILE="${ROOT}/third_party/meld_graph.lock"
if [ ! -f "${LOCK_FILE}" ]; then
  echo "ERROR: Missing lock file: ${LOCK_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${LOCK_FILE}"

TP_DIR="${ROOT}/third_party"
SRC_DIR="${TP_DIR}/meld_graph"
PATCH_PATH="${ROOT}/${MELD_GRAPH_PATCH}"

if [ ! -f "${PATCH_PATH}" ]; then
  echo "ERROR: Missing patch: ${PATCH_PATH}" >&2
  exit 1
fi

mkdir -p "${TP_DIR}"

if [ ! -d "${SRC_DIR}/.git" ]; then
  echo "[1/4] Cloning upstream meld_graph..."
  git clone "${MELD_GRAPH_REPO}" "${SRC_DIR}"
else
  echo "[1/4] Using existing clone: ${SRC_DIR}"
fi

echo "[2/4] Checking out locked commit: ${MELD_GRAPH_REF} (tag ${MELD_GRAPH_TAG})"
(
  cd "${SRC_DIR}"
  git fetch --tags --prune
  git checkout --detach "${MELD_GRAPH_REF}"
)

echo "[3/4] Applying patch (idempotent)..."
(
  cd "${SRC_DIR}"
  if git apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
    echo "  - patch already applied"
  else
    git apply "${PATCH_PATH}"
    echo "  - patch applied"
  fi
)

echo "[4/4] Installing meld_graph from source (editable)..."
python -m pip install -e "${SRC_DIR}"

cat <<EOF

DONE.

Notes:
- For end-to-end runs, set:
    export MELD_DATA_PATH="${ROOT}/meld_data"
- If you need MELD parameters (fsaverage_sym surfaces, etc.), you can download them via:
    python -c "from meld_graph.download_data import get_meld_params; get_meld_params()"
EOF

