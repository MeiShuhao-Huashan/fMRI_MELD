#!/usr/bin/env bash
set -euo pipefail

# Upload weight archives as GitHub Release assets.
#
# Requirements:
#   - export GITHUB_TOKEN=...        (classic PAT: repo scope; or fine-grained token with Releases write)
#   - export GITHUB_REPO=owner/repo  (default: MeiShuhao-Huashan/fMRI_MELD_epilepsia)
#   - export GITHUB_RELEASE_TAG=...  (e.g., epilepsia-weights-v1)
#
# Usage (from repo root):
#   export GITHUB_TOKEN=...
#   export GITHUB_RELEASE_TAG=epilepsia-weights-v1
#   bash weights/upload_release_assets.sh
#
# Notes:
#   - This script uploads files from weights/dist/*.tar.gz
#   - It creates the Release if it does not exist.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found." >&2
  exit 1
fi

GITHUB_TOKEN="${GITHUB_TOKEN:-}"
if [ -z "${GITHUB_TOKEN}" ]; then
  echo "ERROR: Set GITHUB_TOKEN before running (PAT with Releases write access)." >&2
  exit 1
fi

GITHUB_REPO="${GITHUB_REPO:-MeiShuhao-Huashan/fMRI_MELD_epilepsia}"
GITHUB_RELEASE_TAG="${GITHUB_RELEASE_TAG:-}"
if [ -z "${GITHUB_RELEASE_TAG}" ]; then
  echo "ERROR: Set GITHUB_RELEASE_TAG (e.g., epilepsia-weights-v1) before running." >&2
  exit 1
fi

API="https://api.github.com/repos/${GITHUB_REPO}"
DIST_DIR="${ROOT}/dist"

if [ ! -d "${DIST_DIR}" ]; then
  echo "ERROR: Missing ${DIST_DIR}. Build archives first (see weights/README.md)." >&2
  exit 1
fi

ASSETS=()
while IFS= read -r -d '' f; do
  ASSETS+=("$f")
done < <(find "${DIST_DIR}" -maxdepth 1 -type f -name '*.tar.gz' -print0 | sort -z)

if [ "${#ASSETS[@]}" -eq 0 ]; then
  echo "ERROR: No .tar.gz assets found under ${DIST_DIR}." >&2
  exit 1
fi

get_upload_url() {
  python - <<'PY'
import json, sys
obj=json.load(sys.stdin)
u=obj.get("upload_url","")
u=u.split("{",1)[0]
print(u)
PY
}

echo "Checking release tag ${GITHUB_RELEASE_TAG} on ${GITHUB_REPO}..."
status_code="$(curl -sS -o /tmp/_gh_rel.json -w "%{http_code}" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "${API}/releases/tags/${GITHUB_RELEASE_TAG}" || true)"

if [ "${status_code}" = "200" ]; then
  upload_url="$(cat /tmp/_gh_rel.json | get_upload_url)"
  echo "Found existing release. upload_url=${upload_url}"
else
  echo "Release not found (status=${status_code}). Creating..."
  payload="$(python - <<PY
import json
print(json.dumps({
  "tag_name": "${GITHUB_RELEASE_TAG}",
  "name": "${GITHUB_RELEASE_TAG}",
  "draft": False,
  "prerelease": False,
  "generate_release_notes": False,
}))
PY
)"
  status_code2="$(curl -sS -o /tmp/_gh_rel_create.json -w "%{http_code}" \
    -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${API}/releases" || true)"
  if [ "${status_code2}" != "201" ]; then
    echo "ERROR: Failed to create release (status=${status_code2}). Response:" >&2
    cat /tmp/_gh_rel_create.json >&2 || true
    exit 1
  fi
  upload_url="$(cat /tmp/_gh_rel_create.json | get_upload_url)"
  echo "Created release. upload_url=${upload_url}"
fi

for f in "${ASSETS[@]}"; do
  name="$(basename "$f")"
  echo "Uploading ${name}..."
  # GitHub will reject if the asset already exists with the same name.
  curl -sS -L --fail \
    -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Content-Type: application/gzip" \
    --data-binary @"${f}" \
    "${upload_url}?name=${name}" >/tmp/_gh_asset_${name}.json
  echo "  ok: ${name}"
done

echo
echo "DONE. Verify from users' side:"
echo "  export GITHUB_RELEASE_TAG=${GITHUB_RELEASE_TAG}"
echo "  bash weights/download_weights.sh"
