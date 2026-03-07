#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

mkdir -p downloaded

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Download sources
# -----------------------------------------------------------------------------
#
# Recommended: GitHub Releases for this repo (public, no auth required).
#   export GITHUB_REPO="MeiShuhao-Huashan/MELD_fMRI"
#   export GITHUB_RELEASE_TAG="epilepsia-weights-v1"
#
# Advanced: override base URL directly (no trailing slash), e.g.:
#   export GITHUB_BASE_URL="https://github.com/<user>/<repo>/releases/download/<tag>"
#
# Fallback: Zenodo (optional).
#   export ZENODO_RECORD_ID=1234567
# or:
#   export ZENODO_BASE_URL="https://zenodo.org/records/1234567/files"
#
GITHUB_REPO="${GITHUB_REPO:-MeiShuhao-Huashan/MELD_fMRI}"
GITHUB_RELEASE_TAG="${GITHUB_RELEASE_TAG:-}"
GITHUB_BASE_URL="${GITHUB_BASE_URL:-}"

ZENODO_RECORD_ID="${ZENODO_RECORD_ID:-}"
ZENODO_BASE_URL="${ZENODO_BASE_URL:-}"

_MODE=""
_BASE_URL=""
_URL_SUFFIX=""

if [ -n "${GITHUB_BASE_URL}" ] || [ -n "${GITHUB_RELEASE_TAG}" ]; then
  _MODE="github"
  if [ -z "${GITHUB_BASE_URL}" ]; then
    if [ -z "${GITHUB_RELEASE_TAG}" ]; then
      echo "ERROR: Set GITHUB_RELEASE_TAG (or GITHUB_BASE_URL) before running." >&2
      exit 1
    fi
    GITHUB_BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/${GITHUB_RELEASE_TAG}"
  fi
  _BASE_URL="${GITHUB_BASE_URL}"
  _URL_SUFFIX=""
else
  _MODE="zenodo"
  if [ -z "${ZENODO_BASE_URL}" ]; then
    if [ -z "${ZENODO_RECORD_ID}" ]; then
      echo "ERROR: Set GITHUB_RELEASE_TAG (recommended) OR ZENODO_RECORD_ID (or ZENODO_BASE_URL) before running." >&2
      exit 1
    fi
    ZENODO_BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"
  fi
  _BASE_URL="${ZENODO_BASE_URL}"
  _URL_SUFFIX="?download=1"
fi

download_one() {
  local fname="$1"
  local url="${_BASE_URL}/${fname}${_URL_SUFFIX}"
  local out="downloaded/${fname}"
  if [ -f "${out}" ]; then
    echo "skip: ${out} exists"
    return 0
  fi
  echo "download(${_MODE}): ${fname}"
  curl -L --fail --retry 3 --retry-delay 2 -o "${out}" "${url}"
}

# By default, download the rs-fMRI weights required for end-to-end inference + TrackA.
download_one "fmri_models.tar.gz"
download_one "laterality_models.tar.gz"

echo
echo "DONE. Next:"
echo "  sha256sum -c weights/checksums.sha256"
