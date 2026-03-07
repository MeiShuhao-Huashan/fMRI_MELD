#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


def _extract_docx_text(docx: Path) -> str:
    with zipfile.ZipFile(docx, "r") as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    # Remove tags and normalize whitespace.
    txt = re.sub(r"<[^>]+>", "", xml)
    txt = re.sub(r"\\s+", " ", txt).strip()
    return txt


def _must_contain(label: str, haystack: str, needle: str) -> None:
    if needle not in haystack:
        raise SystemExit(f"FAILED: missing {label}: {needle!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that key numeric anchors in the manuscript docx match released results.")
    ap.add_argument("--docx", required=True, type=Path, help="Path to the manuscript .docx (not included in this public repo).")
    args = ap.parse_args()

    docx = args.docx.expanduser().resolve()
    if not docx.exists():
        raise SystemExit(f"Missing docx: {docx}")

    docx_text = _extract_docx_text(docx)

    # ---- Anchors from scope38 Table 3 narrative (primary endpoint complementarity) ----
    for s in ["21/38", "30/38", "10/38", "1/38"]:
        _must_contain("Table3 scope38 anchor", docx_text, s)

    # ---- Anchors from fMRI ablation text (scope38 matrix) ----
    for s in ["58.8%", "28.6%", "13.3%", "1.079", "1.711"]:
        _must_contain("Ablation anchor", docx_text, s)

    # Common formatting variant: "n = 17" vs "n=17"
    if ("n = 17" not in docx_text) and ("n=17" not in docx_text):
        raise SystemExit("FAILED: missing cohort size anchor: 'n = 17' (or 'n=17') not found in docx text.")

    print("ok: manuscript docx contains all required numeric anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
