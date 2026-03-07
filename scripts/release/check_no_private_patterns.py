#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".yml",
    ".yaml",
    ".py",
    ".sh",
    ".cff",
}

SKIP_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".gif",
    ".nii",
    ".nii.gz",
    ".hdf5",
    ".pt",
    ".docx",
    ".xlsx",
    ".zip",
    ".gz",
}


def _iter_text_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel_parts = p.relative_to(root).parts
        except Exception:
            rel_parts = ()
        # Skip third-party source clones (downloaded at runtime).
        if len(rel_parts) >= 2 and rel_parts[0] == "third_party" and rel_parts[1] == "meld_graph":
            continue
        # Skip any git metadata folders.
        if ".git" in rel_parts:
            continue
        # Avoid self-triggering on the literal pattern strings embedded in this checker.
        # (We still scan other scripts.)
        if p.name == "check_no_private_patterns.py":
            continue
        suf = p.suffix.lower()
        if suf in SKIP_SUFFIXES:
            continue
        if suf in TEXT_SUFFIXES:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail if private paths or raw internal subject IDs are present.")
    ap.add_argument("--root", default=".", type=Path)
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    patterns: List[Tuple[str, str]] = [
        ("/cpfs01/", "HPC absolute path"),
        ("P01-", "raw internal subject id prefix"),
    ]

    hits: List[Tuple[Path, str, str]] = []
    for p in _iter_text_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, why in patterns:
            if pat in text:
                hits.append((p, pat, why))

    if hits:
        print("FAILED: private patterns detected:", file=sys.stderr)
        for p, pat, why in hits[:200]:
            print(f"- {p}: contains {pat!r} ({why})", file=sys.stderr)
        if len(hits) > 200:
            print(f"... and {len(hits) - 200} more", file=sys.stderr)
        return 2

    print(f"ok: no private patterns under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
