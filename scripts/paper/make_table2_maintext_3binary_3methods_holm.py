#!/usr/bin/env python3
"""
Post-process Table2(final) into the main-text format requested by the user:

- Keep only 3 binary endpoints (A/B/C):
    A) Det(boxDSC>0.22 & Dice>0.01)
    B) Det(PPV-in-mask>=0.5)
    C) Pinpointing
- Keep only 3 methods:
    - MELD (self-trained)
    - rs-fMRI
    - TrackA (fusion)
- Apply Holm–Bonferroni multiple-comparison correction across the 9 tests
  (3 endpoints × 3 methods) for the p-values reported in the main table.

Inputs (expected to exist):
  - prognosis_table_boxdsc022_<tag>.tsv
  - prognosis_table_ppv05_<tag>.tsv
  - prognosis_table_pinpointing_<tag>.tsv

Outputs:
  - prognosis_table_maintext_3binary_3methods_holm.md

Run:
  source env.sh
  python scripts/paper/make_table2_maintext_3binary_3methods_holm.py \\
    --in_dir paper/revision/table2/final \\
    --tag maintext_selftrained_unconstrained_meld_tracka_paper_plus20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METHOD_ORDER = ["MELD (self-trained)", "rs-fMRI", "TrackA (fusion)"]


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _holm_bonferroni(p: np.ndarray) -> np.ndarray:
    """
    Holm–Bonferroni adjusted p-values (strong FWER control).
    """
    p = np.asarray(p, dtype=float)
    n = int(p.size)
    order = np.argsort(p)
    ranked = p[order]
    adj = (n - np.arange(n)) * ranked
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def _read_panel(in_dir: Path, fname: str) -> pd.DataFrame:
    df = pd.read_csv(in_dir / fname, sep="\t")
    if "Model" not in df.columns or "p" not in df.columns:
        raise KeyError(f"Missing required columns in: {in_dir / fname}")
    df["Model"] = df["Model"].astype(str).str.strip()
    df = df[df["Model"].isin(set(METHOD_ORDER))].copy()
    df["Model"] = pd.Categorical(df["Model"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)
    df["p_raw"] = pd.to_numeric(df["p"], errors="coerce").astype(float)
    return df


def _md_table(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="paper/revision/table2/final", type=Path)
    ap.add_argument(
        "--tag",
        default="maintext_selftrained_unconstrained_meld_tracka_paper_plus20",
        help="Tag used in filenames (same tag as make_prognosis_table_constraints_a30_t80.py).",
    )
    ap.add_argument(
        "--out_md",
        default="paper/revision/table2/final/prognosis_table_maintext_3binary_3methods_holm.md",
        type=Path,
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    tag = str(args.tag).strip()

    panels: Dict[str, pd.DataFrame] = {
        "A": _read_panel(in_dir, f"prognosis_table_boxdsc022_{tag}.tsv"),
        "B": _read_panel(in_dir, f"prognosis_table_ppv05_{tag}.tsv"),
        "C": _read_panel(in_dir, f"prognosis_table_pinpointing_{tag}.tsv"),
    }

    # Holm correction across the 9 tests (3 endpoints × 3 methods).
    all_p = np.concatenate([panels[k]["p_raw"].to_numpy(dtype=float) for k in ["A", "B", "C"]], axis=0)
    if np.any(~np.isfinite(all_p)):
        raise ValueError("Non-finite p-values encountered; cannot apply Holm correction.")
    all_adj = _holm_bonferroni(all_p)

    cursor = 0
    for key in ["A", "B", "C"]:
        df = panels[key].copy()
        n = len(df)
        df["p_holm"] = [_fmt_p(float(x)) for x in all_adj[cursor : cursor + n]]
        cursor += n

        # Replace p column with Holm-adjusted p for the main-text table.
        df["p"] = df["p_holm"]

        # Drop helpers.
        df = df.drop(columns=["p_raw", "p_holm"])
        panels[key] = df

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = []
    md_lines.append("# Prognosis association (adjusted; main-text; Holm-corrected p)")
    md_lines.append("")
    md_lines.append("- Cohort: Intermediate+Difficult; n=58 (ILAE12=38, ILAE3456=20)")
    md_lines.append("- Endpoints: 3 binary endpoints only (A/B/C); Intersection & continuous PPV reported in Supplement.")
    md_lines.append("- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).")
    md_lines.append("")
    md_lines.append("## A. Det(boxDSC>0.22 & Dice>0.01)")
    md_lines.append("")
    md_lines.extend(_md_table(panels["A"]))
    md_lines.append("")
    md_lines.append("## B. Det(PPV-in-mask≥0.5)")
    md_lines.append("")
    md_lines.extend(_md_table(panels["B"]))
    md_lines.append("")
    md_lines.append("## C. Pinpointing (any-cluster COM resected)")
    md_lines.append("")
    md_lines.extend(_md_table(panels["C"]))
    md_lines.append("")
    md_lines.append("Notes:")
    md_lines.append("- p values in this main-text table are Holm–Bonferroni corrected across 9 tests (3 endpoints × 3 methods).")
    md_lines.append("- Inference: penalized likelihood ratio tests; 95% CIs use profile penalized likelihood (Firth logistic).")

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

