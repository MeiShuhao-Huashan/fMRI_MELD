#!/usr/bin/env python3
"""
Generate revision Figure 2 panels (scope38; final main-text protocol).

User request (2026-02-10):
  - Cohort: n=38 (ILAE 1–2 AND MRIDetectability ∈ {Intermediate, Difficult})
  - Protocol: same as paper/revision/table2/final and paper/revision/table3/final
  - Keep Figure2 panels: C, E, F, G
  - Output: one PDF per panel (user will do final layout)

Inputs (single source of truth)
  - paper/revision/table3/final/table3_scope38_audit.json
  - paper/revision/table3/final/table3_scope38_subject_ids.tsv
  - paper/revision/table3/final/table3_source_scope38_final.md  (for Panel E Δ+CI values)

Note on Panel C (deploy gate):
  The +20 TrackA eval_dir does not store `all_folds_gate_decisions.csv`.
  We therefore read the gate decisions from the original paper TrackA eval_dir:
    meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/all_folds_gate_decisions.csv
  and subset to the same 38 subjects.

Outputs (generated)
  - paper/revision/figure2/panelC_deploy_gate_scope38.pdf
  - paper/revision/figure2/panelE_cluster_effects_scope38.pdf
  - paper/revision/figure2/panelF_subject_endpoints_scope38.pdf
  - paper/revision/figure2/panelG_complementarity_scope38.pdf
  - paper/revision/figure2/figure2_scope38_inputs.json

Run:
  source env.sh
  python scripts/paper/make_revision_figure2_scope38_panels.py
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse styling/parsers from the original Figure 2 script.
from paper.figure2.make_figure2 import (  # type: ignore
    MM_PER_INCH,
    _extract_p_value,
    _load_figure_colors,
    _format_delta_ci_text,
    _parse_delta_ci,
    _setup_font,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_DIR))
    except Exception:
        return str(p)


def _read_tsv_column(path: Path, col: str) -> List[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"Empty TSV: {path}")
    header = rows[0].split("\t")
    try:
        j = header.index(col)
    except ValueError as e:
        raise ValueError(f"Missing column {col!r} in {path}") from e
    out: List[str] = []
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if j >= len(parts):
            continue
        v = parts[j].strip()
        if v:
            out.append(v)
    return out


def _load_eval_subject_level(eval_dir: Path) -> pd.DataFrame:
    for cand in [
        eval_dir / "all_folds_subject_level_results.csv",
        eval_dir / "subject_level_results.csv",
        eval_dir / "fold0" / "val" / "subject_level_results.csv",
    ]:
        if cand.exists():
            df = pd.read_csv(cand)
            if "subject_id" not in df.columns:
                raise KeyError(f"Missing subject_id column in {cand}")
            return df
    raise FileNotFoundError(f"Cannot locate subject_level_results.csv under: {eval_dir}")


def _apply_boxdsc_dice_safeguard(df: pd.DataFrame, *, eval_dir: Path, min_cluster_dice: float) -> pd.DataFrame:
    """
    Align the primary endpoint with Table2/Table3 safeguard:
      Det(boxDSC>0.22 & Dice>min_cluster_dice)

    We override the subject-level `detected_boxdsc` by requiring at least one cluster that is a
    box-TP (`is_tp_boxdsc`) AND has mask Dice above the threshold.
    """
    if float(min_cluster_dice) <= 0.0:
        return df

    cl_path = eval_dir / "cluster_level_results.csv"
    if not cl_path.exists():
        return df

    cl = pd.read_csv(cl_path)
    if cl.empty:
        return df

    cl["subject_id"] = cl["subject_id"].astype(str).str.strip()

    if "is_tp_boxdsc" in cl.columns:
        cl_tp = cl["is_tp_boxdsc"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    else:
        # Conservative fallback for legacy cluster tables.
        if "box_dice" in cl.columns:
            cl_tp = pd.to_numeric(cl["box_dice"], errors="coerce").fillna(0.0).astype(float) > 0.22
        else:
            cl_tp = pd.Series(False, index=cl.index, dtype=bool)

    if "dice" in cl.columns:
        cl_dice = pd.to_numeric(cl["dice"], errors="coerce").fillna(0.0).astype(float)
    else:
        cl_dice = pd.Series(0.0, index=cl.index, dtype=float)

    hit_ids = set(cl.loc[cl_tp & (cl_dice > float(min_cluster_dice)), "subject_id"].astype(str).tolist())

    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    no_pred = pd.to_numeric(out.get("n_clusters"), errors="coerce").fillna(0.0).astype(float) <= 0.0
    out["detected_boxdsc"] = out["subject_id"].isin(hit_ids) & (~no_pred)
    return out


def _parse_table3_md_rows(table_md: Path) -> List[Dict[str, str]]:
    """
    Parse the markdown table in `table3_source_scope38_final.md`.

    Expected columns:
      Scale | Metric | ... | ... | ... | Δ (...) | p (...)
    """
    lines = table_md.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, str]] = []
    in_table = False
    for line in lines:
        if not line.startswith("|"):
            if in_table and rows:
                break
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        if parts[0] == "Scale" and parts[1] == "Metric":
            in_table = True
            continue
        if not in_table:
            continue
        if parts[0] == "---" or parts[1] == "---":
            continue
        # Use the last 2 columns as (delta, p) regardless of long header labels.
        rows.append(
            {
                "Scale": parts[0],
                "Metric": parts[1],
                "Delta": parts[-2],
                "p": parts[-1],
            }
        )
    if not rows:
        raise ValueError(f"Failed to parse table rows from: {table_md}")
    return rows


def _find_row(rows: List[Dict[str, str]], *, metric_contains: str, scale_contains: str) -> Dict[str, str]:
    hits = [r for r in rows if (metric_contains in r["Metric"]) and (scale_contains in r["Scale"])]
    if len(hits) != 1:
        raise ValueError(f"Expected 1 hit for metric={metric_contains!r}, scale={scale_contains!r}; got {len(hits)}")
    return hits[0]


def _mcnemar_exact_p(n01: int, n10: int) -> float:
    """
    Exact two-sided McNemar test (binomial on discordant pairs).
    """
    n = int(n01 + n10)
    if n <= 0:
        return 1.0
    k = int(min(n01, n10))

    def _pmf(x: int) -> float:
        return math.comb(n, x) * (0.5**n)

    p = 2.0 * sum(_pmf(x) for x in range(0, k + 1))
    return float(min(1.0, max(0.0, p)))


def _save_panel(fig: plt.Figure, out_pdf: Path, out_png: Optional[Path] = None) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    audit_path = PROJECT_DIR / "paper/revision/table3/final/table3_scope38_audit.json"
    subjects_tsv = PROJECT_DIR / "paper/revision/table3/final/table3_scope38_subject_ids.tsv"
    table3_md = PROJECT_DIR / "paper/revision/table3/final/table3_source_scope38_final.md"

    out_dir = PROJECT_DIR / "paper/revision/figure2"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scope_subjects = _read_tsv_column(subjects_tsv, "subject_id")
    if len(scope_subjects) != 38:
        raise ValueError(f"Expected 38 scope subjects, got {len(scope_subjects)}")

    boxdsc_min_cluster_dice = float(audit.get("params", {}).get("boxdsc_min_cluster_dice", 0.0) or 0.0)
    endpoint_label = "Det(boxDSC>0.22)"
    if boxdsc_min_cluster_dice > 0.0:
        endpoint_label = f"Det(boxDSC>0.22 & Dice>{boxdsc_min_cluster_dice:g})"

    t1_eval_dir = Path(audit["inputs"]["t1_eval_dir"])
    fmri_eval_dir = Path(audit["inputs"]["fmri_eval_dir"])
    tracka_eval_dir = Path(audit["inputs"]["tracka_eval_dir"])

    # Subject-level rows (filter to scope38)
    t1_sub = _load_eval_subject_level(t1_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()
    fm_sub = _load_eval_subject_level(fmri_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()
    tr_sub = _load_eval_subject_level(tracka_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()

    t1_sub = _apply_boxdsc_dice_safeguard(t1_sub, eval_dir=t1_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)
    fm_sub = _apply_boxdsc_dice_safeguard(fm_sub, eval_dir=fmri_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)
    tr_sub = _apply_boxdsc_dice_safeguard(tr_sub, eval_dir=tracka_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)

    COLORS = _load_figure_colors()
    _setup_font()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
        }
    )

    # ---------------- Panel C: deploy gate (quadrant plot) ----------------
    gate_csv = Path(
        os.environ.get(
            "MELD_TRACKA_GATE_CSV",
            PROJECT_DIR
            / "meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/all_folds_gate_decisions.csv",
        )
    )
    gate = pd.read_csv(gate_csv)
    gate = gate[gate["subject_id"].astype(str).isin(scope_subjects)].copy()
    gate["t1_conf"] = gate["t1_conf"].astype(float)
    gate["fmri_minus_t1"] = gate["fmri_minus_t1"].astype(float)

    thr_conf = float(gate["t1_conf_threshold"].iloc[0])
    thr_diff = 0.15  # paper TrackA default

    figC = plt.figure(figsize=(95.0 / MM_PER_INCH, 72.0 / MM_PER_INCH))
    axC = figC.add_subplot(1, 1, 1)
    axC.set_title("C  Deploy gate (n=38)", loc="left", fontsize=11, pad=6)

    def _scatter(sel: pd.Series, *, color: str, label: str) -> None:
        axC.scatter(
            gate.loc[sel, "t1_conf"],
            gate.loc[sel, "fmri_minus_t1"],
            s=44,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
            label=label,
            zorder=3,
        )

    _scatter(gate["source"].astype(str) == "T1", color=COLORS["t1"], label="Keep T1")
    _scatter(gate["source"].astype(str) == "fMRI", color=COLORS["fmri"], label="Switch to fMRI")
    _scatter(gate["source"].astype(str) == "T1+fMRI", color=COLORS["trackA"], label="Inject fMRI")

    axC.axvline(thr_conf, color="black", lw=1.3, alpha=0.55)
    axC.axhline(thr_diff, color="black", lw=1.0, alpha=0.50, linestyle="--")

    axC.set_xlim(0.0, 1.0)
    axC.set_ylim(min(-0.20, float(gate["fmri_minus_t1"].min()) - 0.03), float(gate["fmri_minus_t1"].max()) + 0.05)
    axC.set_xlabel("T1 confidence")
    axC.set_ylabel("fMRI advantage (Δ)")
    axC.grid(True, linestyle=":", alpha=0.35)

    y0, y1 = axC.get_ylim()
    axC.text(thr_conf + 0.01, y0 + 0.02 * (y1 - y0), f"thr={thr_conf:.4f}", fontsize=9, ha="left", va="bottom")
    axC.text(0.01, thr_diff + 0.01, "0.15", fontsize=9, ha="left", va="bottom")
    axC.text(
        0.99,
        0.98,
        f"keep={int((gate['source']=='T1').sum())}, switch={int((gate['source']=='fMRI').sum())}, inject={int((gate['source']=='T1+fMRI').sum())}",
        transform=axC.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )
    axC.legend(loc="upper left", frameon=False, fontsize=9)
    figC.tight_layout()

    # ---------------- Panel E: forest plot (cluster-level deltas; boxDSC) ----------------
    table_rows = _parse_table3_md_rows(table3_md)
    metric_specs = [
        ("FP/pt", "False-positive clusters per subject"),
        ("Precision", "Precision (TP clusters / all clusters)"),
        ("F1", "F1 score"),
    ]
    deltas: List[Tuple[str, object, str, bool]] = []
    for lbl, metric_contains in metric_specs:
        row = _find_row(table_rows, metric_contains=metric_contains, scale_contains="Cluster-level (boxDSC")
        d = _parse_delta_ci(row["Delta"])
        p_cell = row["p"]
        p = _extract_p_value(p_cell)
        sig = (p is not None) and (p < 0.05)
        deltas.append((lbl, d, p_cell, sig))

    figE = plt.figure(figsize=(180.0 / MM_PER_INCH, 70.0 / MM_PER_INCH))
    axE = figE.add_subplot(1, 1, 1)
    axE.set_title(f"E  Cluster-level effects ({endpoint_label}; Δ Track A − T1)", loc="left", fontsize=11)

    y = np.arange(len(deltas))[::-1]
    axE.axvline(0, color="black", lw=1, alpha=0.6)
    span = max(1e-6, float(max(d.hi for _, d, _, _ in deltas) - min(d.lo for _, d, _, _ in deltas)))
    for i, (lbl, d, _, sig) in enumerate(deltas):
        yi = y[i]
        axE.plot([d.lo, d.hi], [yi, yi], color=COLORS["trackA"], lw=2)
        axE.plot(d.delta, yi, marker="o", color=COLORS["trackA"])
        txt = _format_delta_ci_text(d) + (" *" if sig else "")
        axE.text(d.hi + 0.04 * span, yi, txt, va="center", ha="left", fontsize=9)
    axE.set_yticks(y)
    axE.set_yticklabels([lbl for (lbl, _, _, _) in deltas])
    axE.set_xlabel("Δ (Track A − T1)")
    axE.tick_params(axis="y", pad=2)
    lo_all = min(d.lo for _, d, _, _ in deltas)
    hi_all = max(d.hi for _, d, _, _ in deltas)
    pad = 0.05 * max(1e-6, hi_all - lo_all)
    axE.set_xlim(lo_all - pad, hi_all + 0.60 * (hi_all - lo_all + 1e-6))
    axE.grid(axis="x", linestyle=":", alpha=0.4)
    axE.text(0.0, -0.14, "* p<0.05 (paired; Table 3 scope38)", transform=axE.transAxes, fontsize=9, ha="left")
    figE.tight_layout()

    # ---------------- Panel F: subject endpoints (Detection + PPV>=0.5 + Pinpointing) ----------------
    t1_found = t1_sub["detected_boxdsc"].astype(bool).to_numpy()
    fmri_found = fm_sub["detected_boxdsc"].astype(bool).to_numpy()
    track_found = tr_sub["detected_boxdsc"].astype(bool).to_numpy()

    t1_ppv = t1_sub["detected_ppv50"].astype(bool).to_numpy()
    fmri_ppv = fm_sub["detected_ppv50"].astype(bool).to_numpy()
    track_ppv = tr_sub["detected_ppv50"].astype(bool).to_numpy()

    t1_pin = t1_sub["pinpointed"].astype(bool).to_numpy()
    fmri_pin = fm_sub["pinpointed"].astype(bool).to_numpy()
    track_pin = tr_sub["pinpointed"].astype(bool).to_numpy()

    p_det = _mcnemar_exact_p(int(np.sum((~t1_found) & track_found)), int(np.sum(t1_found & (~track_found))))

    figF = plt.figure(figsize=(95.0 / MM_PER_INCH, 70.0 / MM_PER_INCH))
    axF = figF.add_subplot(1, 1, 1)
    axF.set_title("F  Subject endpoints (n=38)", loc="left", fontsize=11)

    groups = ["Detection", "PPV≥0.5", "Pinpointing"]
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    vals_t1 = [float(np.mean(t1_found)), float(np.mean(t1_ppv)), float(np.mean(t1_pin))]
    vals_fm = [float(np.mean(fmri_found)), float(np.mean(fmri_ppv)), float(np.mean(fmri_pin))]
    vals_tr = [float(np.mean(track_found)), float(np.mean(track_ppv)), float(np.mean(track_pin))]

    axF.bar(x - width, vals_t1, width=width, color=COLORS["t1"], label="T1-only")
    axF.bar(x, vals_fm, width=width, color=COLORS["fmri"], label="fMRI-only")
    axF.bar(x + width, vals_tr, width=width, color=COLORS["trackA"], label="Track A")
    axF.set_ylim(0, 1.0)
    axF.set_xticks(x)
    axF.set_xticklabels(groups)
    axF.set_ylabel("Rate")
    axF.grid(axis="y", linestyle=":", alpha=0.4)
    axF.legend(loc="lower right", frameon=False)
    axF.text(0.02, 0.02, f"Primary: {endpoint_label}", transform=axF.transAxes, fontsize=8.0, va="bottom")

    if p_det < 0.05:
        y0 = max(vals_t1[0], vals_tr[0]) + 0.10
        h = 0.02
        axF.plot([x[0] - width, x[0] - width, x[0] + width, x[0] + width], [y0, y0 + h, y0 + h, y0], color="black", lw=1)
        axF.text(x[0], y0 + h + 0.01, "*", ha="center", va="bottom", fontsize=14)
    figF.tight_layout()

    # ---------------- Panel G: complementarity 2×2 ----------------
    n11 = int(np.sum(t1_found & track_found))
    n10 = int(np.sum(t1_found & ~track_found))  # hurt
    n01 = int(np.sum(~t1_found & track_found))  # rescued
    n00 = int(np.sum(~t1_found & ~track_found))
    star = "*" if p_det < 0.05 else ""

    figG = plt.figure(figsize=(95.0 / MM_PER_INCH, 70.0 / MM_PER_INCH))
    axG = figG.add_subplot(1, 1, 1)
    axG.set_title(f"G  Complementarity (2×2; primary endpoint){star}", loc="left", fontsize=11)
    axG.axis("off")
    table_data = [
        ["", "Track A:\nnot detected", "Track A:\ndetected"],
        ["T1:\nnot detected", f"{n00}", f"{n01}  (rescued)"],
        ["T1:\ndetected", f"{n10}  (hurt)", f"{n11}"],
    ]
    tbl = axG.table(cellText=table_data, cellLoc="center", bbox=[0.0, 0.18, 1.0, 0.68])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0 or c == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")
        if r == 1 and c == 2:
            cell.set_facecolor("#E8F0FE")
        if r == 2 and c == 1:
            cell.set_facecolor("#FCE8E6")
    axG.text(0.5, 0.10, f"rescued={n01}, hurt={n10}; p={p_det:.4f}", ha="center", va="center", transform=axG.transAxes, fontsize=10)
    figG.tight_layout()

    # ---------------- Save ----------------
    _save_panel(figC, out_dir / "panelC_deploy_gate_scope38.pdf", out_dir / "panelC_deploy_gate_scope38.png")
    _save_panel(figE, out_dir / "panelE_cluster_effects_scope38.pdf", out_dir / "panelE_cluster_effects_scope38.png")
    _save_panel(figF, out_dir / "panelF_subject_endpoints_scope38.pdf", out_dir / "panelF_subject_endpoints_scope38.png")
    _save_panel(figG, out_dir / "panelG_complementarity_scope38.pdf", out_dir / "panelG_complementarity_scope38.png")

    (out_dir / "figure2_scope38_inputs.json").write_text(
        json.dumps(
            {
                "audit_json": _relpath(Path(audit_path)),
                "scope_subjects_tsv": _relpath(Path(subjects_tsv)),
                "table3_scope38_md": _relpath(Path(table3_md)),
                "eval_dirs": {
                    "t1": _relpath(Path(t1_eval_dir)),
                    "fmri": _relpath(Path(fmri_eval_dir)),
                    "trackA": _relpath(Path(tracka_eval_dir)),
                },
                "gate_csv": _relpath(Path(gate_csv)),
                "gate_thresholds": {"t1_conf_thr": thr_conf, "diff_thr": thr_diff},
                "panelG": {"n00": n00, "n01_rescued": n01, "n10_hurt": n10, "n11": n11, "mcnemar_p": p_det},
                "primary_endpoint": endpoint_label,
                "figure_colors": COLORS,
                "outputs": {
                    "panelC_pdf": _relpath(out_dir / "panelC_deploy_gate_scope38.pdf"),
                    "panelE_pdf": _relpath(out_dir / "panelE_cluster_effects_scope38.pdf"),
                    "panelF_pdf": _relpath(out_dir / "panelF_subject_endpoints_scope38.pdf"),
                    "panelG_pdf": _relpath(out_dir / "panelG_complementarity_scope38.pdf"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
