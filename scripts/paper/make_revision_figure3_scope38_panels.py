#!/usr/bin/env python3
"""
Generate revision Figure 3 panels (scope38; final main-text protocol).

User request (2026-02-10):
  - Cohort: n=38 (ILAE 1–2 AND MRIDetectability ∈ {Intermediate, Difficult})
  - Protocol: same as paper/revision/table2/final and paper/revision/table3/final
  - Keep Figure3 panels: A, B, C, D (from the original paper Figure 3 design)
  - Output: one PDF per panel (user will do final layout)

Inputs (single source of truth)
  - paper/revision/table3/final/table3_scope38_audit.json   (eval dirs + bootstrap params)
  - paper/revision/table3/final/table3_scope38_subject_ids.tsv
  - supplement_table_hs_patient_three_level_metrics.filled.csv (clinical stratifiers; has subject_id)

Outputs (generated)
  - paper/revision/figure3/panelA_detectability_scope38.pdf
  - paper/revision/figure3/panelB_complementarity_scope38.pdf
  - paper/revision/figure3/panelC_seeg_scope38.pdf
  - paper/revision/figure3/panelD_location_delta_scope38.pdf
  - paper/revision/figure3/panelE_intermediate_vs_difficult_scope38.pdf
  - paper/revision/figure3/figure3_scope38_inputs.json

Run:
  source env.sh
  python scripts/paper/make_revision_figure3_scope38_panels.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.figure3.make_figure3 import (  # type: ignore
    MM_PER_INCH,
    DEFAULT_FIGURE_WIDTH_MM,
    _fisher_exact_two_sided,
    _load_figure_colors,
    _mcnemar_exact_pvalue,
    _paired_bootstrap_ci_diff,
    _p_to_stars,
    _plot_complementarity_stacked,
    _plot_detection_dotwhisker,
    _plot_location_delta_forest,
    _setup_font,
    _wilson_ci,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

def _relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_DIR))
    except Exception:
        return str(p)

def _holm_bonferroni(p: np.ndarray) -> np.ndarray:
    """
    Holm–Bonferroni adjusted p-values (strong FWER control).

    Used for within-panel multiple comparisons in exploratory subgroup plots.
    """
    p = np.asarray(p, dtype=float)
    n = int(p.size)
    if n <= 1:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    adj = (n - np.arange(n)) * ranked
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


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


def _load_subject_level(eval_dir: Path) -> pd.DataFrame:
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
    Align `detected_boxdsc` with the Table2/Table3 safeguard:
      Det(boxDSC>0.22 & Dice>min_cluster_dice)
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


def _save_panel(fig: plt.Figure, out_pdf: Path, out_png: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_intermediate_vs_difficult_bar(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    detected_col: str,
    color: str,
    title: str,
    p_value: Optional[float] = None,
) -> None:
    """
    Intermediate vs Difficult bar chart for a given detected column.

    Mirrors the original Figure3 Panel B style (Fisher exact + Wilson CI).
    """

    if "MRIDetectability" not in merged.columns:
        raise KeyError("Missing clinical column: MRIDetectability")
    if detected_col not in merged.columns:
        raise KeyError(f"Missing column: {detected_col}")

    detectability = merged["MRIDetectability"].astype(str)
    int_mask = detectability.eq("Intermediate").to_numpy()
    diff_mask = detectability.eq("Difficult").to_numpy()

    n_int = int(np.sum(int_mask))
    n_diff = int(np.sum(diff_mask))
    if n_int == 0 or n_diff == 0:
        return

    detected = merged[detected_col].astype(bool).to_numpy()
    k_int = int(np.sum(int_mask & detected))
    k_diff = int(np.sum(diff_mask & detected))

    rate_int = float(k_int / n_int)
    rate_diff = float(k_diff / n_diff)
    ci_int = _wilson_ci(k_int, n_int)
    ci_diff = _wilson_ci(k_diff, n_diff)

    if p_value is None:
        try:
            p_value = float(_fisher_exact_two_sided(k_int, n_int - k_int, k_diff, n_diff - k_diff))
        except Exception:
            p_value = float("nan")

    bar_x = np.array([0.0, 1.0], dtype=float)
    bar_y = np.array([rate_int, rate_diff], dtype=float)
    bar_alphas = [0.90, 0.45]

    bars = ax.bar(
        bar_x,
        bar_y,
        width=0.68,
        color=[color, color],
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for b, a in zip(bars, bar_alphas):
        b.set_alpha(a)

    def _yerr(rate: float, ci: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = ci
        return (max(0.0, rate - float(lo)), max(0.0, float(hi) - rate))

    yerr = np.array([_yerr(rate_int, ci_int), _yerr(rate_diff, ci_diff)], dtype=float).T
    ax.errorbar(bar_x, bar_y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3.0, zorder=3)

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"Intermediate\n(n={n_int})", f"Difficult\n(n={n_diff})"], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)

    stars = _p_to_stars(float(p_value))
    if stars:
        ax.text(0.5, 1.02, stars, transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    if not np.isnan(float(p_value)):
        ax.text(0.5, 0.94, f"p={float(p_value):.3g}", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _merge_scope38(
    *,
    scope_subjects: List[str],
    t1_eval_dir: Path,
    fmri_eval_dir: Path,
    tracka_eval_dir: Path,
    meta_csv: Path,
    boxdsc_min_cluster_dice: float = 0.0,
) -> pd.DataFrame:
    # Subject-level endpoints
    t1 = _load_subject_level(t1_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()
    fm = _load_subject_level(fmri_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()
    tr = _load_subject_level(tracka_eval_dir).set_index("subject_id").loc[scope_subjects].reset_index()

    t1 = _apply_boxdsc_dice_safeguard(t1, eval_dir=t1_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)
    fm = _apply_boxdsc_dice_safeguard(fm, eval_dir=fmri_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)
    tr = _apply_boxdsc_dice_safeguard(tr, eval_dir=tracka_eval_dir, min_cluster_dice=boxdsc_min_cluster_dice)

    keep = ["subject_id", "detected_boxdsc", "pinpointed"]
    t1 = t1[keep].rename(columns={"detected_boxdsc": "detected_boxdsc_t1", "pinpointed": "pinpointed_t1"})
    fm = fm[keep].rename(columns={"detected_boxdsc": "detected_boxdsc_fmri", "pinpointed": "pinpointed_fmri"})
    tr = tr[keep].rename(columns={"detected_boxdsc": "detected_boxdsc_trackA", "pinpointed": "pinpointed_trackA"})

    merged = t1.merge(fm, on="subject_id", how="inner").merge(tr, on="subject_id", how="inner")
    if len(merged) != 38:
        raise ValueError(f"Expected 38 merged subjects, got {len(merged)}")

    # Clinical stratifiers (single authoritative table already keyed by subject_id)
    clinical = pd.read_csv(meta_csv)
    clinical = clinical[clinical["subject_id"].astype(str).isin(scope_subjects)].copy()
    cols = [
        "subject_id",
        "MRIDetectability",
        "SEEG",
        "DominantResectionCavityLocation_EZproxy",
    ]
    missing = [c for c in cols if c not in clinical.columns]
    if missing:
        raise KeyError(f"Missing clinical columns in {meta_csv}: {missing}")
    clinical = clinical[cols]

    merged = merged.merge(clinical, on="subject_id", how="left")
    return merged


def main() -> None:
    _setup_font()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    audit_path = PROJECT_DIR / "paper/revision/table3/final/table3_scope38_audit.json"
    subjects_tsv = PROJECT_DIR / "paper/revision/table3/final/table3_scope38_subject_ids.tsv"
    out_dir = PROJECT_DIR / "paper/revision/figure3"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    n_boot = int(audit.get("params", {}).get("n_boot", 10000))
    seed = int(audit.get("params", {}).get("seed", 20260209))
    boxdsc_min_cluster_dice = float(audit.get("params", {}).get("boxdsc_min_cluster_dice", 0.0) or 0.0)
    endpoint_desc = "boxDSC>0.22"
    if boxdsc_min_cluster_dice > 0.0:
        endpoint_desc = f"boxDSC>0.22 & Dice>{boxdsc_min_cluster_dice:g}"
    endpoint_label = f"Det({endpoint_desc})"

    scope_subjects = _read_tsv_column(subjects_tsv, "subject_id")
    if len(scope_subjects) != 38:
        raise ValueError(f"Expected 38 scope subjects, got {len(scope_subjects)}")

    t1_eval_dir = Path(audit["inputs"]["t1_eval_dir"])
    fmri_eval_dir = Path(audit["inputs"]["fmri_eval_dir"])
    tracka_eval_dir = Path(audit["inputs"]["tracka_eval_dir"])
    meta_csv = Path(audit["scope"]["meta_csv"])

    COLORS = _load_figure_colors()
    methods: List[Tuple[str, str, str]] = [
        ("T1-only", "t1", COLORS["t1"]),
        ("rs-fMRI-only", "fmri", COLORS["fmri"]),
        ("Track A", "trackA", COLORS["trackA"]),
    ]

    merged = _merge_scope38(
        scope_subjects=scope_subjects,
        t1_eval_dir=t1_eval_dir,
        fmri_eval_dir=fmri_eval_dir,
        tracka_eval_dir=tracka_eval_dir,
        meta_csv=meta_csv,
        boxdsc_min_cluster_dice=boxdsc_min_cluster_dice,
    )
    merged.to_csv(out_dir / "figure3_scope38_merged_eval_clinical.csv", index=False)

    # ---------------- Panel A: detectability stratified detection ----------------
    det_order = ["Intermediate", "Difficult"]
    det_label = {"Intermediate": "Intermediate", "Difficult": "Difficult"}
    det_rows: List[Dict[str, object]] = []
    det_p_by_group_raw: Dict[str, float] = {}
    for g in det_order:
        sub = merged[merged["MRIDetectability"].astype(str) == g].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1 = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        tr = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        n01 = int(np.sum((~t1) & tr))
        n10 = int(np.sum(t1 & (~tr)))
        det_p_by_group_raw[g] = float(_mcnemar_exact_pvalue(n01, n10))

        for _, key, _ in methods:
            k = int(np.sum(sub[f"detected_boxdsc_{key}"].astype(bool)))
            rate = float(k / n)
            ci_lo, ci_hi = _wilson_ci(k, n)
            det_rows.append({"group": g, "n": n, "method_key": key, "k_detected": k, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi})

    det_groups = [g for g in det_order if g in det_p_by_group_raw]
    det_p_adj = _holm_bonferroni(np.array([det_p_by_group_raw[g] for g in det_groups], dtype=float))
    det_p_by_group: Dict[str, float] = {g: float(p) for g, p in zip(det_groups, det_p_adj)}

    figA = plt.figure(figsize=(140.0 / MM_PER_INCH, 72.0 / MM_PER_INCH))
    axA = figA.add_subplot(1, 1, 1)
    _plot_detection_dotwhisker(
        axA,
        rows=det_rows,
        group_order=[g for g in det_order if any(r["group"] == g for r in det_rows)],
        group_label=det_label,
        method_order=methods,
        title="A  MRI detectability: detection rate (n=38)",
        ylabel=f"Detection rate ({endpoint_desc})",
        show_legend=True,
        p_by_group=det_p_by_group,
        p_footnote=None,
    )
    figA.tight_layout()

    # ---------------- Panel B: complementarity by detectability (T1 vs TrackA) ----------------
    comp_rows: List[Dict[str, object]] = []
    for g in det_order:
        sub = merged[merged["MRIDetectability"].astype(str) == g].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1 = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        tr = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        both = int(np.sum(t1 & tr))
        t1_only = int(np.sum(t1 & ~tr))
        tr_only = int(np.sum(~t1 & tr))
        neither = int(np.sum(~t1 & ~tr))
        comp_rows.append(
            {
                "group": g,
                "n": n,
                "count_both": both,
                "count_t1_only": t1_only,
                "count_trackA_only": tr_only,
                "count_neither": neither,
                "prop_both": float(both / n),
                "prop_t1_only": float(t1_only / n),
                "prop_trackA_only": float(tr_only / n),
                "prop_neither": float(neither / n),
            }
        )

    figB = plt.figure(figsize=(140.0 / MM_PER_INCH, 72.0 / MM_PER_INCH))
    axB = figB.add_subplot(1, 1, 1)
    _plot_complementarity_stacked(
        axB,
        rows=comp_rows,
        group_order=[g for g in det_order if any(r["group"] == g for r in comp_rows)],
        group_label=det_label,
        title="B  Complementarity by detectability (T1 vs Track A)",
        p_by_group=det_p_by_group,
        p_footnote=None,
    )
    figB.tight_layout()

    # ---------------- Panel C: SEEG stratified detection ----------------
    seeg_order = ["Yes", "No"]
    seeg_label = {"Yes": "SEEG yes", "No": "SEEG no"}
    seeg_rows: List[Dict[str, object]] = []
    seeg_p_by_group_raw: Dict[str, float] = {}
    for g in seeg_order:
        sub = merged[merged["SEEG"].astype(str) == g].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1 = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        tr = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        n01 = int(np.sum((~t1) & tr))
        n10 = int(np.sum(t1 & (~tr)))
        seeg_p_by_group_raw[g] = float(_mcnemar_exact_pvalue(n01, n10))
        for _, key, _ in methods:
            k = int(np.sum(sub[f"detected_boxdsc_{key}"].astype(bool)))
            rate = float(k / n)
            ci_lo, ci_hi = _wilson_ci(k, n)
            seeg_rows.append({"group": g, "n": n, "method_key": key, "k_detected": k, "rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi})

    seeg_groups = [g for g in seeg_order if g in seeg_p_by_group_raw]
    seeg_p_adj = _holm_bonferroni(np.array([seeg_p_by_group_raw[g] for g in seeg_groups], dtype=float))
    seeg_p_by_group: Dict[str, float] = {g: float(p) for g, p in zip(seeg_groups, seeg_p_adj)}

    figC = plt.figure(figsize=(140.0 / MM_PER_INCH, 72.0 / MM_PER_INCH))
    axC = figC.add_subplot(1, 1, 1)
    _plot_detection_dotwhisker(
        axC,
        rows=seeg_rows,
        group_order=[g for g in seeg_order if any(r["group"] == g for r in seeg_rows)],
        group_label=seeg_label,
        method_order=methods,
        title="C  SEEG: detection rate (n=38)",
        ylabel=f"Detection rate ({endpoint_desc})",
        show_legend=True,
        p_by_group=seeg_p_by_group,
        p_footnote=None,
    )
    figC.tight_layout()

    # ---------------- Panel D: location forest plot (Δ detection TrackA − T1) ----------------
    loc_map = {"Insular/Peri-insular (incl. operculum)": "Operculo-insular"}
    merged["LocationGroup"] = (
        merged["DominantResectionCavityLocation_EZproxy"]
        .astype(str)
        .map(loc_map)
        .fillna(merged["DominantResectionCavityLocation_EZproxy"].astype(str))
    )
    loc_order = ["Frontal", "Temporal", "Parietal", "Occipital", "Operculo-insular", "Central", "Cingulate", "Multilobar"]
    loc_rows: List[Dict[str, object]] = []
    for i, loc in enumerate(loc_order):
        sub = merged[merged["LocationGroup"].astype(str) == loc].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1b = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        trb = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        n01 = int(np.sum((~t1b) & trb))
        n10 = int(np.sum(t1b & (~trb)))
        p_loc = _mcnemar_exact_pvalue(n01, n10)
        delta = float(np.mean(trb.astype(float)) - np.mean(t1b.astype(float)))
        ci_lo, ci_hi = _paired_bootstrap_ci_diff(t1b.astype(float), trb.astype(float), n_boot=n_boot, seed=seed + 1000 + i)
        loc_rows.append({"location": loc, "n": n, "delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_mcnemar": p_loc})

    figD = plt.figure(figsize=(DEFAULT_FIGURE_WIDTH_MM / MM_PER_INCH, 90.0 / MM_PER_INCH))
    axD = figD.add_subplot(1, 1, 1)
    _plot_location_delta_forest(axD, rows=loc_rows, title="D  EZ-proxy location: Δ detection rate (Track A − T1)")
    figD.tight_layout()

    # ---------------- Panel E: Intermediate vs Difficult (T1-only + rs-fMRI-only) ----------------
    figE = plt.figure(figsize=(90.0 / MM_PER_INCH, 88.0 / MM_PER_INCH))
    gsE = figE.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.55)
    axE1 = figE.add_subplot(gsE[0, 0])
    axE2 = figE.add_subplot(gsE[1, 0], sharey=axE1)

    COLORS = _load_figure_colors()

    # Fisher exact p-values (Intermediate vs Difficult) per method → Holm correction within Panel E.
    def _p_fisher_for(col: str) -> float:
        detectability = merged["MRIDetectability"].astype(str)
        int_mask = detectability.eq("Intermediate").to_numpy()
        diff_mask = detectability.eq("Difficult").to_numpy()
        detected = merged[col].astype(bool).to_numpy()
        k_int = int(np.sum(int_mask & detected))
        n_int = int(np.sum(int_mask))
        k_diff = int(np.sum(diff_mask & detected))
        n_diff = int(np.sum(diff_mask))
        return float(_fisher_exact_two_sided(k_int, n_int - k_int, k_diff, n_diff - k_diff))

    p_raw = np.array(
        [
            _p_fisher_for("detected_boxdsc_t1"),
            _p_fisher_for("detected_boxdsc_fmri"),
        ],
        dtype=float,
    )
    p_adj = _holm_bonferroni(p_raw)
    _plot_intermediate_vs_difficult_bar(
        axE1,
        merged=merged,
        detected_col="detected_boxdsc_t1",
        color=COLORS["t1"],
        title="T1-only",
        p_value=float(p_adj[0]),
    )
    _plot_intermediate_vs_difficult_bar(
        axE2,
        merged=merged,
        detected_col="detected_boxdsc_fmri",
        color=COLORS["fmri"],
        title="rs-fMRI-only",
        p_value=float(p_adj[1]),
    )
    axE1.set_ylabel(f"Detection rate ({endpoint_label})", fontsize=10)
    axE2.set_ylabel("")
    axE1.tick_params(axis="x", bottom=False, labelbottom=False)
    figE.tight_layout()

    # ---------------- Save ----------------
    _save_panel(figA, out_dir / "panelA_detectability_scope38.pdf", out_dir / "panelA_detectability_scope38.png")
    _save_panel(figB, out_dir / "panelB_complementarity_scope38.pdf", out_dir / "panelB_complementarity_scope38.png")
    _save_panel(figC, out_dir / "panelC_seeg_scope38.pdf", out_dir / "panelC_seeg_scope38.png")
    _save_panel(figD, out_dir / "panelD_location_delta_scope38.pdf", out_dir / "panelD_location_delta_scope38.png")
    _save_panel(figE, out_dir / "panelE_intermediate_vs_difficult_scope38.pdf", out_dir / "panelE_intermediate_vs_difficult_scope38.png")

    (out_dir / "figure3_scope38_inputs.json").write_text(
        json.dumps(
            {
                "audit_json": _relpath(Path(audit_path)),
                "scope_subjects_tsv": _relpath(Path(subjects_tsv)),
                "meta_csv": _relpath(Path(meta_csv)),
                "eval_dirs": {
                    "t1": _relpath(Path(t1_eval_dir)),
                    "fmri": _relpath(Path(fmri_eval_dir)),
                    "trackA": _relpath(Path(tracka_eval_dir)),
                },
                "endpoint": f"detected_boxdsc ({endpoint_desc})",
                "ci_methods": {"rate": "Wilson", "delta_by_location": f"paired bootstrap (n_boot={n_boot}, seed={seed})"},
                "outputs": {
                    "panelA_pdf": _relpath(out_dir / "panelA_detectability_scope38.pdf"),
                    "panelB_pdf": _relpath(out_dir / "panelB_complementarity_scope38.pdf"),
                    "panelC_pdf": _relpath(out_dir / "panelC_seeg_scope38.pdf"),
                    "panelD_pdf": _relpath(out_dir / "panelD_location_delta_scope38.pdf"),
                    "panelE_pdf": _relpath(out_dir / "panelE_intermediate_vs_difficult_scope38.pdf"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
