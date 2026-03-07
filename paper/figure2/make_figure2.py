#!/usr/bin/env python3
"""
Generate Figure 2 (Main Results) in the manuscript layout requested by the user.

Panel A  Radar: vertex-level + cluster-level (boxDSC) summary metrics
Panel B  Radar: subject-level endpoints (Det(boxDSC), Det(distance), Det(PPV50), Recall@0.15, Pinpointing)
Panel C  Track A deploy gate (quadrant plot; single-column width; no pseudocode)
Panel D  Vertex-level DSC (paired): T1-only vs Track A
Panel E  Forest plot: Δ(Track A − T1) with 95% CI (cluster-level, boxDSC only)
Panel F  Subject-level endpoints (Detection + Pinpointing)
Panel G  Complementarity (2×2): T1-only vs Track A detection (with significance *)

Inputs:
  - paper/table2/table2_multiscale_performance.tsv
  - paper/table2/table2_source_data.csv
  - paper/table2/table2_audit.json

Outputs:
  - paper/figure2/figure2.pdf
  - paper/figure2/figure2.png
  - paper/figure2/figure2_inputs.json

Run:
  source env.sh
  python paper/figure2/make_figure2.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURE_COLORS_PATH = PROJECT_DIR / "paper/figure_colors.json"

# Publication sizing (default full-width: 180 mm)
DEFAULT_FIGURE_WIDTH_MM = 180.0
DEFAULT_FIGURE_HEIGHT_MM = 225.0  # ≤225 mm (allowed whitespace; reduces label overlap)
MM_PER_INCH = 25.4


def _load_figure_colors() -> Dict[str, str]:
    if FIGURE_COLORS_PATH.exists():
        try:
            return json.loads(FIGURE_COLORS_PATH.read_text())
        except Exception:
            pass
    # Fallback defaults (must match paper/FIGURE_COLOR_PALETTE.md)
    return {"t1": "#2ca02c", "fmri": "#1f77b4", "trackA": "#9467bd", "gt": "#d62728", "overlap": "#ffbf00"}


@dataclass(frozen=True)
class DeltaCI:
    delta: float
    lo: float
    hi: float
    raw: str


_DELTA_CI_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*$"
)


def _parse_delta_ci(cell: str) -> DeltaCI:
    raw = str(cell)
    s = raw
    scale = 1.0
    if "%" in s:
        s = s.replace("%", "")
        scale = 0.01
    m = _DELTA_CI_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse delta CI cell: {cell!r}")
    return DeltaCI(
        delta=float(m.group(1)) * scale,
        lo=float(m.group(2)) * scale,
        hi=float(m.group(3)) * scale,
        raw=raw,
    )


def _read_table2_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _get_row(df: pd.DataFrame, metric_contains: str, *, scale_contains: Optional[str] = None) -> pd.Series:
    m = df["Metric"].astype(str).str.contains(metric_contains, regex=False)
    if scale_contains is not None:
        m = m & df["Scale"].astype(str).str.contains(scale_contains, regex=False)
    hits = df[m]
    if len(hits) != 1:
        raise ValueError(
            f"Expected 1 row containing metric={metric_contains!r} (scale={scale_contains!r}), got {len(hits)}"
        )
    return hits.iloc[0]


def _extract_p_value(cell: str) -> Optional[float]:
    s = str(cell).strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_point_estimate(cell: str) -> Tuple[Optional[float], bool]:
    """
    Extract point estimates from Table 2 formatted cells.

    Supported patterns:
      - "33/52 (63.5%) [..]" -> (63.5, True)
      - "47.8% [..]" -> (47.8, True)
      - "0.367 [..]" -> (0.367, False)
    """
    s = str(cell).strip()
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", s)
    if m:
        return (float(m.group(1)), True)
    m = re.search(r"([0-9]*\.?[0-9]+)\s*%", s)
    if m:
        return (float(m.group(1)), True)
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)), False)
    return (None, False)


def _format_point_estimate(x: Optional[float], *, is_percent: bool) -> str:
    if x is None:
        return "NA"
    if is_percent:
        return f"{x:.1f}%"
    return f"{x:.2f}" if abs(x) >= 1.0 else f"{x:.3f}"


def _format_delta_ci_text(d: DeltaCI) -> str:
    if "%" in str(d.raw):
        return f"{d.delta*100.0:+.1f}% [{d.lo*100.0:.1f}, {d.hi*100.0:.1f}]"
    return f"{d.delta:+.3f} [{d.lo:.3f}, {d.hi:.3f}]"


def _wrap_rate_label(s: str) -> str:
    # "33/52 (63.5%) [50.0, 76.9]" -> "33/52 (63.5%)\n[50.0, 76.9]"
    s = str(s)
    if " [" in s:
        return s.replace(" [", "\n[", 1)
    return s


def _setup_font() -> None:
    """
    Set a consistent sans-serif font.

    Arial may not exist on HPC nodes. If you need strict Arial, provide a TTF file and set:
      MELD_PAPER_FONT_TTF=/path/to/Arial.ttf
    """
    def _find_arial_ttf() -> Optional[str]:
        # 1) Explicit override
        env_ttf = os.environ.get("MELD_PAPER_FONT_TTF")
        if env_ttf and Path(env_ttf).exists():
            return env_ttf

        # 2) Repo-local convention (no admin needed)
        for p in [
            PROJECT_DIR / "paper" / "fonts" / "Arial.ttf",
            PROJECT_DIR / "paper" / "fonts" / "arial.ttf",
        ]:
            if p.exists():
                return str(p)

        # 3) Conda-forge `mscorefonts` installs into $CONDA_PREFIX/fonts/arial.ttf
        prefixes = []
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            prefixes.append(Path(conda_prefix))
        prefixes.append(Path(sys.prefix))
        for prefix in prefixes:
            for name in ["arial.ttf", "Arial.ttf"]:
                p = prefix / "fonts" / name
                if p.exists():
                    return str(p)
        return None

    font_ttf = _find_arial_ttf()
    if font_ttf and Path(font_ttf).exists():
        try:
            font_manager.fontManager.addfont(font_ttf)
            name = font_manager.FontProperties(fname=font_ttf).get_name()
            plt.rcParams.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": [name],
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                }
            )
            return
        except Exception:
            pass

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Hard-check Arial availability (for publication). If missing, keep fallback but warn loudly.
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except Exception:
        print(
            "[Figure2] WARNING: Arial is not available; falling back to DejaVu Sans.\n"
            "To use Arial:\n"
            "  1) Copy a licensed `Arial.ttf` (e.g., from Windows: C:\\\\Windows\\\\Fonts\\\\arial.ttf)\n"
            "     to `paper/fonts/Arial.ttf` (or any path), then run with:\n"
            "       export MELD_PAPER_FONT_TTF=/path/to/Arial.ttf\n"
            "  2) Or (Ubuntu, requires sudo) install Microsoft core fonts:\n"
            "       sudo apt-get update && sudo apt-get install -y ttf-mscorefonts-installer\n"
            "       fc-cache -f -v\n",
            file=sys.stderr,
        )


def _save_inputs(out_dir: Path, payload: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "figure2_inputs.json").open("w") as f:
        json.dump(payload, f, indent=2)


def _radar_plot(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    labels: Tuple[str, ...],
    color: str,
    title: str,
) -> None:
    n = len(labels)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    vals = np.concatenate([values, [values[0]]])
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, color=color, alpha=0.18)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(title, fontsize=11, pad=10)


def _radar_plot_multi(
    ax: plt.Axes,
    *,
    values_by_method: List[Tuple[str, np.ndarray, str]],
    labels: Tuple[str, ...],
    title: str,
    rmax: float = 1.0,
) -> List[Line2D]:
    n = len(labels)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    rmax = float(rmax)
    if not (0.0 < rmax):
        raise ValueError(f"Invalid rmax for radar plot: {rmax}")
    ax.set_ylim(0, rmax)
    if rmax <= 1.0:
        ticks = [0.2, 0.4, 0.6, 0.8]
        if rmax < 0.95:
            ticks = [t for t in ticks if t < (rmax - 1e-6)] + [rmax]
        else:
            ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        if abs(rmax - 1.0) > 1e-6:
            ticks.append(rmax)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_title(title, fontsize=11, pad=8)

    handles: List[matplotlib.lines.Line2D] = []
    for name, vals0, color in values_by_method:
        vals0 = np.asarray(vals0, dtype=float)
        vals = np.concatenate([vals0, [vals0[0]]])
        (h,) = ax.plot(angles, vals, color=color, lw=2.2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.10)
        handles.append(h)
    return handles


def main() -> None:
    table2_tsv = PROJECT_DIR / "paper/table2/table2_multiscale_performance.tsv"
    source_csv = PROJECT_DIR / "paper/table2/table2_source_data.csv"
    audit_json = PROJECT_DIR / "paper/table2/table2_audit.json"
    out_dir = PROJECT_DIR / "paper/figure2"
    out_pdf = out_dir / "figure2.pdf"
    out_png = out_dir / "figure2.png"

    df_table = _read_table2_tsv(table2_tsv)
    df = pd.read_csv(source_csv)
    audit = json.loads(audit_json.read_text())
    track_eval_dir = Path(audit["inputs"]["tracka_eval_dir"])
    t1_eval_dir = Path(audit["inputs"]["t1_eval_dir"])
    fmri_eval_dir = Path(audit["inputs"]["fmri_eval_dir"])

    # ---------- Colors (single source of truth) ----------
    COLORS = _load_figure_colors()

    # ---------- Panel A (radars): audited point estimates ----------
    pe = audit["point_estimates"]
    method_specs = [("T1-only", "t1"), ("fMRI-only", "fmri"), ("Track A", "trackA")]

    def _sig(metric_contains: str, *, scale_contains: Optional[str] = None) -> bool:
        p = _extract_p_value(str(_get_row(df_table, metric_contains, scale_contains=scale_contains)["p (TrackA vs T1)"]))
        return (p is not None) and (p < 0.05)

    # Subject-level Det(distance) is not in Table 2 TSV; read it from three-level eval outputs.
    def _resolve_subject_level_csv(eval_dir: Path) -> Path:
        for cand in [eval_dir / "all_folds_subject_level_results.csv", eval_dir / "subject_level_results.csv"]:
            if cand.exists():
                return cand
        fp = eval_dir / "fold0" / "val" / "subject_level_results.csv"
        if fp.exists():
            return fp
        raise FileNotFoundError(f"Cannot resolve subject_level_results.csv under: {eval_dir}")

    sids = df["subject_id"].astype(str).tolist()
    t1_sub = pd.read_csv(_resolve_subject_level_csv(t1_eval_dir)).set_index("subject_id").loc[sids].reset_index()
    fmri_sub = pd.read_csv(_resolve_subject_level_csv(fmri_eval_dir)).set_index("subject_id").loc[sids].reset_index()
    track_sub = pd.read_csv(_resolve_subject_level_csv(track_eval_dir)).set_index("subject_id").loc[sids].reset_index()

    detdist_rate_by_key = {
        "t1": float(np.mean(t1_sub["detected_distance"].astype(bool))),
        "fmri": float(np.mean(fmri_sub["detected_distance"].astype(bool))),
        "trackA": float(np.mean(track_sub["detected_distance"].astype(bool))),
    }

    radar1_axes = (
        "Vertex DSC" + ("*" if _sig("Dice similarity coefficient (DSC)", scale_contains="Vertex-level") else ""),
        "Cluster precision",
        "Cluster sensitivity" + ("*" if _sig("Sensitivity (≥1 TP cluster)", scale_contains="Cluster-level (boxDSC") else ""),
        "Cluster F1" + ("*" if _sig("F1 score", scale_contains="Cluster-level (boxDSC") else ""),
        "FP/pt",
    )
    radar2_axes = (
        "Det(boxDSC)" + ("*" if _sig("Detection rate (Det(boxDSC>0.22))", scale_contains="Subject-level") else ""),
        "Det(dist)",
        "Det(PPV50)",
        "Recall@0.15",
        "Pinpoint",
    )

    radar1_values: Dict[str, np.ndarray] = {}
    radar2_values: Dict[str, np.ndarray] = {}
    for _, key in method_specs:
        m = pe[key]
        radar1_values[key] = np.array(
            [
                float(m["vertex_dsc_mean"]),
                float(m["precision_boxdsc"]),
                float(m["sensitivity_boxdsc"]),
                float(m["f1_boxdsc"]),
                float(m["fp_per_patient_mean"]),
            ],
            dtype=float,
        )
        radar2_values[key] = np.array(
            [
                float(m["detection_rate_boxdsc"]),
                float(detdist_rate_by_key[key]),
                float(m["detection_rate_ppv50"]),
                float(m["recall_at_0.15"]),
                float(m["pinpointing_rate"]),
            ],
            dtype=float,
        )

    # Radial maxima (allow >1.0 for FP/pt on Panel A).
    radar1_max = float(max(float(np.max(v)) for v in radar1_values.values()))
    radar2_max = float(max(float(np.max(v)) for v in radar2_values.values()))
    radar1_rmax = max(0.90, radar1_max + 0.05)
    radar2_rmax = min(1.0, max(0.90, radar2_max + 0.05))

    # ---------- Panel E (forest): cluster-level deltas (boxDSC) from Table 2 ----------
    metric_specs = [
        ("FP/pt", "False-positive clusters per subject", "Cluster-level (boxDSC>0.22)"),
        ("Precision", "Precision (TP clusters / all clusters)", "Cluster-level (boxDSC>0.22)"),
        ("Sensitivity", "Sensitivity (≥1 TP cluster)", "Cluster-level (boxDSC>0.22)"),
        ("F1", "F1 score", "Cluster-level (boxDSC>0.22)"),
    ]
    deltas = []
    for label, key, scale in metric_specs:
        row = _get_row(df_table, key, scale_contains=scale)
        d = _parse_delta_ci(row["Δ (TrackA − T1)"])
        p_cell = str(row["p (TrackA vs T1)"])
        p = _extract_p_value(p_cell)
        sig = (p is not None) and (p < 0.05)
        deltas.append((label, d, p_cell, sig))

    # ---------- Panel E: subject endpoints labels from Table 2 ----------
    row_det = _get_row(df_table, "Detection rate (Det(boxDSC>0.22))")
    det_t1 = str(row_det["T1-only"])
    det_fmri = str(row_det["fMRI-only (fMRIhemi B2a)"])
    det_track = str(row_det["Track A (T1+fMRI)"])
    det_p = str(row_det["p (TrackA vs T1)"])

    row_pin = _get_row(df_table, "Pinpointing rate")
    pin_t1 = str(row_pin["T1-only"])
    pin_fmri = str(row_pin["fMRI-only (fMRIhemi B2a)"])
    pin_track = str(row_pin["Track A (T1+fMRI)"])
    pin_p = str(row_pin["p (TrackA vs T1)"])

    # ---------- Panel C: paired DSC ----------
    dsc_t1 = df["t1_vertex_dsc_lesion_hemi"].astype(float).to_numpy()
    dsc_fmri = df["fmri_vertex_dsc_lesion_hemi"].astype(float).to_numpy()
    dsc_track = df["trackA_vertex_dsc_lesion_hemi"].astype(float).to_numpy()

    # ---------- Panel F: complementarity (primary endpoint Det(boxDSC)) ----------
    t1_found = df["t1_detected_boxdsc"].astype(bool).to_numpy()
    fmri_found = df["fmri_detected_boxdsc"].astype(bool).to_numpy()
    track_found = df["trackA_detected_boxdsc"].astype(bool).to_numpy()
    n11 = int(np.sum(t1_found & track_found))
    n10 = int(np.sum(t1_found & ~track_found))  # hurt
    n01 = int(np.sum(~t1_found & track_found))  # rescued
    n00 = int(np.sum(~t1_found & ~track_found))

    # ---------- Plot ----------
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

    width_mm = float(os.environ.get("MELD_PAPER_FIGURE_WIDTH_MM", str(DEFAULT_FIGURE_WIDTH_MM)))
    height_mm = float(os.environ.get("MELD_PAPER_FIGURE_HEIGHT_MM", str(DEFAULT_FIGURE_HEIGHT_MM)))
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))
    gs = fig.add_gridspec(
        4,
        2,
        left=0.10,
        right=0.99,
        bottom=0.05,
        top=0.92,
        height_ratios=[1.05, 0.95, 1.05, 0.90],
        width_ratios=[1.0, 1.0],
        wspace=0.35,
        hspace=0.55,
    )

    # Panel A: radar (vertex + cluster; boxDSC)
    axA = fig.add_subplot(gs[0, 0], projection="polar")
    handles = _radar_plot_multi(
        axA,
        values_by_method=[(name, radar1_values[key], COLORS[key]) for name, key in method_specs],
        labels=radar1_axes,
        title="A  Vertex + cluster (boxDSC)",
        rmax=radar1_rmax,
    )

    # Panel B: radar (subject-level endpoints)
    axB = fig.add_subplot(gs[0, 1], projection="polar")
    _radar_plot_multi(
        axB,
        values_by_method=[(name, radar2_values[key], COLORS[key]) for name, key in method_specs],
        labels=radar2_axes,
        title="B  Subject endpoints",
        rmax=radar2_rmax,
    )

    # Keep legend inside the fixed-size canvas (avoid bbox_inches="tight" cropping).
    fig.legend(
        handles=handles,
        labels=[name for name, _ in method_specs],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
        ncol=3,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.2,
    )

    # Panel C: gate quadrant plot (single-column width; no pseudocode)
    axC = fig.add_subplot(gs[1, 0])
    axC.set_title("C  Deploy gate", loc="left", fontsize=11, pad=6)
    gate_csv = track_eval_dir / "all_folds_gate_decisions.csv"
    gate = pd.read_csv(gate_csv)

    gate["t1_conf"] = gate["t1_conf"].astype(float)
    gate["fmri_minus_t1"] = gate["fmri_minus_t1"].astype(float)
    thr_conf = float(gate["t1_conf_threshold"].iloc[0])
    thr_diff = 0.15
    # Decision markers (compact: color encodes decision)
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

    # Quadrant boundaries (gate thresholds)
    axC.axvline(thr_conf, color="black", lw=1.3, alpha=0.55)
    axC.axhline(thr_diff, color="black", lw=1.0, alpha=0.50, linestyle="--")

    axC.set_xlim(0.0, 1.0)
    axC.set_ylim(min(-0.20, float(gate["fmri_minus_t1"].min()) - 0.03), float(gate["fmri_minus_t1"].max()) + 0.05)
    axC.set_xlabel("T1 confidence")
    axC.set_ylabel("fMRI advantage (Δ)")
    axC.grid(True, linestyle=":", alpha=0.35)
    axC.tick_params(axis="y", pad=2)

    # Minimal annotations (avoid pseudocode blocks)
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

    # Panel D: paired DSC (T1 vs TrackA)
    axD = fig.add_subplot(gs[1, 1])
    axD.set_title("D  Vertex DSC (paired)", loc="left", fontsize=11)
    order = np.argsort(dsc_t1)
    for i in order:
        axD.plot([0, 1], [dsc_t1[i], dsc_track[i]], color="black", alpha=0.15, lw=0.6)
    axD.scatter(np.zeros_like(dsc_t1), dsc_t1, s=18, color=COLORS["t1"], alpha=0.85, label="T1-only", edgecolor="none")
    axD.scatter(np.ones_like(dsc_t1), dsc_track, s=18, color=COLORS["trackA"], alpha=0.85, label="Track A", edgecolor="none")
    axD.set_xlim(-0.3, 1.3)
    axD.set_xticks([0, 1])
    axD.set_xticklabels(["T1-only", "Track A"])
    axD.set_ylabel("DSC")
    axD.set_ylim(0, max(0.9, float(np.max([dsc_t1.max(), dsc_track.max(), dsc_fmri.max()])) + 0.05))
    axD.grid(axis="y", linestyle=":", alpha=0.4)
    axD.legend(loc="upper left", frameon=False)

    # Panel E: forest plot (cluster-level; boxDSC only)
    axE = fig.add_subplot(gs[2, :])
    axE.set_title("E  Cluster-level effects (boxDSC; Δ Track A − T1)", loc="left", fontsize=11)
    y = np.arange(len(deltas))[::-1]
    axE.axvline(0, color="black", lw=1, alpha=0.6)
    span = max(1e-6, float(max(d.hi for _, d, _, _ in deltas) - min(d.lo for _, d, _, _ in deltas)))
    for i, (lbl, d, _, sig) in enumerate(deltas):
        yi = y[i]
        axE.plot([d.lo, d.hi], [yi, yi], color=COLORS["trackA"], lw=2)
        axE.plot(d.delta, yi, marker="o", color=COLORS["trackA"])
        txt = _format_delta_ci_text(d) + (" *" if sig else "")
        axE.text(
            d.hi + 0.04 * span,
            yi,
            txt,
            va="center",
            ha="left",
            fontsize=9,
        )
    axE.set_yticks(y)
    axE.set_yticklabels([lbl for (lbl, _, _, _) in deltas])
    axE.set_xlabel("Δ (Track A − T1)")
    axE.tick_params(axis="y", pad=2)
    lo_all = min(d.lo for _, d, _, _ in deltas)
    hi_all = max(d.hi for _, d, _, _ in deltas)
    pad = 0.05 * max(1e-6, hi_all - lo_all)
    axE.set_xlim(lo_all - pad, hi_all + 0.55 * (hi_all - lo_all + 1e-6))
    axE.grid(axis="x", linestyle=":", alpha=0.4)
    axE.text(0.0, -0.12, "* p<0.05 (paired; Table 2)", transform=axE.transAxes, fontsize=9, ha="left")

    # Panel F: subject-level endpoints bars (original Panel E)
    axF = fig.add_subplot(gs[3, 0])
    axF.set_title("F  Subject endpoints", loc="left", fontsize=11)
    t1_pin = df["t1_pinpointed"].astype(bool).to_numpy()
    fmri_pin = df["fmri_pinpointed"].astype(bool).to_numpy()
    track_pin = df["trackA_pinpointed"].astype(bool).to_numpy()

    groups = ["Detection", "Pinpointing"]
    x = np.arange(len(groups), dtype=float)
    width = 0.24
    vals_t1 = [float(np.mean(t1_found)), float(np.mean(t1_pin))]
    vals_fm = [float(np.mean(fmri_found)), float(np.mean(fmri_pin))]
    vals_tr = [float(np.mean(track_found)), float(np.mean(track_pin))]

    axF.bar(x - width, vals_t1, width=width, color=COLORS["t1"], label="T1-only")
    axF.bar(x, vals_fm, width=width, color=COLORS["fmri"], label="fMRI-only")
    axF.bar(x + width, vals_tr, width=width, color=COLORS["trackA"], label="Track A")
    axF.set_ylim(0, 1.0)
    axF.set_xticks(x)
    axF.set_xticklabels(groups)
    axF.set_ylabel("Rate")
    axF.grid(axis="y", linestyle=":", alpha=0.4)
    axF.legend(loc="lower right", frameon=False)
    axF.text(0.02, 0.02, "Primary: Det(boxDSC>0.22)", transform=axF.transAxes, fontsize=8.0, va="bottom")

    # Do not print numeric labels on bars (per manuscript style).

    det_p_num = _extract_p_value(det_p)
    if det_p_num is not None and det_p_num < 0.05:
        y0 = max(vals_t1[0], vals_tr[0]) + 0.10
        h = 0.02
        axF.plot([x[0] - width, x[0] - width, x[0] + width, x[0] + width], [y0, y0 + h, y0 + h, y0], color="black", lw=1)
        axF.text(x[0], y0 + h + 0.01, "*", ha="center", va="bottom", fontsize=14)

    # Panel G: complementarity table (+ significance star)
    axG = fig.add_subplot(gs[3, 1])
    det_p_num = _extract_p_value(det_p)
    star = "*" if (det_p_num is not None and det_p_num < 0.05) else ""
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
        if r == 1 and c == 2:  # rescued
            cell.set_facecolor("#E8F0FE")
        if r == 2 and c == 1:  # hurt
            cell.set_facecolor("#FCE8E6")
    note = f"rescued={n01}, hurt={n10}"
    if det_p_num is not None:
        note += f"; p={det_p_num:.4f}"
    axG.text(0.5, 0.10, note, ha="center", va="center", transform=axG.transAxes, fontsize=10)

    fig.suptitle("Figure 2. Main results (n=52)", y=0.992, fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    _save_inputs(
        out_dir,
        {
            "table2_multiscale_performance_tsv": str(table2_tsv),
            "table2_source_data_csv": str(source_csv),
            "table2_audit_json": str(audit_json),
            "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
            "complementarity": {"n00": n00, "n01_rescued": n01, "n10_hurt": n10, "n11": n11},
            "panelF": {"detection_p": det_p, "pinpointing_p": pin_p},
            "radar1_axes": list(radar1_axes),
            "radar2_axes": list(radar2_axes),
            "radar1_point_estimates": {k: radar1_values[k].tolist() for _, k in method_specs},
            "radar2_point_estimates": {k: radar2_values[k].tolist() for _, k in method_specs},
            "detected_distance_rate": detdist_rate_by_key,
            "gate_csv": str(gate_csv),
            "figure_colors_json": str(FIGURE_COLORS_PATH),
            "figure_colors": COLORS,
        },
    )


if __name__ == "__main__":
    main()
