#!/usr/bin/env python3
"""
Generate Figure 3 (Clinical Subgroups) following `paper/figure3/figure3design.txt`.

Panels
  A  MRI detectability stratified subject-level detection rate (Det boxDSC>0.22)
  B  Non-difficult vs difficult (T1-only + rs-fMRI-only; Det boxDSC>0.22; Fisher exact)
  C  Complementarity (Track A vs T1) by MRI detectability (both / hurt / rescued / neither)
  D  SEEG stratified subject-level detection rate (Det boxDSC>0.22)
  E  EZ-proxy location forest plot: Δ detection rate (Track A − T1) with 95% CI (exploratory)

Methods (colors, fixed across figures; see `paper/figure_colors.json`)
  - T1-only baseline (no post-hoc area/cluster constraints; deployment evaluation)
  - rs-fMRI-only (fMRIhemi B2a; 30/80 budgets)
  - Track A (Route A; threshold gate + tiered-low injection; 30/80 budgets on fMRI takeover)

Inputs (single source of truth)
  - `paper/table2/table2_audit.json` (locates the exact eval dirs used in Table 2)
  - `paper/supplement_table_hs_patient_characteristics.csv` (clinical stratifiers)

Outputs (generated)
  - `paper/figure3/figure3.pdf`
  - `paper/figure3/figure3.png`
  - `paper/figure3/figure3_inputs.json`
  - `paper/figure3/figure3_merged_eval_clinical.csv`
  - `paper/figure3/figure3_detectability_detection.tsv`
  - `paper/figure3/figure3_detectability_complementarity.tsv`
  - `paper/figure3/figure3_seeg_detection.tsv`
  - `paper/figure3/figure3_location_delta_forest.tsv`

Run:
  source env.sh
  python paper/figure3/make_figure3.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURE_COLORS_PATH = PROJECT_DIR / "paper/figure_colors.json"

DEFAULT_FIGURE_WIDTH_MM = 180.0
DEFAULT_FIGURE_HEIGHT_MM = 225.0  # ≤225 mm (allowed whitespace)
MM_PER_INCH = 25.4

_PANEL_B_AXES_SIZE_CACHE_IN: Optional[Tuple[float, float]] = None


def _panel_b_axes_size_in() -> Tuple[float, float]:
    """
    Return the axes-box (window extent) size for Figure3 Panel B (inches).

    We use the axes box (not tightbbox) because the request is to match the *plot area* size.
    """

    global _PANEL_B_AXES_SIZE_CACHE_IN
    if _PANEL_B_AXES_SIZE_CACHE_IN is not None:
        return _PANEL_B_AXES_SIZE_CACHE_IN

    fig0 = plt.figure(figsize=(DEFAULT_FIGURE_WIDTH_MM / MM_PER_INCH, DEFAULT_FIGURE_HEIGHT_MM / MM_PER_INCH))
    gs0 = fig0.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.05, 0.95, 1.15],
        width_ratios=[1.0, 1.0],
        left=0.16,
        right=0.94,
        top=0.97,
        bottom=0.06,
        hspace=0.65,
        wspace=0.50,
    )
    axb = fig0.add_subplot(gs0[1, 0])
    fig0.canvas.draw()
    bbox = axb.get_window_extent().transformed(fig0.dpi_scale_trans.inverted())
    w_in, h_in = float(bbox.width), float(bbox.height)
    plt.close(fig0)
    _PANEL_B_AXES_SIZE_CACHE_IN = (w_in, h_in)
    return _PANEL_B_AXES_SIZE_CACHE_IN


def _load_figure_colors() -> Dict[str, str]:
    if FIGURE_COLORS_PATH.exists():
        try:
            return json.loads(FIGURE_COLORS_PATH.read_text())
        except Exception:
            pass
    return {"t1": "#2ca02c", "fmri": "#1f77b4", "trackA": "#9467bd", "gt": "#d62728", "overlap": "#ffbf00"}


def _setup_font() -> None:
    """
    Use a consistent sans-serif font.

    Arial may not exist on HPC nodes. If you need strict Arial, provide a TTF file and set:
      MELD_PAPER_FONT_TTF=/path/to/Arial.ttf
    """

    def _find_arial_ttf() -> Optional[str]:
        env_ttf = os.environ.get("MELD_PAPER_FONT_TTF")
        if env_ttf and Path(env_ttf).exists():
            return env_ttf

        for p in [
            PROJECT_DIR / "paper" / "fonts" / "Arial.ttf",
            PROJECT_DIR / "paper" / "fonts" / "arial.ttf",
        ]:
            if p.exists():
                return str(p)

        # conda-forge `mscorefonts` convention
        prefixes: List[Path] = []
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
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except Exception:
        print(
            "[Figure3] WARNING: Arial is not available; falling back to DejaVu Sans.\n"
            "To use Arial:\n"
            "  1) Install via conda-forge (license: PROPRIETARY):\n"
            "       mamba install -c conda-forge -y mscorefonts\n"
            "  2) Or copy a licensed `Arial.ttf` to `paper/fonts/Arial.ttf`, or export:\n"
            "       export MELD_PAPER_FONT_TTF=/path/to/Arial.ttf\n",
            file=sys.stderr,
        )


def _write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in fieldnames) + "\n")


def _subject_to_bbb(subject_id: str) -> int:
    """
    Convert a subject_id to a deterministic integer for plotting/labeling.

    Public-release builds use hashed IDs (e.g., `sub-<hex>`); legacy internal IDs may include digits.
    """
    s = str(subject_id).strip()
    if "_" in s:
        s = s.split("_", 1)[0]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return int(digits)
    # Hashed IDs: fall back to parsing the hex substring (stable but non-identifying).
    hex_chars = "".join(ch for ch in s.lower() if ch in "0123456789abcdef")
    if not hex_chars:
        raise ValueError(f"Cannot derive numeric label from subject_id: {subject_id!r}")
    return int(hex_chars[:8], 16)


def _load_eval_csv(eval_dir: Path, filename: str, *, prefer_all_folds: bool = True) -> pd.DataFrame:
    """
    Load an eval CSV from one of these layouts:
      - all_folds_{filename} (preferred)
      - {filename} (flat)
      - fold{0-4}/val/{filename} (folded)
    """
    if prefer_all_folds:
        p = eval_dir / f"all_folds_{filename}"
        if p.exists():
            return pd.read_csv(p)
    p = eval_dir / filename
    if p.exists():
        return pd.read_csv(p)
    rows = []
    for fold in range(5):
        fp = eval_dir / f"fold{fold}" / "val" / filename
        if not fp.exists():
            raise FileNotFoundError(f"Missing expected eval file: {fp}")
        df = pd.read_csv(fp)
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _wilson_ci(k: int, n: int, *, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054  # 97.5% quantile
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def _mcnemar_exact_pvalue(n01: int, n10: int) -> float:
    """
    Exact McNemar (two-sided) using the binomial distribution on discordant pairs.

    n01: count(T1=0, TrackA=1)
    n10: count(T1=1, TrackA=0)
    """
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n <= 0:
        return 1.0
    k = min(n01, n10)
    # Binomial CDF with p=0.5: sum_{i=0..k} C(n,i) / 2^n
    denom = 2**n
    cdf_num = 0
    for i in range(k + 1):
        cdf_num += math.comb(n, i)
    p = 2.0 * (cdf_num / denom)
    return float(min(1.0, p))


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher's exact test p-value for a 2x2 table:
      [[a, b],
       [c, d]]

    Implemented via exact enumeration over all tables with the same margins,
    summing probabilities <= observed (SciPy-style).
    """
    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)
    n1 = a + b
    n2 = c + d
    m1 = a + c
    m2 = b + d
    n = n1 + n2
    if n <= 0:
        return 1.0

    # Support for a with fixed margins:
    amin = max(0, m1 - n2)
    amax = min(m1, n1)
    denom = math.comb(n, n1)

    def _pmf(x: int) -> float:
        return (math.comb(m1, x) * math.comb(m2, n1 - x)) / denom

    p_obs = _pmf(a)
    p = 0.0
    for x in range(int(amin), int(amax) + 1):
        px = _pmf(int(x))
        if px <= p_obs + 1e-12:
            p += px
    return float(min(1.0, p))


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _paired_bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float]:
    """
    Paired bootstrap CI for mean(b - a).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    n = int(a.shape[0])
    if n <= 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(b[idx] - a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(lo), float(hi))


def _fmt_ci(lo: float, hi: float, *, decimals: int = 3) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{lo:.{decimals}f},{hi:.{decimals}f}]"


def _cochran_armitage_trend_p(*, k: List[int], n: List[int], scores: Optional[List[float]] = None) -> float:
    """
    Cochran–Armitage trend test for a binary outcome across ordered groups.

    Args:
      k: successes per group
      n: totals per group
      scores: optional numeric scores (default 0..K-1)

    Returns:
      two-sided p-value (normal approximation).
    """
    if len(k) != len(n) or len(k) < 2:
        return 1.0
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    if scores is None:
        s_arr = np.arange(len(k), dtype=float)
    else:
        if len(scores) != len(k):
            raise ValueError("scores must have same length as k/n")
        s_arr = np.asarray(scores, dtype=float)

    N = float(np.sum(n_arr))
    if N <= 0:
        return 1.0
    p_hat = float(np.sum(k_arr) / N)
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 1.0

    T = float(np.sum(s_arr * (k_arr - n_arr * p_hat)))
    var = float(p_hat * (1.0 - p_hat) * (np.sum(n_arr * (s_arr**2)) - (np.sum(n_arr * s_arr) ** 2) / N))
    if var <= 0.0:
        return 1.0
    z = T / math.sqrt(var)
    # two-sided normal p-value
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(min(1.0, max(0.0, p)))

def _merge_eval_with_clinical(*, audit: dict, clinical_path: Path) -> pd.DataFrame:
    t1_eval_dir = Path(audit["inputs"]["t1_eval_dir"])
    fmri_eval_dir = Path(audit["inputs"]["fmri_eval_dir"])
    track_eval_dir = Path(audit["inputs"]["tracka_eval_dir"])

    subj_t1 = _load_eval_csv(t1_eval_dir, "subject_level_results.csv")
    subj_fmri = _load_eval_csv(fmri_eval_dir, "subject_level_results.csv")
    subj_track = _load_eval_csv(track_eval_dir, "subject_level_results.csv")

    need = ["subject_id", "split", "is_lesion", "detected_boxdsc", "detected_ppv50", "pinpointed"]
    for name, df in [("t1", subj_t1), ("fmri", subj_fmri), ("trackA", subj_track)]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {name} subject-level results: {missing}")

    def _prep(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        out = df[need].copy()
        out = out[out["split"].astype(str) == "val"].copy()
        out = out[out["is_lesion"].astype(bool)].copy()
        out.rename(
            columns={
                "detected_boxdsc": f"detected_boxdsc_{suffix}",
                "detected_ppv50": f"detected_ppv50_{suffix}",
                "pinpointed": f"pinpointed_{suffix}",
            },
            inplace=True,
        )
        return out

    t1 = _prep(subj_t1, "t1")
    fmri = _prep(subj_fmri, "fmri")
    track = _prep(subj_track, "trackA")

    merged = t1.merge(fmri, on=["subject_id", "split", "is_lesion"], how="inner").merge(
        track, on=["subject_id", "split", "is_lesion"], how="inner"
    )
    merged["BBB"] = merged["subject_id"].map(_subject_to_bbb).astype(int)

    clinical = pd.read_csv(clinical_path)
    if "BBB" not in clinical.columns:
        raise KeyError("Clinical CSV missing required column: BBB")
    clinical["BBB"] = clinical["BBB"].astype(int)

    merged = merged.merge(clinical, on="BBB", how="left")
    return merged


def _plot_detection_dotwhisker(
    ax: plt.Axes,
    *,
    rows: List[Dict[str, object]],
    group_order: List[str],
    group_label: Dict[str, str],
    method_order: List[Tuple[str, str, str]],
    title: str,
    ylabel: str,
    show_legend: bool,
    p_by_group: Optional[Dict[str, float]] = None,
    p_footnote: Optional[str] = "* p<0.05, ** p<0.01 (unadjusted)",
) -> None:
    x = np.arange(len(group_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(method_order))
    offset_by_key = {key: float(off) for (_, key, _), off in zip(method_order, offsets)}

    n_by_group = {r["group"]: int(r["n"]) for r in rows if r["method_key"] == method_order[0][1]}
    for (name, key, color), off in zip(method_order, offsets):
        sub = [r for r in rows if r["method_key"] == key]
        y = np.array([float(next(rr for rr in sub if rr["group"] == g)["rate"]) for g in group_order], dtype=float)
        lo = np.array([float(next(rr for rr in sub if rr["group"] == g)["ci_lo"]) for g in group_order], dtype=float)
        hi = np.array([float(next(rr for rr in sub if rr["group"] == g)["ci_hi"]) for g in group_order], dtype=float)
        # Light connecting line (ordered groups) improves readability without implying continuity.
        ax.plot(x + off, y, color=color, lw=1.0, alpha=0.30, zorder=2)
        # Manual error bars: matplotlib's `errorbar` may omit the top cap when (hi == y),
        # which happens for rate==1.0 (e.g., Track A in Intermediate). We draw caps explicitly.
        # Longer caps so they remain visible even when the point sits on the cap (e.g., rate==1.0).
        cap_dx = 0.12  # in x-axis units (category spacing is 1.0)
        ax.vlines(x + off, lo, hi, color=color, lw=1.2, zorder=3)
        ax.hlines(lo, x + off - cap_dx, x + off + cap_dx, color=color, lw=1.2, zorder=3)
        ax.hlines(hi, x + off - cap_dx, x + off + cap_dx, color=color, lw=1.2, zorder=3)
        ax.scatter(
            x + off,
            y,
            s=46,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label=name,
        )

    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{group_label.get(g,g)}\n(n={n_by_group.get(g,'?')})" for g in group_order])
    # Keep the logical scale (0..1) but add headroom so points/CI at 1.0 and
    # significance brackets are not clipped (e.g., Track A = 1.0 in Intermediate).
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.2)

    # Significance (Track A vs T1) per subgroup:
    # draw a horizontal bracket between T1 and Track A markers and put stars above the bracket.
    if p_by_group:
        for i, g in enumerate(group_order):
            p = float(p_by_group.get(g, float("nan")))
            stars = _p_to_stars(p)
            if not stars:
                continue

            def _get_ci_hi(method_key: str) -> float:
                r = next(rr for rr in rows if rr["group"] == g and rr["method_key"] == method_key)
                return float(r["ci_hi"])

            # Bracket between T1 and Track A points
            x1 = x[i] + offset_by_key.get("t1", -0.24)
            x2 = x[i] + offset_by_key.get("trackA", 0.24)
            y = max(_get_ci_hi("t1"), _get_ci_hi("trackA")) + 0.03
            h = 0.02
            y_max = float(ax.get_ylim()[1])
            y = min(y, y_max - h - 0.03)
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.0, zorder=4)
            ax.text((x1 + x2) / 2.0, y + h + 0.01, stars, ha="center", va="bottom", fontsize=12, zorder=5)

    if p_footnote:
        ax.text(0.995, 0.02, str(p_footnote), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def _add_t1_nondifficult_inset(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    colors: Dict[str, str],
) -> None:
    """
    Add an inset bar chart within Panel A comparing T1-only detection in:
      - non-difficult (Easy+Intermediate)
      - difficult
    """

    if "MRIDetectability" not in merged.columns:
        return

    detectability = merged["MRIDetectability"].astype(str)
    non_mask = detectability.isin(["Easy", "Intermediate"])
    diff_mask = detectability.eq("Difficult")

    n_non = int(np.sum(non_mask))
    n_diff = int(np.sum(diff_mask))
    if n_non == 0 or n_diff == 0:
        return

    t1_detected = merged["detected_boxdsc_t1"].astype(bool).to_numpy()
    k_non = int(np.sum(non_mask.to_numpy() & t1_detected))
    k_diff = int(np.sum(diff_mask.to_numpy() & t1_detected))

    rate_non = float(k_non / n_non)
    rate_diff = float(k_diff / n_diff)
    ci_non = _wilson_ci(k_non, n_non)
    ci_diff = _wilson_ci(k_diff, n_diff)

    try:
        p_fisher = _fisher_exact_two_sided(k_non, n_non - k_non, k_diff, n_diff - k_diff)
    except Exception:
        p_fisher = float("nan")

    # Give Panel A a bit of left x-margin so the inset doesn't cover the main curves/points.
    x0, x1 = ax.get_xlim()
    if x0 > -1.0:
        ax.set_xlim(-1.0, x1)

    # Inset placed in the left margin created above (axes fraction coordinates).
    # Layout: stack T1-only (top) + rs-fMRI-only (bottom) without occluding the main panel.
    ax_in = ax.inset_axes([0.03, 0.55, 0.23, 0.36])
    ax_in.set_facecolor("white")
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    bar_x = np.array([0.0, 1.0], dtype=float)
    bar_y = np.array([rate_non, rate_diff], dtype=float)
    bar_alphas = [0.90, 0.45]

    bars = ax_in.bar(
        bar_x,
        bar_y,
        width=0.68,
        color=[colors["t1"], colors["t1"]],
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for b, a in zip(bars, bar_alphas):
        b.set_alpha(a)

    def _yerr(rate: float, ci: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = ci
        return (max(0.0, rate - float(lo)), max(0.0, float(hi) - rate))

    yerr = np.array([_yerr(rate_non, ci_non), _yerr(rate_diff, ci_diff)], dtype=float).T
    ax_in.errorbar(bar_x, bar_y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3.0, zorder=3)

    ax_in.set_ylim(0.0, 1.05)
    ax_in.set_yticks([0.0, 0.5, 1.0])
    ax_in.set_xticks(bar_x)
    ax_in.set_xticklabels([f"Non-diff\n(n={n_non})", f"Difficult\n(n={n_diff})"], fontsize=9)
    ax_in.tick_params(axis="y", labelsize=9)
    ax_in.set_title("T1-only", fontsize=10, pad=2)
    ax_in.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)

    stars = _p_to_stars(float(p_fisher))
    if stars:
        ax_in.text(0.5, 1.02, stars, transform=ax_in.transAxes, ha="center", va="bottom", fontsize=12)
    if not math.isnan(float(p_fisher)):
        ax_in.text(0.5, 0.94, f"p={p_fisher:.3g}", transform=ax_in.transAxes, ha="center", va="top", fontsize=9)


def _plot_nondifficult_vs_difficult_bar(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    detected_col: str,
    color: str,
    title: str,
) -> None:
    """
    Plot non-difficult (Easy+Intermediate) vs difficult bar chart for a given detected column.
    """

    if "MRIDetectability" not in merged.columns:
        raise KeyError("Missing clinical column: MRIDetectability")
    if detected_col not in merged.columns:
        raise KeyError(f"Missing column: {detected_col}")

    detectability = merged["MRIDetectability"].astype(str)
    non_mask = detectability.isin(["Easy", "Intermediate"]).to_numpy()
    diff_mask = detectability.eq("Difficult").to_numpy()

    n_non = int(np.sum(non_mask))
    n_diff = int(np.sum(diff_mask))
    if n_non == 0 or n_diff == 0:
        return

    detected = merged[detected_col].astype(bool).to_numpy()
    k_non = int(np.sum(non_mask & detected))
    k_diff = int(np.sum(diff_mask & detected))

    rate_non = float(k_non / n_non)
    rate_diff = float(k_diff / n_diff)
    ci_non = _wilson_ci(k_non, n_non)
    ci_diff = _wilson_ci(k_diff, n_diff)

    try:
        p_fisher = _fisher_exact_two_sided(k_non, n_non - k_non, k_diff, n_diff - k_diff)
    except Exception:
        p_fisher = float("nan")

    bar_x = np.array([0.0, 1.0], dtype=float)
    bar_y = np.array([rate_non, rate_diff], dtype=float)
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

    yerr = np.array([_yerr(rate_non, ci_non), _yerr(rate_diff, ci_diff)], dtype=float).T
    ax.errorbar(bar_x, bar_y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3.0, zorder=3)

    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"Non-diff\n(n={n_non})", f"Difficult\n(n={n_diff})"], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)

    stars = _p_to_stars(float(p_fisher))
    if stars:
        ax.text(0.5, 1.02, stars, transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    if not math.isnan(float(p_fisher)):
        ax.text(0.5, 0.94, f"p={p_fisher:.3g}", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _add_fmri_nondifficult_inset(
    ax: plt.Axes,
    *,
    merged: pd.DataFrame,
    colors: Dict[str, str],
) -> None:
    """
    Add an inset bar chart within Panel A comparing rs-fMRI-only detection in:
      - non-difficult (Easy+Intermediate)
      - difficult
    """

    if "MRIDetectability" not in merged.columns:
        return
    if "detected_boxdsc_fmri" not in merged.columns:
        return

    detectability = merged["MRIDetectability"].astype(str)
    non_mask = detectability.isin(["Easy", "Intermediate"])
    diff_mask = detectability.eq("Difficult")

    n_non = int(np.sum(non_mask))
    n_diff = int(np.sum(diff_mask))
    if n_non == 0 or n_diff == 0:
        return

    fmri_detected = merged["detected_boxdsc_fmri"].astype(bool).to_numpy()
    k_non = int(np.sum(non_mask.to_numpy() & fmri_detected))
    k_diff = int(np.sum(diff_mask.to_numpy() & fmri_detected))

    rate_non = float(k_non / n_non)
    rate_diff = float(k_diff / n_diff)
    ci_non = _wilson_ci(k_non, n_non)
    ci_diff = _wilson_ci(k_diff, n_diff)

    try:
        p_fisher = _fisher_exact_two_sided(k_non, n_non - k_non, k_diff, n_diff - k_diff)
    except Exception:
        p_fisher = float("nan")

    # Ensure the extra left margin exists (shared with the T1 inset).
    x0, x1 = ax.get_xlim()
    if x0 > -1.0:
        ax.set_xlim(-1.0, x1)

    ax_in = ax.inset_axes([0.03, 0.14, 0.23, 0.36])
    ax_in.set_facecolor("white")
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    bar_x = np.array([0.0, 1.0], dtype=float)
    bar_y = np.array([rate_non, rate_diff], dtype=float)
    bar_alphas = [0.90, 0.45]

    bars = ax_in.bar(
        bar_x,
        bar_y,
        width=0.68,
        color=[colors["fmri"], colors["fmri"]],
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for b, a in zip(bars, bar_alphas):
        b.set_alpha(a)

    def _yerr(rate: float, ci: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = ci
        return (max(0.0, rate - float(lo)), max(0.0, float(hi) - rate))

    yerr = np.array([_yerr(rate_non, ci_non), _yerr(rate_diff, ci_diff)], dtype=float).T
    ax_in.errorbar(bar_x, bar_y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3.0, zorder=3)

    ax_in.set_ylim(0.0, 1.05)
    ax_in.set_yticks([0.0, 0.5, 1.0])
    ax_in.set_xticks(bar_x)
    ax_in.set_xticklabels([f"Non-diff\n(n={n_non})", f"Difficult\n(n={n_diff})"], fontsize=9)
    ax_in.tick_params(axis="y", labelsize=9)
    ax_in.set_title("rs-fMRI-only", fontsize=10, pad=2)
    ax_in.grid(axis="y", linestyle=":", alpha=0.25, zorder=0)

    stars = _p_to_stars(float(p_fisher))
    if stars:
        ax_in.text(0.5, 1.02, stars, transform=ax_in.transAxes, ha="center", va="bottom", fontsize=12)
    if not math.isnan(float(p_fisher)):
        ax_in.text(0.5, 0.94, f"p={p_fisher:.3g}", transform=ax_in.transAxes, ha="center", va="top", fontsize=9)


def _save_t1_nondifficult_bar(
    *,
    out_pdf: Path,
    out_png: Path,
    merged: pd.DataFrame,
    colors: Dict[str, str],
) -> None:
    """
    Standalone version of the Panel A inset (for manual layout in Illustrator/PowerPoint).
    """

    panel_b_w_in, panel_b_h_in = _panel_b_axes_size_in()
    target_ax_w_in = panel_b_w_in * (2.0 / 3.0)
    target_ax_h_in = panel_b_h_in

    # Add minimal padding so tick labels/p-value aren't clipped, while keeping the axes box exact.
    pad_left_in = 0.32
    pad_right_in = 0.06
    pad_bottom_in = 0.50
    pad_top_in = 0.18

    fig_w_in = target_ax_w_in + pad_left_in + pad_right_in
    fig_h_in = target_ax_h_in + pad_bottom_in + pad_top_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes(
        [
            pad_left_in / fig_w_in,
            pad_bottom_in / fig_h_in,
            target_ax_w_in / fig_w_in,
            target_ax_h_in / fig_h_in,
        ]
    )
    _plot_nondifficult_vs_difficult_bar(
        ax,
        merged=merged,
        detected_col="detected_boxdsc_t1",
        color=colors["t1"],
        title="T1-only",
    )

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _save_fmri_nondifficult_bar(
    *,
    out_pdf: Path,
    out_png: Path,
    merged: pd.DataFrame,
    colors: Dict[str, str],
) -> None:
    """
    Standalone bar chart (same size as T1 version): fMRI-only non-difficult vs difficult.
    """

    panel_b_w_in, panel_b_h_in = _panel_b_axes_size_in()
    target_ax_w_in = panel_b_w_in * (2.0 / 3.0)
    target_ax_h_in = panel_b_h_in

    pad_left_in = 0.32
    pad_right_in = 0.06
    pad_bottom_in = 0.50
    pad_top_in = 0.18

    fig_w_in = target_ax_w_in + pad_left_in + pad_right_in
    fig_h_in = target_ax_h_in + pad_bottom_in + pad_top_in
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes(
        [
            pad_left_in / fig_w_in,
            pad_bottom_in / fig_h_in,
            target_ax_w_in / fig_w_in,
            target_ax_h_in / fig_h_in,
        ]
    )

    _plot_nondifficult_vs_difficult_bar(
        ax,
        merged=merged,
        detected_col="detected_boxdsc_fmri",
        color=colors["fmri"],
        title="rs-fMRI-only",
    )

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_complementarity_stacked(
    ax: plt.Axes,
    *,
    rows: List[Dict[str, object]],
    group_order: List[str],
    group_label: Dict[str, str],
    title: str,
    p_by_group: Optional[Dict[str, float]] = None,
    p_footnote: Optional[str] = "* p<0.05, ** p<0.01 (unadjusted)",
) -> None:
    x = np.arange(len(group_order), dtype=float)
    width = 0.68

    # Category order (stack bottom->top)
    cat_order = ["neither", "t1_only", "trackA_only", "both"]
    cat_label = {
        "both": "Both detected",
        "t1_only": "T1 only (hurt)",
        "trackA_only": "Track A only (rescued)",
        "neither": "Neither detected",
    }
    COLORS = _load_figure_colors()
    cat_color = {
        "both": COLORS["overlap"],
        "t1_only": COLORS["t1"],
        "trackA_only": COLORS["trackA"],
        "neither": "#d9d9d9",
    }

    bottoms = np.zeros(len(group_order), dtype=float)
    for cat in cat_order:
        vals = np.array([float(next(r for r in rows if r["group"] == g)[f"prop_{cat}"]) for g in group_order])
        ax.bar(x, vals, bottom=bottoms, width=width, color=cat_color[cat], edgecolor="white", linewidth=0.6, label=cat_label[cat])
        bottoms += vals

    for i, g in enumerate(group_order):
        r = next(rr for rr in rows if rr["group"] == g)
        ax.text(x[i], 1.02, f"n={int(r['n'])}", ha="center", va="bottom", fontsize=9)
        if p_by_group:
            stars = _p_to_stars(float(p_by_group.get(g, float("nan"))))
            if stars:
                y0 = 1.045
                ax.plot([x[i] - width / 2.0, x[i] + width / 2.0], [y0, y0], color="black", lw=1.0, zorder=5)
                ax.text(x[i], y0 + 0.01, stars, ha="center", va="bottom", fontsize=12, zorder=6)

    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([group_label.get(g, g) for g in group_order])
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Proportion")
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    if p_footnote:
        ax.text(0.995, 0.02, str(p_footnote), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def _plot_location_delta_forest(
    ax: plt.Axes,
    *,
    rows: List[Dict[str, object]],
    title: str,
) -> None:
    y = np.arange(len(rows), dtype=float)
    delta = np.array([float(r["delta"]) for r in rows], dtype=float)
    lo = np.array([float(r["ci_lo"]) for r in rows], dtype=float)
    hi = np.array([float(r["ci_hi"]) for r in rows], dtype=float)

    COLORS = _load_figure_colors()
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.5, zorder=1)
    ax.hlines(y, lo, hi, color="black", lw=1.2, zorder=2)
    ax.scatter(delta, y, s=28, color=COLORS["trackA"], edgecolor="black", linewidth=0.35, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([str(r["location"]) for r in rows])
    ax.invert_yaxis()

    # Extend x-range to the right for the numeric text column (forest-plot style).
    lo_all = float(np.nanmin(lo)) if len(rows) else -0.5
    hi_all = float(np.nanmax(hi)) if len(rows) else 0.5
    lo_all = min(lo_all, 0.0)
    hi_all = max(hi_all, 0.0)
    span = max(1e-6, hi_all - lo_all)
    pad = 0.05 * span
    ax.set_xlim(lo_all - pad, hi_all + 0.75 * span)

    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Δ detection rate (Track A − T1)")
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)

    # Right-side n labels (not clipped)
    for yi, r in zip(y, rows):
        # Numeric text column (effect size + CI), aligned across rows.
        txt = (
            f"{float(r['delta']) * 100.0:+0.1f}% "
            f"[{float(r['ci_lo']) * 100.0:0.1f}, {float(r['ci_hi']) * 100.0:0.1f}]"
        )
        ax.text(hi_all + 0.05 * span, yi, txt, va="center", ha="left", fontsize=9)

        ax.text(
            1.01,
            yi,
            f"n={int(r['n'])}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9,
            clip_on=False,
        )
        stars = _p_to_stars(float(r.get("p_mcnemar", float("nan"))))
        if stars:
            # Place stars on the CI line (near the right end), not in the label column.
            x_star = float(r.get("ci_hi", 0.0))
            ax.text(
                x_star,
                yi,
                stars,
                ha="left",
                va="center",
                fontsize=12,
                clip_on=False,
            )
    ax.text(0.995, 1.02, "Exploratory; small n in some lobes", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.text(0.995, 0.02, "* p<0.05, ** p<0.01 (unadjusted)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


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

    audit_path = PROJECT_DIR / "paper/table2/table2_audit.json"
    clinical_path = PROJECT_DIR / "paper/supplement_table_hs_patient_characteristics.csv"

    out_dir = PROJECT_DIR / "paper/figure3"
    out_pdf = out_dir / "figure3.pdf"
    out_png = out_dir / "figure3.png"
    out_inset_pdf = out_dir / "panelA_t1only_nondifficult_vs_difficult.pdf"
    out_inset_png = out_dir / "panelA_t1only_nondifficult_vs_difficult.png"
    out_inset_fmri_pdf = out_dir / "panelA_fmrionly_nondifficult_vs_difficult.pdf"
    out_inset_fmri_png = out_dir / "panelA_fmrionly_nondifficult_vs_difficult.png"

    audit = json.loads(audit_path.read_text())
    n_boot = int(audit.get("params", {}).get("n_boot", 10000))
    seed = int(audit.get("params", {}).get("seed", 42))

    COLORS = _load_figure_colors()
    methods = [
        ("T1-only", "t1", COLORS["t1"]),
        ("rs-fMRI-only", "fmri", COLORS["fmri"]),
        ("Track A", "trackA", COLORS["trackA"]),
    ]

    merged = _merge_eval_with_clinical(audit=audit, clinical_path=clinical_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "figure3_merged_eval_clinical.csv", index=False)
    _save_t1_nondifficult_bar(out_pdf=out_inset_pdf, out_png=out_inset_png, merged=merged, colors=COLORS)
    _save_fmri_nondifficult_bar(out_pdf=out_inset_fmri_pdf, out_png=out_inset_fmri_png, merged=merged, colors=COLORS)

    # ---------------- Panel A: MRI detectability stratified detection ----------------
    if "MRIDetectability" not in merged.columns:
        raise KeyError("Missing clinical column: MRIDetectability")
    det_order = ["Easy", "Intermediate", "Difficult"]
    det_label = {"Easy": "Easy", "Intermediate": "Intermediate", "Difficult": "Difficult"}
    det_rows: List[Dict[str, object]] = []
    det_p_by_group: Dict[str, float] = {}
    for g in det_order:
        sub = merged[merged["MRIDetectability"].astype(str) == g].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1 = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        tr = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        n01 = int(np.sum((~t1) & tr))
        n10 = int(np.sum(t1 & (~tr)))
        det_p_by_group[g] = _mcnemar_exact_pvalue(n01, n10)

        for _, key, _ in methods:
            k = int(np.sum(sub[f"detected_boxdsc_{key}"].astype(bool)))
            rate = float(k / n)
            ci_lo, ci_hi = _wilson_ci(k, n)
            det_rows.append(
                {
                    "group": g,
                    "n": n,
                    "method_key": key,
                    "k_detected": k,
                    "rate": rate,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n01_trackA_vs_t1": n01,
                    "n10_trackA_vs_t1": n10,
                    "p_mcnemar_trackA_vs_t1": det_p_by_group[g],
                }
            )
    if det_rows:
        _write_tsv(out_dir / "figure3_detectability_detection.tsv", det_rows, list(det_rows[0].keys()))

    # Extra evidence: T1-only performance drop in "Difficult" detectability cases.
    # Compare non-Difficult (Easy+Intermediate) vs Difficult using an unpaired Fisher exact test.
    t1_drop_summary: Optional[Dict[str, object]] = None
    try:
        t1_easy = next(r for r in det_rows if r["group"] == "Easy" and r["method_key"] == "t1")
        t1_int = next(r for r in det_rows if r["group"] == "Intermediate" and r["method_key"] == "t1")
        t1_diff = next(r for r in det_rows if r["group"] == "Difficult" and r["method_key"] == "t1")

        k_non = int(t1_easy["k_detected"]) + int(t1_int["k_detected"])
        n_non = int(t1_easy["n"]) + int(t1_int["n"])
        k_diff = int(t1_diff["k_detected"])
        n_diff = int(t1_diff["n"])

        p_fisher = _fisher_exact_two_sided(k_non, n_non - k_non, k_diff, n_diff - k_diff)
        t1_drop_summary = {
            "contrast": "T1-only: non-Difficult vs Difficult (Det boxDSC>0.22)",
            "non_difficult_k": k_non,
            "non_difficult_n": n_non,
            "non_difficult_rate": float(k_non / n_non) if n_non > 0 else float("nan"),
            "difficult_k": k_diff,
            "difficult_n": n_diff,
            "difficult_rate": float(k_diff / n_diff) if n_diff > 0 else float("nan"),
            "p_fisher_two_sided": float(p_fisher),
            "stars": _p_to_stars(float(p_fisher)),
        }
        _write_tsv(out_dir / "figure3_detectability_t1_drop.tsv", [t1_drop_summary], list(t1_drop_summary.keys()))
    except Exception:
        t1_drop_summary = None

    # Group summaries (for sanity checks / SI tables; not used for plotting).
    det_summary_rows: List[Dict[str, object]] = []
    for g in det_order:
        # Use the plotting rows (Wilson CI) to avoid drift.
        try:
            r_t1 = next(r for r in det_rows if r["group"] == g and r["method_key"] == "t1")
            r_fmri = next(r for r in det_rows if r["group"] == g and r["method_key"] == "fmri")
            r_tr = next(r for r in det_rows if r["group"] == g and r["method_key"] == "trackA")
        except StopIteration:
            continue

        # Pinpointing summaries come from the merged table (subject-level).
        sub = merged[merged["MRIDetectability"].astype(str) == g].copy()
        n = int(len(sub))
        if n <= 0:
            continue
        pin_rows = {}
        pin_p = float("nan")
        try:
            t1p = sub["pinpointed_t1"].astype(bool).to_numpy()
            trp = sub["pinpointed_trackA"].astype(bool).to_numpy()
            n01p = int(np.sum((~t1p) & trp))
            n10p = int(np.sum(t1p & (~trp)))
            pin_p = _mcnemar_exact_pvalue(n01p, n10p)
        except Exception:
            pin_p = float("nan")

        for key in ("t1", "fmri", "trackA"):
            kpin = int(np.sum(sub[f"pinpointed_{key}"].astype(bool)))
            pin_rows[key] = {
                "rate": float(kpin / n),
                "ci": _fmt_ci(*_wilson_ci(kpin, n)),
            }

        det_summary_rows.append(
            {
                "Group": g,
                "n": n,
                "T1_det": float(r_t1["rate"]),
                "T1_det_CI95": _fmt_ci(float(r_t1["ci_lo"]), float(r_t1["ci_hi"])),
                "fMRI_det": float(r_fmri["rate"]),
                "fMRI_det_CI95": _fmt_ci(float(r_fmri["ci_lo"]), float(r_fmri["ci_hi"])),
                "TrackA_det": float(r_tr["rate"]),
                "TrackA_det_CI95": _fmt_ci(float(r_tr["ci_lo"]), float(r_tr["ci_hi"])),
                "p_det_trackA_vs_T1_mcnemar": float(det_p_by_group.get(g, float("nan"))),
                "T1_pin": float(pin_rows["t1"]["rate"]),
                "T1_pin_CI95": str(pin_rows["t1"]["ci"]),
                "fMRI_pin": float(pin_rows["fmri"]["rate"]),
                "fMRI_pin_CI95": str(pin_rows["fmri"]["ci"]),
                "TrackA_pin": float(pin_rows["trackA"]["rate"]),
                "TrackA_pin_CI95": str(pin_rows["trackA"]["ci"]),
                "p_pin_trackA_vs_T1_mcnemar": float(pin_p),
            }
        )

    if det_summary_rows:
        _write_tsv(out_dir / "figure3_detectability_summary.tsv", det_summary_rows, list(det_summary_rows[0].keys()))

    # Trend test for T1-only across ordered detectability groups (exploratory).
    try:
        k_t1 = [int(next(r for r in det_rows if r["group"] == g and r["method_key"] == "t1")["k_detected"]) for g in det_order]
        n_t1 = [int(next(r for r in det_rows if r["group"] == g and r["method_key"] == "t1")["n"]) for g in det_order]
        p_trend = _cochran_armitage_trend_p(k=k_t1, n=n_t1, scores=[0.0, 1.0, 2.0])
        trend_row = {
            "method": "T1-only",
            "endpoint": "Det(boxDSC>0.22)",
            "groups_ordered": "Easy,Intermediate,Difficult",
            "k": ",".join(str(x) for x in k_t1),
            "n": ",".join(str(x) for x in n_t1),
            "p_cochran_armitage_trend": float(p_trend),
        }
        _write_tsv(out_dir / "figure3_detectability_t1_trend.tsv", [trend_row], list(trend_row.keys()))
    except Exception:
        pass

    # ---------------- Panel B: complementarity by detectability (T1 vs Track A) ----------------
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
                "n01_trackA_vs_t1": int(tr_only),
                "n10_trackA_vs_t1": int(t1_only),
                "p_mcnemar_trackA_vs_t1": det_p_by_group.get(g, float("nan")),
            }
        )
    if comp_rows:
        _write_tsv(out_dir / "figure3_detectability_complementarity.tsv", comp_rows, list(comp_rows[0].keys()))

    # ---------------- Panel C: SEEG stratified detection ----------------
    if "SEEG" not in merged.columns:
        raise KeyError("Missing clinical column: SEEG")
    seeg_order = ["Yes", "No"]
    seeg_label = {"Yes": "SEEG yes", "No": "SEEG no"}
    seeg_rows: List[Dict[str, object]] = []
    seeg_p_by_group: Dict[str, float] = {}
    for g in seeg_order:
        sub = merged[merged["SEEG"].astype(str) == g].copy()
        n = int(len(sub))
        if n == 0:
            continue
        t1 = sub["detected_boxdsc_t1"].astype(bool).to_numpy()
        tr = sub["detected_boxdsc_trackA"].astype(bool).to_numpy()
        n01 = int(np.sum((~t1) & tr))
        n10 = int(np.sum(t1 & (~tr)))
        seeg_p_by_group[g] = _mcnemar_exact_pvalue(n01, n10)

        for _, key, _ in methods:
            k = int(np.sum(sub[f"detected_boxdsc_{key}"].astype(bool)))
            rate = float(k / n)
            ci_lo, ci_hi = _wilson_ci(k, n)
            seeg_rows.append(
                {
                    "group": g,
                    "n": n,
                    "method_key": key,
                    "k_detected": k,
                    "rate": rate,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n01_trackA_vs_t1": n01,
                    "n10_trackA_vs_t1": n10,
                    "p_mcnemar_trackA_vs_t1": seeg_p_by_group[g],
                }
            )
    if seeg_rows:
        _write_tsv(out_dir / "figure3_seeg_detection.tsv", seeg_rows, list(seeg_rows[0].keys()))

    # ---------------- Panel D: location forest plot (Δ detection TrackA − T1) ----------------
    if "DominantResectionCavityLocation_EZproxy" not in merged.columns:
        raise KeyError("Missing clinical column: DominantResectionCavityLocation_EZproxy")
    loc_map = {"Insular/Peri-insular (incl. operculum)": "Operculo-insular"}
    merged["LocationGroup"] = (
        merged["DominantResectionCavityLocation_EZproxy"].astype(str).map(loc_map).fillna(merged["DominantResectionCavityLocation_EZproxy"].astype(str))
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

        t1 = t1b.astype(float)
        tr = trb.astype(float)
        delta = float(np.mean(tr) - np.mean(t1))
        ci_lo, ci_hi = _paired_bootstrap_ci_diff(t1, tr, n_boot=n_boot, seed=seed + 1000 + i)
        loc_rows.append(
            {
                "location": loc,
                "n": n,
                "t1_rate": float(np.mean(t1)),
                "trackA_rate": float(np.mean(tr)),
                "delta": delta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n01_trackA_vs_t1": n01,
                "n10_trackA_vs_t1": n10,
                "p_mcnemar": p_loc,
            }
        )
    if loc_rows:
        _write_tsv(out_dir / "figure3_location_delta_forest.tsv", loc_rows, list(loc_rows[0].keys()))

    # ---------------- Plot figure ----------------
    fig = plt.figure(figsize=(DEFAULT_FIGURE_WIDTH_MM / MM_PER_INCH, DEFAULT_FIGURE_HEIGHT_MM / MM_PER_INCH))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.05, 0.95, 1.15],
        width_ratios=[2.2, 1.0],  # Panel A (wide) + Panel B (two small bars)
        left=0.16,
        right=0.94,  # keep some space for the forest-plot numeric column
        top=0.97,
        bottom=0.06,
        hspace=0.65,
        wspace=0.55,
    )

    # Row 1: A (main) + B (two bar charts)
    axA = fig.add_subplot(gs[0, 0])
    gsB = gs[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.55)
    axB1 = fig.add_subplot(gsB[0, 0])
    axB2 = fig.add_subplot(gsB[1, 0], sharey=axB1)

    # Row 2: C + D
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # Row 3: E
    axE = fig.add_subplot(gs[2, :])

    _plot_detection_dotwhisker(
        axA,
        rows=det_rows,
        group_order=[g for g in det_order if any(r["group"] == g for r in det_rows)],
        group_label=det_label,
        method_order=methods,
        title="MRI detectability: detection rate",
        ylabel="Detection rate (boxDSC>0.22)",
        show_legend=True,
        p_by_group=det_p_by_group,
    )

    _plot_nondifficult_vs_difficult_bar(
        axB1,
        merged=merged,
        detected_col="detected_boxdsc_t1",
        color=COLORS["t1"],
        title="T1-only",
    )
    _plot_nondifficult_vs_difficult_bar(
        axB2,
        merged=merged,
        detected_col="detected_boxdsc_fmri",
        color=COLORS["fmri"],
        title="rs-fMRI-only",
    )
    axB1.set_ylabel("Detection rate", fontsize=10)
    axB2.set_ylabel("")
    # Avoid overlapping x-labels between the two stacked subplots:
    # show x tick labels only on the bottom bar chart (categories are identical).
    axB1.tick_params(axis="x", bottom=False, labelbottom=False)
    axB2.tick_params(axis="x", labelsize=9)
    plt.setp(axB1.get_xticklabels(), rotation=0, ha="center")
    plt.setp(axB2.get_xticklabels(), rotation=0, ha="center")

    _plot_complementarity_stacked(
        axC,
        rows=comp_rows,
        group_order=[g for g in det_order if any(r["group"] == g for r in comp_rows)],
        group_label=det_label,
        title="Complementarity by detectability (T1 vs Track A)",
        p_by_group=det_p_by_group,
    )
    _plot_detection_dotwhisker(
        axD,
        rows=seeg_rows,
        group_order=[g for g in seeg_order if any(r["group"] == g for r in seeg_rows)],
        group_label=seeg_label,
        method_order=methods,
        title="SEEG: detection rate",
        ylabel="Detection rate (boxDSC>0.22)",
        show_legend=False,
        p_by_group=seeg_p_by_group,
    )
    _plot_location_delta_forest(
        axE,
        rows=loc_rows,
        title="EZ-proxy location: Δ detection rate (Track A − T1)",
    )

    # Panel labels (avoid title collisions)
    axA.text(-0.10, 1.02, "A", transform=axA.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    axB1.text(-0.35, 1.02, "B", transform=axB1.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    axC.text(-0.10, 1.02, "C", transform=axC.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    axD.text(-0.25, 1.02, "D", transform=axD.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")
    axE.text(-0.06, 1.02, "E", transform=axE.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    with (out_dir / "figure3_inputs.json").open("w") as f:
        json.dump(
            {
                "design_spec": str(PROJECT_DIR / "paper/figure3/figure3design.txt"),
                "table2_audit_json": str(audit_path),
                "clinical_csv": str(clinical_path),
                "eval_dirs": {
                    "t1": str(Path(audit["inputs"]["t1_eval_dir"])),
                    "fmri": str(Path(audit["inputs"]["fmri_eval_dir"])),
                    "trackA": str(Path(audit["inputs"]["tracka_eval_dir"])),
                },
                "endpoint": {"subject_level_detection": "detected_boxdsc (boxDSC>0.22)"},
                "ci_methods": {"detection_rate": "Wilson", "delta_detection_rate": f"paired bootstrap (n_boot={n_boot}, seed={seed})"},
                "figure_colors_json": str(FIGURE_COLORS_PATH),
                "figure_colors": COLORS,
                "outputs": {"pdf": str(out_pdf), "png": str(out_png)},
                "generated_tables": {
                    "detectability_detection": str(out_dir / "figure3_detectability_detection.tsv"),
                    "detectability_t1_drop": str(out_dir / "figure3_detectability_t1_drop.tsv"),
                    "detectability_summary": str(out_dir / "figure3_detectability_summary.tsv"),
                    "detectability_t1_trend": str(out_dir / "figure3_detectability_t1_trend.tsv"),
                    "detectability_complementarity": str(out_dir / "figure3_detectability_complementarity.tsv"),
                    "seeg_detection": str(out_dir / "figure3_seeg_detection.tsv"),
                    "location_delta_forest": str(out_dir / "figure3_location_delta_forest.tsv"),
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
