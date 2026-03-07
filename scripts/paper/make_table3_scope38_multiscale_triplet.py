#!/usr/bin/env python3
"""
Make a Table-2-style multi-scale performance table on the 38-subject subset:
  - Start from paper/table2/table2_source_data.csv (52 seizure-free cohort)
  - Keep only MRIDetectability in {intermediate, difficult}
  - Keep only ILAE in {1,2} (seizure-free reference standard = resection cavity)

This script is used to generate:
  - Main-text table: self-trained MELD baseline + rs-fMRI + TrackA (all under consistent deploy constraints)
  - Supplementary table: native MELD baseline + rs-fMRI + TrackA (native pipeline), as a sensitivity analysis.

It mirrors the row structure and formatting of:
  paper/revision/table3/table3_source_original.md

Δ and p-values are computed for TrackA vs the T1 baseline method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paper.table2.generate_table2 import (  # type: ignore
    _bootstrap,
    _compute_metrics,
    _format_mean_ci,
    _format_n_pct_ci,
    _format_pct_ci,
    _load_cluster_rows,
    _load_subject_rows,
    _mcnemar_exact,
    _signflip_pvalue,
    _two_sided_boot_pvalue,
)

P_ADJUST_METHODS = ["none", "holm", "fdr_bh", "holm3_fdr_bh"]

def _relpath(p: Path) -> str:
    """
    Make paths reproducible by writing them relative to PROJECT_DIR when possible.
    """
    try:
        return str(p.resolve().relative_to(PROJECT_DIR))
    except Exception:
        return str(p)


def _holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """
    Holm–Bonferroni adjusted p-values (strong FWER control).
    """
    p = [float(x) for x in pvals]
    n = len(p)
    if n <= 1:
        return p
    order = sorted(range(n), key=lambda i: p[i])
    ranked = [p[i] for i in order]
    adj_ranked = []
    prev = 0.0
    for j, pv in enumerate(ranked):
        a = (n - j) * float(pv)
        if a < prev:
            a = prev
        prev = a
        adj_ranked.append(min(1.0, max(0.0, a)))
    out = [0.0] * n
    for i, a in zip(order, adj_ranked):
        out[i] = float(a)
    return out


def _fdr_bh(pvals: Sequence[float]) -> List[float]:
    """
    Benjamini–Hochberg FDR adjusted p-values.
    """
    p = [float(x) for x in pvals]
    n = len(p)
    if n <= 1:
        return p
    order = sorted(range(n), key=lambda i: p[i])
    ranked = [p[i] for i in order]
    q = [min(1.0, max(0.0, ranked[i] * n / (i + 1))) for i in range(n)]
    # enforce monotone non-decreasing q-values
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = [0.0] * n
    for i, qi in zip(order, q):
        out[i] = float(qi)
    return out


def _holm3_fdr_bh(*, pvals: Sequence[float], primary_idx: Sequence[int]) -> List[float]:
    """
    Hierarchical correction for the table:
      - Primary family: Holm–Bonferroni on `primary_idx`.
      - Secondary family: Benjamini–Hochberg FDR on the remaining tests.

    This matches the manuscript convention:
      - Primary: Det(boxDSC>0.22 & Dice>0.01), Det(PPV-in-mask>=0.5), Pinpointing.
      - Secondary: all other metrics reported in the table.
    """
    p = [float(x) for x in pvals]
    n = len(p)
    primary_set = set(int(i) for i in primary_idx)
    if any(i < 0 or i >= n for i in primary_set):
        raise ValueError(f"primary_idx out of range: {sorted(primary_set)} (n={n})")

    secondary_idx = [i for i in range(n) if i not in primary_set]
    primary_p = [p[i] for i in range(n) if i in primary_set]
    secondary_p = [p[i] for i in secondary_idx]

    primary_adj = _holm_bonferroni(primary_p) if primary_p else []
    secondary_adj = _fdr_bh(secondary_p) if secondary_p else []

    out = list(p)
    # Assign back (preserve original ordering).
    # Primary: iterate indices in sorted order so mapping is stable.
    for j, i in enumerate(sorted(primary_set)):
        out[i] = float(primary_adj[j])
    for j, i in enumerate(secondary_idx):
        out[i] = float(secondary_adj[j])
    return out


def _fmt_p(p: float) -> str:
    if not math.isfinite(float(p)):
        return "NA"
    p = float(p)
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _parse_ilae(x: object) -> int | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        v = int(float(s))
        if 1 <= v <= 6:
            return v
    except Exception:
        return None
    return None


def _load_scope38(*, meta_csv: Path, paper_table2_csv: Path) -> List[str]:
    """
    Return the 38 seizure-free (ILAE 1-2) intermediate/difficult cases within the original 52 cohort.
    """
    paper_subjects: List[str] = []
    with paper_table2_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row["subject_id"]).strip()
            if sid:
                paper_subjects.append(sid)
    paper_set = set(paper_subjects)

    subs: List[str] = []
    with meta_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row.get("subject_id", "")).strip()
            if not sid or sid not in paper_set:
                continue
            det = str(row.get("MRIDetectability", "")).strip().lower()
            if det not in {"intermediate", "difficult"}:
                continue
            ilae = _parse_ilae(row.get("Prognosis(ILAE)", ""))
            if ilae not in {1, 2}:
                continue
            subs.append(sid)

    out: List[str] = []
    seen = set()
    for s in subs:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return sorted(out)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return float(sorted_values[lo] * (1 - w) + sorted_values[hi] * w)


def _ci95(values: Sequence[float]) -> Tuple[float, float]:
    sv = sorted(float(v) for v in values)
    return (_quantile(sv, 0.025), _quantile(sv, 0.975))


def _series(samples: Sequence[Dict[str, float]], key: str) -> List[float]:
    return [float(s[key]) for s in samples]


def _paired_boot_diffs(
    *,
    subject_ids: Sequence[str],
    track_sub: Dict[str, object],
    track_cl: Dict[str, object],
    base_sub: Dict[str, object],
    base_cl: Dict[str, object],
    n_boot: int,
    seed: int,
) -> Dict[str, List[float]]:
    import random

    rng = random.Random(seed)
    keys = [
        "vertex_dsc_mean",
        "clusters_per_subject_mean",
        "fp_per_patient_mean",
        "precision_boxdsc",
        "f1_boxdsc",
        "detection_rate_boxdsc",
        "detection_rate_ppv50",
        "detection_rate_distance",
        "pinpointing_rate",
        "recall_at_0.15",
    ]
    diffs: Dict[str, List[float]] = {k: [] for k in keys}
    n = len(subject_ids)
    for _ in range(int(n_boot)):
        sample = [subject_ids[rng.randrange(n)] for _ in range(n)]
        a = _compute_metrics(sample, track_sub, track_cl)
        b = _compute_metrics(sample, base_sub, base_cl)
        for k in keys:
            diffs[k].append(float(a[k] - b[k]))
    return diffs


def _fmt_point_ci(*, point: float, boot: Sequence[Dict[str, float]], key: str, is_pct: bool, decimals: int = 3) -> str:
    lo, hi = _ci95(_series(boot, key))
    if is_pct:
        return _format_pct_ci(point, lo, hi, decimals=1)
    return _format_mean_ci(point, lo, hi, decimals=decimals)


def _fmt_n_pct_ci(*, n: int, N: int, boot: Sequence[Dict[str, float]], key: str) -> str:
    lo, hi = _ci95(_series(boot, key))
    return _format_n_pct_ci(n, N, lo, hi, decimals=1)


def _fmt_delta(*, point_delta: float, boot_diffs: Sequence[float], is_pct: bool, decimals: int = 3) -> str:
    lo, hi = _ci95(boot_diffs)
    if is_pct:
        return _format_pct_ci(point_delta, lo, hi, decimals=1)
    return _format_mean_ci(point_delta, lo, hi, decimals=decimals)


def _write_subject_ids(subject_ids: Sequence[str], out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject_id"])
        for sid in subject_ids:
            w.writerow([sid])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default="supplement_table_hs_patient_three_level_metrics.filled.csv", type=Path)
    ap.add_argument("--paper_table2_csv", default="paper/table2/table2_source_data.csv", type=Path)
    ap.add_argument("--t1_eval_dir", required=True, type=Path)
    ap.add_argument("--fmri_eval_dir", required=True, type=Path)
    ap.add_argument("--tracka_eval_dir", required=True, type=Path)
    ap.add_argument("--out_md", required=True, type=Path)
    ap.add_argument("--out_subjects_tsv", default=None, type=Path)
    ap.add_argument("--out_audit_json", default=None, type=Path)
    ap.add_argument("--n_boot", default=10000, type=int)
    ap.add_argument("--seed", default=20260209, type=int)
    ap.add_argument("--n_perm", default=20000, type=int)
    ap.add_argument("--title", default=None, type=str)
    ap.add_argument("--t1_name", default="T1-only", type=str)
    ap.add_argument("--fmri_name", default="fMRI-only (fMRIhemi B2a)", type=str)
    ap.add_argument("--tracka_name", default="Track A (T1+fMRI)", type=str)
    ap.add_argument("--delta_label", default=None, type=str)
    ap.add_argument("--p_label", default=None, type=str)
    ap.add_argument(
        "--boxdsc_min_cluster_dice",
        default=0.0,
        type=float,
        help="Optional extra requirement for the boxDSC endpoint: a cluster is TP only if boxDSC>0.22 and cluster Dice > threshold (e.g. 0.01).",
    )
    ap.add_argument(
        "--p_adjust",
        default="none",
        choices=P_ADJUST_METHODS,
        help=(
            "Multiple-comparison correction. "
            "'holm'/'fdr_bh' apply across *all* p-values in the table. "
            "'holm3_fdr_bh' applies Holm on 3 primary endpoints "
            "(Det(boxDSC), Det(PPV>=0.5), Pinpointing) and BH-FDR on the remaining metrics."
        ),
    )
    ap.add_argument(
        "--constraints_note",
        default=None,
        type=str,
        help="Optional free-text note about post-processing constraints (e.g., a30_t80).",
    )
    args = ap.parse_args()

    subject_ids = _load_scope38(meta_csv=args.meta_csv.resolve(), paper_table2_csv=args.paper_table2_csv.resolve())
    if len(subject_ids) != 38:
        raise ValueError(f"Expected 38 subjects in scope, got {len(subject_ids)}")

    t1_dir = args.t1_eval_dir.resolve()
    fmri_dir = args.fmri_eval_dir.resolve()
    track_dir = args.tracka_eval_dir.resolve()

    # Load rows
    t1_sub = _load_subject_rows(t1_dir)
    t1_cl = _load_cluster_rows(t1_dir, boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice))
    fm_sub = _load_subject_rows(fmri_dir)
    fm_cl = _load_cluster_rows(fmri_dir, boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice))
    tr_sub = _load_subject_rows(track_dir)
    tr_cl = _load_cluster_rows(track_dir, boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice))

    # Update subject-level detected_boxdsc to match the (optional) cluster-level dice safeguard.
    if float(args.boxdsc_min_cluster_dice) > 0.0:
        def _refresh(sub_rows, cl_rows):
            out = {}
            for sid, r in sub_rows.items():
                det = any(c.is_tp_boxdsc for c in cl_rows.get(sid, []))
                out[sid] = replace(r, detected_boxdsc=bool(det))
            return out

        t1_sub = _refresh(t1_sub, t1_cl)
        fm_sub = _refresh(fm_sub, fm_cl)
        tr_sub = _refresh(tr_sub, tr_cl)

    # Guard: all scope subjects exist
    for name, d in [("t1", t1_sub), ("fmri", fm_sub), ("trackA", tr_sub)]:
        missing = [sid for sid in subject_ids if sid not in d]
        if missing:
            raise ValueError(f"Missing {len(missing)} scope subjects in {name} eval_dir. Example={missing[:3]}")

    # Point estimates
    t1 = _compute_metrics(subject_ids, t1_sub, t1_cl)
    fm = _compute_metrics(subject_ids, fm_sub, fm_cl)
    tr = _compute_metrics(subject_ids, tr_sub, tr_cl)

    # Bootstrap CIs
    t1_boot = _bootstrap(subject_ids, subject_rows=t1_sub, cluster_rows=t1_cl, n_boot=args.n_boot, seed=args.seed)
    fm_boot = _bootstrap(subject_ids, subject_rows=fm_sub, cluster_rows=fm_cl, n_boot=args.n_boot, seed=args.seed + 1)
    tr_boot = _bootstrap(subject_ids, subject_rows=tr_sub, cluster_rows=tr_cl, n_boot=args.n_boot, seed=args.seed + 2)

    # Paired bootstrap diffs (TrackA - T1)
    diffs = _paired_boot_diffs(
        subject_ids=subject_ids,
        track_sub=tr_sub,
        track_cl=tr_cl,
        base_sub=t1_sub,
        base_cl=t1_cl,
        n_boot=args.n_boot,
        seed=args.seed + 3,
    )

    # Paired tests (TrackA vs T1)
    d_dsc = [tr_sub[sid].vertex_dsc_lesion_hemi - t1_sub[sid].vertex_dsc_lesion_hemi for sid in subject_ids]
    d_nclu = [tr_sub[sid].n_clusters - t1_sub[sid].n_clusters for sid in subject_ids]
    d_fp = [tr_sub[sid].n_fp_clusters - t1_sub[sid].n_fp_clusters for sid in subject_ids]
    p_dsc = _signflip_pvalue(d_dsc, n_perm=args.n_perm, seed=args.seed + 10)
    p_nclu = _signflip_pvalue(d_nclu, n_perm=args.n_perm, seed=args.seed + 11)
    p_fp = _signflip_pvalue(d_fp, n_perm=args.n_perm, seed=args.seed + 12)

    t1_det = [t1_sub[sid].detected_boxdsc for sid in subject_ids]
    tr_det = [tr_sub[sid].detected_boxdsc for sid in subject_ids]
    p_det, det_n01, det_n10 = _mcnemar_exact(t1_det, tr_det)

    t1_ppv = [t1_sub[sid].detected_ppv50 for sid in subject_ids]
    tr_ppv = [tr_sub[sid].detected_ppv50 for sid in subject_ids]
    p_ppv, ppv_n01, ppv_n10 = _mcnemar_exact(t1_ppv, tr_ppv)

    t1_dist = [t1_sub[sid].detected_distance for sid in subject_ids]
    tr_dist = [tr_sub[sid].detected_distance for sid in subject_ids]
    p_dist, dist_n01, dist_n10 = _mcnemar_exact(t1_dist, tr_dist)

    t1_pin = [t1_sub[sid].pinpointed for sid in subject_ids]
    tr_pin = [tr_sub[sid].pinpointed for sid in subject_ids]
    p_pin, pin_n01, pin_n10 = _mcnemar_exact(t1_pin, tr_pin)

    t1_rec015 = [t1_sub[sid].lesion_dice_union >= 0.15 for sid in subject_ids]
    tr_rec015 = [tr_sub[sid].lesion_dice_union >= 0.15 for sid in subject_ids]
    p_rec015, rec015_n01, rec015_n10 = _mcnemar_exact(t1_rec015, tr_rec015)

    p_prec = _two_sided_boot_pvalue(diffs["precision_boxdsc"])
    p_f1 = _two_sided_boot_pvalue(diffs["f1_boxdsc"])
    p_pin_boot = _two_sided_boot_pvalue(diffs["pinpointing_rate"])
    p_det_ppv_boot = _two_sided_boot_pvalue(diffs["detection_rate_ppv50"])
    p_det_dist_boot = _two_sided_boot_pvalue(diffs["detection_rate_distance"])
    p_rec015_boot = _two_sided_boot_pvalue(diffs["recall_at_0.15"])

    # Optional: multiple-comparison correction across all tests in this table.
    p_raw_list = [
        float(p_dsc),
        float(p_nclu),
        float(p_fp),
        float(p_prec),
        float(p_f1),
        float(p_det),
        float(p_ppv),
        float(p_dist),
        float(p_pin),
        float(p_rec015),
    ]
    if args.p_adjust == "holm":
        p_adj_list = _holm_bonferroni(p_raw_list)
    elif args.p_adjust == "holm3_fdr_bh":
        # Primary family indices in p_raw_list:
        #   5: Det(boxDSC>0.22 & Dice>threshold)
        #   6: Det(PPV-in-mask>=0.5)
        #   8: Pinpointing
        p_adj_list = _holm3_fdr_bh(pvals=p_raw_list, primary_idx=[5, 6, 8])
    elif args.p_adjust == "fdr_bh":
        p_adj_list = _fdr_bh(p_raw_list)
    else:
        p_adj_list = list(p_raw_list)

    (
        p_dsc,
        p_nclu,
        p_fp,
        p_prec,
        p_f1,
        p_det,
        p_ppv,
        p_dist,
        p_pin,
        p_rec015,
    ) = [float(x) for x in p_adj_list]

    # Compose markdown (mirror table3_source_original.md)
    out_md = args.out_md.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    N = len(subject_ids)
    title = (
        args.title
        or "Table 2. Multi-scale performance of T1-only baseline, fMRI-only, and deployable multimodal fusion for EZ detection (n=38)"
    )
    delta_label = args.delta_label or f"Δ ({args.tracka_name} − {args.t1_name})"
    p_label = args.p_label or f"p ({args.tracka_name} vs {args.t1_name})"
    box_label = (
        f"boxDSC>0.22 & Dice>{float(args.boxdsc_min_cluster_dice):g}"
        if float(args.boxdsc_min_cluster_dice) > 0.0
        else "boxDSC>0.22"
    )
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if float(args.boxdsc_min_cluster_dice) > 0.0:
        lines.append(f"Primary endpoint: **Det(boxDSC > 0.22 & Dice > {float(args.boxdsc_min_cluster_dice):g})**.  ")
    else:
        lines.append("Primary endpoint: **Det(boxDSC > 0.22)**.  ")
    lines.append("Co-primary endpoint: **Det(PPV-in-mask ≥ 0.5)**.")
    lines.append("")
    lines.append(
        f"| Scale | Metric | {args.t1_name} | {args.fmri_name} | {args.tracka_name} | {delta_label} | {p_label} |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    # Vertex-level
    lines.append(
        "| Vertex-level | Dice similarity coefficient (DSC), mean [95% CI] "
        f"| {_fmt_point_ci(point=float(t1['vertex_dsc_mean']), boot=t1_boot, key='vertex_dsc_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(fm['vertex_dsc_mean']), boot=fm_boot, key='vertex_dsc_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(tr['vertex_dsc_mean']), boot=tr_boot, key='vertex_dsc_mean', is_pct=False)} "
        f"| {_fmt_delta(point_delta=float(tr['vertex_dsc_mean'] - t1['vertex_dsc_mean']), boot_diffs=diffs['vertex_dsc_mean'], is_pct=False)} "
        f"| {_fmt_p(p_dsc)} |"
    )

    # Cluster-level (boxDSC>0.22) [+ optional Dice safeguard]
    lines.append(
        f"| Cluster-level ({box_label}) | Predicted clusters per subject, mean [95% CI] "
        f"| {_fmt_point_ci(point=float(t1['clusters_per_subject_mean']), boot=t1_boot, key='clusters_per_subject_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(fm['clusters_per_subject_mean']), boot=fm_boot, key='clusters_per_subject_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(tr['clusters_per_subject_mean']), boot=tr_boot, key='clusters_per_subject_mean', is_pct=False)} "
        f"| {_fmt_delta(point_delta=float(tr['clusters_per_subject_mean'] - t1['clusters_per_subject_mean']), boot_diffs=diffs['clusters_per_subject_mean'], is_pct=False)} "
        f"| {_fmt_p(p_nclu)} |"
    )
    lines.append(
        f"| Cluster-level ({box_label}) | False-positive clusters per subject, mean [95% CI] "
        f"| {_fmt_point_ci(point=float(t1['fp_per_patient_mean']), boot=t1_boot, key='fp_per_patient_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(fm['fp_per_patient_mean']), boot=fm_boot, key='fp_per_patient_mean', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(tr['fp_per_patient_mean']), boot=tr_boot, key='fp_per_patient_mean', is_pct=False)} "
        f"| {_fmt_delta(point_delta=float(tr['fp_per_patient_mean'] - t1['fp_per_patient_mean']), boot_diffs=diffs['fp_per_patient_mean'], is_pct=False)} "
        f"| {_fmt_p(p_fp)} |"
    )
    lines.append(
        f"| Cluster-level ({box_label}) | Precision (TP clusters / all clusters), % [95% CI] "
        f"| {_fmt_point_ci(point=float(t1['precision_boxdsc']), boot=t1_boot, key='precision_boxdsc', is_pct=True)} "
        f"| {_fmt_point_ci(point=float(fm['precision_boxdsc']), boot=fm_boot, key='precision_boxdsc', is_pct=True)} "
        f"| {_fmt_point_ci(point=float(tr['precision_boxdsc']), boot=tr_boot, key='precision_boxdsc', is_pct=True)} "
        f"| {_fmt_delta(point_delta=float(tr['precision_boxdsc'] - t1['precision_boxdsc']), boot_diffs=diffs['precision_boxdsc'], is_pct=True)} "
        f"| {_fmt_p(p_prec)} |"
    )
    lines.append(
        f"| Cluster-level ({box_label}) | F1 score, mean [95% CI] "
        f"| {_fmt_point_ci(point=float(t1['f1_boxdsc']), boot=t1_boot, key='f1_boxdsc', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(fm['f1_boxdsc']), boot=fm_boot, key='f1_boxdsc', is_pct=False)} "
        f"| {_fmt_point_ci(point=float(tr['f1_boxdsc']), boot=tr_boot, key='f1_boxdsc', is_pct=False)} "
        f"| {_fmt_delta(point_delta=float(tr['f1_boxdsc'] - t1['f1_boxdsc']), boot_diffs=diffs['f1_boxdsc'], is_pct=False)} "
        f"| {_fmt_p(p_f1)} |"
    )

    # Subject-level
    lines.append(
        f"| Subject-level (primary) | Detection rate (Det({box_label})), n/N (%) [95% CI] "
        f"| {_fmt_n_pct_ci(n=int(t1['n_detected_boxdsc']), N=N, boot=t1_boot, key='detection_rate_boxdsc')} "
        f"| {_fmt_n_pct_ci(n=int(fm['n_detected_boxdsc']), N=N, boot=fm_boot, key='detection_rate_boxdsc')} "
        f"| {_fmt_n_pct_ci(n=int(tr['n_detected_boxdsc']), N=N, boot=tr_boot, key='detection_rate_boxdsc')} "
        f"| {_fmt_delta(point_delta=float(tr['detection_rate_boxdsc'] - t1['detection_rate_boxdsc']), boot_diffs=diffs['detection_rate_boxdsc'], is_pct=False)} "
        f"| {_fmt_p(p_det)} (McNemar; n01={det_n01}, n10={det_n10}) |"
    )
    lines.append(
        "| Subject-level (co-primary) | Detection rate (Det(PPV-in-mask≥0.5)), n/N (%) [95% CI] "
        f"| {_fmt_n_pct_ci(n=int(t1['n_detected_ppv50']), N=N, boot=t1_boot, key='detection_rate_ppv50')} "
        f"| {_fmt_n_pct_ci(n=int(fm['n_detected_ppv50']), N=N, boot=fm_boot, key='detection_rate_ppv50')} "
        f"| {_fmt_n_pct_ci(n=int(tr['n_detected_ppv50']), N=N, boot=tr_boot, key='detection_rate_ppv50')} "
        f"| {_fmt_delta(point_delta=float(tr['detection_rate_ppv50'] - t1['detection_rate_ppv50']), boot_diffs=diffs['detection_rate_ppv50'], is_pct=False)} "
        f"| {_fmt_p(p_ppv)} (McNemar; n01={ppv_n01}, n10={ppv_n10}; boot-p={p_det_ppv_boot:.4f}) |"
    )
    lines.append(
        "| Subject-level | Detection rate (Det(distance≤20mm)), n/N (%) [95% CI] "
        f"| {_fmt_n_pct_ci(n=int(t1['n_detected_distance']), N=N, boot=t1_boot, key='detection_rate_distance')} "
        f"| {_fmt_n_pct_ci(n=int(fm['n_detected_distance']), N=N, boot=fm_boot, key='detection_rate_distance')} "
        f"| {_fmt_n_pct_ci(n=int(tr['n_detected_distance']), N=N, boot=tr_boot, key='detection_rate_distance')} "
        f"| {_fmt_delta(point_delta=float(tr['detection_rate_distance'] - t1['detection_rate_distance']), boot_diffs=diffs['detection_rate_distance'], is_pct=False)} "
        f"| {_fmt_p(p_dist)} (McNemar; n01={dist_n01}, n10={dist_n10}; boot-p={p_det_dist_boot:.4f}) |"
    )
    lines.append(
        "| Subject-level | Pinpointing rate (any cluster COM within lesion), n/N (%) [95% CI] "
        f"| {_fmt_n_pct_ci(n=int(t1['n_pinpointed']), N=N, boot=t1_boot, key='pinpointing_rate')} "
        f"| {_fmt_n_pct_ci(n=int(fm['n_pinpointed']), N=N, boot=fm_boot, key='pinpointing_rate')} "
        f"| {_fmt_n_pct_ci(n=int(tr['n_pinpointed']), N=N, boot=tr_boot, key='pinpointing_rate')} "
        f"| {_fmt_delta(point_delta=float(tr['pinpointing_rate'] - t1['pinpointing_rate']), boot_diffs=diffs['pinpointing_rate'], is_pct=False)} "
        f"| {_fmt_p(p_pin)} (McNemar; n01={pin_n01}, n10={pin_n10}; boot-p={p_pin_boot:.4f}) |"
    )
    lines.append(
        "| Subject-level | Recall@0.15 (lesion Dice union ≥ 0.15), n/N (%) [95% CI] "
        f"| {_fmt_n_pct_ci(n=int(t1['n_recall_at_0.15']), N=N, boot=t1_boot, key='recall_at_0.15')} "
        f"| {_fmt_n_pct_ci(n=int(fm['n_recall_at_0.15']), N=N, boot=fm_boot, key='recall_at_0.15')} "
        f"| {_fmt_n_pct_ci(n=int(tr['n_recall_at_0.15']), N=N, boot=tr_boot, key='recall_at_0.15')} "
        f"| {_fmt_delta(point_delta=float(tr['recall_at_0.15'] - t1['recall_at_0.15']), boot_diffs=diffs['recall_at_0.15'], is_pct=False)} "
        f"| {_fmt_p(p_rec015)} (McNemar; n01={rec015_n01}, n10={rec015_n10}; boot-p={p_rec015_boot:.4f}) |"
    )

    lines.append("")
    lines.append("## Notes (for manuscript footnotes)")
    lines.append("")
    lines.append(f"1) **Cohort (n={N})**: intermediate/difficult cases within the original 52-case seizure-free cohort, restricted to ILAE 1–2.")
    lines.append("2) **Reference standard**: resection cavity mask.")
    lines.append("3) **Vertex-level DSC**: computed on the lesion hemisphere (rows with `lesion_area_cm2>0` in `vertex_level_results.csv`).")
    if float(args.boxdsc_min_cluster_dice) > 0.0:
        lines.append(f"4) **Cluster-level criterion**: a cluster is a TP if `boxDSC > 0.22` and `Dice > {float(args.boxdsc_min_cluster_dice):g}`.")
    else:
        lines.append("4) **Cluster-level criterion**: a cluster is a TP if `boxDSC > 0.22`.")
    lines.append("5) **Precision**: TP clusters / all predicted clusters (pooled across subjects).")
    lines.append(
        "6) **Sensitivity vs detection rate**: in this seizure-free cohort (all positive), the cluster-level sensitivity "
        "(subjects with ≥1 TP cluster / all subjects) is mathematically equivalent to the subject-level detection rate "
        "under the same TP definition; therefore, we report only the detection rate to avoid redundancy."
    )
    lines.append("7) **Pinpointing**: at least one predicted cluster has its weighted center-of-mass (COM) within the lesion mask.")
    lines.append("8) **Det(PPV-in-mask≥0.5)**: subject-level endpoint; a subject is detected if ≥1 predicted cluster has PPV-in-mask ≥ 0.5.")
    lines.append("9) **Det(distance≤20mm)**: subject-level endpoint; a subject is detected if ≥1 predicted cluster has distance-to-lesion ≤ 20 mm (MELD Graph criterion).")
    lines.append("10) **Recall@0.15**: subject-level endpoint; `lesion_dice_union ≥ 0.15`.")
    lines.append(f"11) **Uncertainty**: 95% CIs from subject-level bootstrapping (`n_boot={args.n_boot}`).")
    lines.append(f"12) **Paired statistics ({args.tracka_name} vs {args.t1_name})**:")
    lines.append("   - DSC / cluster counts / FP burden: paired sign-flip randomization test.")
    lines.append("   - Detection rate / pinpointing / Recall@0.15: exact McNemar’s test (reported with discordant counts n01/n10).")
    lines.append("   - Precision / F1: bootstrap-based two-sided p-value from the empirical Δ distribution.")
    if args.p_adjust == "holm3_fdr_bh":
        lines.append(
            "13) **Multiple comparisons**: p values are multiplicity-adjusted using a hierarchical scheme: "
            "Holm–Bonferroni for the **primary family** of 3 endpoints "
            "(Det(boxDSC>0.22 & Dice safeguard), Det(PPV-in-mask≥0.5), Pinpointing), "
            "and Benjamini–Hochberg FDR for the **secondary family** of the remaining metrics."
        )
    elif args.p_adjust != "none":
        lines.append(
            f"13) **Multiple comparisons**: p values are adjusted across all metrics in this table "
            f"(family size={len(p_raw_list)}; method={args.p_adjust})."
        )
    if args.constraints_note:
        lines.append(f"14) **Constraints**: {args.constraints_note}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.out_subjects_tsv is not None:
        _write_subject_ids(subject_ids, args.out_subjects_tsv.resolve())

    if args.out_audit_json is not None:
        audit = {
            "scope": {
                "definition": "paper52 ∩ MRIDetectability{intermediate,difficult} ∩ ILAE{1,2}",
                "n": len(subject_ids),
                "meta_csv": _relpath(args.meta_csv),
                "paper_table2_csv": _relpath(args.paper_table2_csv),
            },
            "inputs": {
                "t1_eval_dir": _relpath(Path(t1_dir)),
                "fmri_eval_dir": _relpath(Path(fmri_dir)),
                "tracka_eval_dir": _relpath(Path(track_dir)),
            },
            "params": {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "n_perm": args.n_perm,
                "boxdsc_min_cluster_dice": float(args.boxdsc_min_cluster_dice),
                "p_adjust": str(args.p_adjust),
            },
            "out_md": _relpath(Path(out_md)),
            "labels": {
                "t1_name": args.t1_name,
                "fmri_name": args.fmri_name,
                "tracka_name": args.tracka_name,
                "delta_label": delta_label,
                "p_label": p_label,
                "constraints_note": args.constraints_note,
            },
        }
        args.out_audit_json.resolve().write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
