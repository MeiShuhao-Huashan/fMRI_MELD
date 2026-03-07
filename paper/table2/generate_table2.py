#!/usr/bin/env python3
"""
Generate publication-ready Table 2 (multi-scale performance) with:
- point estimates (pooled across subjects)
- 95% bootstrap CIs (subject-level bootstrap)
- paired tests (Track A vs T1-only)

Inputs are MELD three-level evaluation outputs:
  - preferred: {eval_dir}/all_folds_subject_level_results.csv (and *_cluster/_vertex)
  - fallback: {eval_dir}/fold{0-4}/val/subject_level_results.csv (and *_cluster/_vertex)
  - fallback: {eval_dir}/subject_level_results.csv (and *_cluster/_vertex)

This script intentionally avoids pandas/scipy to keep dependencies minimal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _pct(x: float) -> float:
    return 100.0 * x


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    w = pos - lo
    return float(sorted_values[lo] * (1 - w) + sorted_values[hi] * w)


def _ci95(values: Sequence[float]) -> Tuple[float, float]:
    sv = sorted(values)
    return (_quantile(sv, 0.025), _quantile(sv, 0.975))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _format_mean_ci(x: float, lo: float, hi: float, *, decimals: int = 3) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(x)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _format_pct_ci(x: float, lo: float, hi: float, *, decimals: int = 1) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(_pct(x))}% [{fmt.format(_pct(lo))}, {fmt.format(_pct(hi))}]"


def _format_n_pct_ci(n: int, N: int, lo: float, hi: float, *, decimals: int = 1) -> str:
    pct = _pct(n / N) if N else float("nan")
    fmt = f"{{:.{decimals}f}}"
    return f"{n}/{N} ({fmt.format(pct)}%) [{fmt.format(_pct(lo))}, {fmt.format(_pct(hi))}]"


def _two_sided_boot_pvalue(diffs: Sequence[float]) -> float:
    """
    Bootstrap-based two-sided p-value from the empirical distribution of diffs.
    p = 2 * min(P(diff<=0), P(diff>=0)).
    """
    if not diffs:
        return float("nan")
    n = len(diffs)
    le0 = sum(1 for d in diffs if d <= 0) / n
    ge0 = sum(1 for d in diffs if d >= 0) / n
    return float(min(1.0, 2.0 * min(le0, ge0)))


def _signflip_pvalue(diffs: Sequence[float], *, n_perm: int, seed: int) -> float:
    """
    Paired randomization test (sign-flip) on the mean difference.
    """
    if not diffs:
        return float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    extreme = 0
    for _ in range(int(n_perm)):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= obs:
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def _mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Tuple[float, int, int]:
    """
    Exact McNemar test for paired binary outcomes.
    Returns: (p_value, n01, n10), where
      n01 = # (a=0,b=1), n10 = # (a=1,b=0)
    """
    if len(a) != len(b):
        raise ValueError("McNemar inputs must have same length.")
    n01 = sum((not ai) and bi for ai, bi in zip(a, b))
    n10 = sum(ai and (not bi) for ai, bi in zip(a, b))
    n = n01 + n10
    if n == 0:
        return (1.0, n01, n10)
    k = min(n01, n10)
    # two-sided exact binomial test under p=0.5
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return (float(min(1.0, p)), n01, n10)


@dataclass(frozen=True)
class MethodInputs:
    name: str
    eval_dir: Path


@dataclass(frozen=True)
class SubjectRow:
    subject_id: str
    fold: int
    vertex_dsc_lesion_hemi: float
    lesion_dice_union: float
    detected_boxdsc: bool
    detected_distance: bool
    detected_ppv50: bool
    pinpointed: bool
    n_clusters: int
    n_fp_clusters: int


@dataclass(frozen=True)
class ClusterRow:
    subject_id: str
    fold: int
    is_tp_boxdsc: bool
    is_tp_ppv50: bool


def _iter_eval_csvs(eval_dir: Path, filename: str) -> Iterable[Tuple[int, Path]]:
    """
    Yield CSV paths for an eval_dir, supporting multiple layouts:
      1) all-folds file:  eval_dir/all_folds_{filename}
      2) flat file:       eval_dir/{filename}
      3) fold structure:  eval_dir/fold{0-4}/val/{filename}

    Returns tuples (fold_hint, path), where fold_hint == -1 indicates "unknown/flat".
    """
    all_folds = eval_dir / f"all_folds_{filename}"
    if all_folds.exists():
        yield (-1, all_folds)
        return
    flat = eval_dir / filename
    if flat.exists():
        yield (-1, flat)
        return
    any_found = False
    for fold in range(5):
        p = eval_dir / f"fold{fold}" / "val" / filename
        if p.exists():
            any_found = True
            yield (fold, p)
    if not any_found:
        raise FileNotFoundError(f"Missing {filename} under eval_dir (checked all_folds/flat/folds): {eval_dir}")


def _load_subject_rows(eval_dir: Path) -> Dict[str, SubjectRow]:
    vertex_dsc_by_subject: Dict[str, float] = {}
    for _, p in _iter_eval_csvs(eval_dir, "vertex_level_results.csv"):
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if float(row.get("lesion_area_cm2", 0.0)) <= 0:
                    continue
                sid = str(row["subject_id"])
                vertex_dsc_by_subject[sid] = float(row["DSC"])

    out: Dict[str, SubjectRow] = {}
    for fold_hint, p in _iter_eval_csvs(eval_dir, "subject_level_results.csv"):
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = str(row["subject_id"])
                if sid not in vertex_dsc_by_subject:
                    raise ValueError(f"Missing lesion-hemi vertex DSC for subject: {sid} (eval_dir={eval_dir})")
                fold = int(row.get("fold", fold_hint))
                out[sid] = SubjectRow(
                    subject_id=sid,
                    fold=fold,
                    vertex_dsc_lesion_hemi=float(vertex_dsc_by_subject[sid]),
                    lesion_dice_union=float(row["lesion_dice_union"]),
                    detected_boxdsc=_parse_bool(row["detected_boxdsc"]),
                    detected_distance=_parse_bool(row.get("detected_distance", False)),
                    detected_ppv50=_parse_bool(row.get("detected_ppv50", False)),
                    pinpointed=_parse_bool(row["pinpointed"]),
                    n_clusters=int(row["n_clusters"]),
                    n_fp_clusters=int(row["n_fp_clusters"]),
                )
    return out


def _load_cluster_rows(eval_dir: Path, *, boxdsc_min_cluster_dice: float = 0.0) -> Dict[str, List[ClusterRow]]:
    out: Dict[str, List[ClusterRow]] = {}
    for fold_hint, p in _iter_eval_csvs(eval_dir, "cluster_level_results.csv"):
        with p.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = str(row["subject_id"])
                fold = int(row.get("fold", fold_hint))
                is_tp_boxdsc = _parse_bool(row["is_tp_boxdsc"])
                if float(boxdsc_min_cluster_dice) > 0.0 and is_tp_boxdsc:
                    try:
                        dice = float(row.get("dice", 0.0))
                    except Exception:
                        dice = 0.0
                    if not (dice > float(boxdsc_min_cluster_dice)):
                        is_tp_boxdsc = False
                out.setdefault(sid, []).append(
                    ClusterRow(
                        subject_id=sid,
                        fold=fold,
                        is_tp_boxdsc=is_tp_boxdsc,
                        is_tp_ppv50=_parse_bool(row.get("is_tp_ppv50", False)),
                    )
                )
    return out


def _compute_metrics(
    subject_ids: Sequence[str],
    subject_rows: Dict[str, SubjectRow],
    cluster_rows: Dict[str, List[ClusterRow]],
) -> Dict[str, float]:
    n = len(subject_ids)
    dsc = [subject_rows[sid].vertex_dsc_lesion_hemi for sid in subject_ids]
    n_clusters = [subject_rows[sid].n_clusters for sid in subject_ids]
    fp_clusters = [subject_rows[sid].n_fp_clusters for sid in subject_ids]
    detected = [subject_rows[sid].detected_boxdsc for sid in subject_ids]
    detected_distance = [subject_rows[sid].detected_distance for sid in subject_ids]
    detected_ppv50 = [subject_rows[sid].detected_ppv50 for sid in subject_ids]
    pinpointed = [subject_rows[sid].pinpointed for sid in subject_ids]
    recall_015 = [subject_rows[sid].lesion_dice_union >= 0.15 for sid in subject_ids]

    # Cluster-level precision (boxDSC) pooled across all predicted clusters
    tp = 0
    tp_ppv50 = 0
    total = 0
    for sid in subject_ids:
        for c in cluster_rows.get(sid, []):
            total += 1
            if c.is_tp_boxdsc:
                tp += 1
            if c.is_tp_ppv50:
                tp_ppv50 += 1
    precision = (tp / total) if total else 0.0
    precision_ppv50 = (tp_ppv50 / total) if total else 0.0

    sensitivity = sum(detected) / n if n else float("nan")
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    sensitivity_distance = sum(detected_distance) / n if n else float("nan")

    sensitivity_ppv50 = (sum(detected_ppv50) / n) if n else float("nan")
    f1_ppv50 = (
        (2 * precision_ppv50 * sensitivity_ppv50 / (precision_ppv50 + sensitivity_ppv50))
        if (precision_ppv50 + sensitivity_ppv50) > 0
        else 0.0
    )

    fp_ppv50_per_subject = []
    for sid in subject_ids:
        cs = cluster_rows.get(sid, [])
        tp_s = sum(1 for c in cs if c.is_tp_ppv50)
        fp_ppv50_per_subject.append(len(cs) - tp_s)

    return {
        "n_subjects": float(n),
        "vertex_dsc_mean": _mean(dsc),
        "clusters_per_subject_mean": _mean(n_clusters),
        "fp_per_patient_mean": _mean(fp_clusters),
        "fp_per_patient_ppv50_mean": _mean(fp_ppv50_per_subject),
        "precision_boxdsc": precision,
        "precision_ppv50": precision_ppv50,
        "sensitivity_boxdsc": sensitivity,
        "sensitivity_ppv50": sensitivity_ppv50,
        "f1_boxdsc": f1,
        "f1_ppv50": f1_ppv50,
        "detection_rate_boxdsc": sensitivity,  # equals subject detection in this cohort
        "detection_rate_distance": sensitivity_distance,
        "detection_rate_ppv50": (sum(detected_ppv50) / n) if n else float("nan"),
        "pinpointing_rate": (sum(pinpointed) / n) if n else float("nan"),
        "recall_at_0.15": (sum(recall_015) / n) if n else float("nan"),
        "n_detected_boxdsc": float(sum(detected)),
        "n_detected_distance": float(sum(detected_distance)),
        "n_detected_ppv50": float(sum(detected_ppv50)),
        "n_pinpointed": float(sum(pinpointed)),
        "n_recall_at_0.15": float(sum(recall_015)),
        "n_total_clusters": float(total),
        "n_tp_clusters_boxdsc": float(tp),
        "n_tp_clusters_ppv50": float(tp_ppv50),
    }


def _bootstrap(
    subject_ids: Sequence[str],
    *,
    subject_rows: Dict[str, SubjectRow],
    cluster_rows: Dict[str, List[ClusterRow]],
    n_boot: int,
    seed: int,
) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    out: List[Dict[str, float]] = []
    n = len(subject_ids)
    for _ in range(int(n_boot)):
        sample = [subject_ids[rng.randrange(n)] for _ in range(n)]
        out.append(_compute_metrics(sample, subject_rows, cluster_rows))
    return out


def _extract_series(samples: Sequence[Dict[str, float]], key: str) -> List[float]:
    return [float(s[key]) for s in samples]


def _write_table2_tsv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_table2_md(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("")
        return
    cols = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r[c] for c in cols) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def _write_source_data(
    *,
    subject_ids: Sequence[str],
    t1_subject: Dict[str, SubjectRow],
    track_subject: Dict[str, SubjectRow],
    fmri_subject: Optional[Dict[str, SubjectRow]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "subject_id",
        "fold_t1",
        "t1_vertex_dsc_lesion_hemi",
        "t1_lesion_dice_union",
        "t1_detected_boxdsc",
        "t1_detected_distance",
        "t1_detected_ppv50",
        "t1_pinpointed",
        "t1_n_clusters",
        "t1_n_fp_clusters",
        "trackA_vertex_dsc_lesion_hemi",
        "trackA_lesion_dice_union",
        "trackA_detected_boxdsc",
        "trackA_detected_distance",
        "trackA_detected_ppv50",
        "trackA_pinpointed",
        "trackA_n_clusters",
        "trackA_n_fp_clusters",
    ]
    if fmri_subject is not None:
        cols += [
            "fmri_vertex_dsc_lesion_hemi",
            "fmri_lesion_dice_union",
            "fmri_detected_boxdsc",
            "fmri_detected_distance",
            "fmri_detected_ppv50",
            "fmri_pinpointed",
            "fmri_n_clusters",
            "fmri_n_fp_clusters",
        ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sid in subject_ids:
            t1 = t1_subject[sid]
            tr = track_subject[sid]
            row = {
                "subject_id": sid,
                "fold_t1": str(t1.fold if t1.fold != -1 else tr.fold),
                "t1_vertex_dsc_lesion_hemi": f"{t1.vertex_dsc_lesion_hemi:.6f}",
                "t1_lesion_dice_union": f"{t1.lesion_dice_union:.6f}",
                "t1_detected_boxdsc": str(t1.detected_boxdsc),
                "t1_detected_distance": str(t1.detected_distance),
                "t1_detected_ppv50": str(t1.detected_ppv50),
                "t1_pinpointed": str(t1.pinpointed),
                "t1_n_clusters": str(t1.n_clusters),
                "t1_n_fp_clusters": str(t1.n_fp_clusters),
                "trackA_vertex_dsc_lesion_hemi": f"{tr.vertex_dsc_lesion_hemi:.6f}",
                "trackA_lesion_dice_union": f"{tr.lesion_dice_union:.6f}",
                "trackA_detected_boxdsc": str(tr.detected_boxdsc),
                "trackA_detected_distance": str(tr.detected_distance),
                "trackA_detected_ppv50": str(tr.detected_ppv50),
                "trackA_pinpointed": str(tr.pinpointed),
                "trackA_n_clusters": str(tr.n_clusters),
                "trackA_n_fp_clusters": str(tr.n_fp_clusters),
            }
            if fmri_subject is not None:
                fm = fmri_subject[sid]
                row.update(
                    {
                        "fmri_vertex_dsc_lesion_hemi": f"{fm.vertex_dsc_lesion_hemi:.6f}",
                        "fmri_lesion_dice_union": f"{fm.lesion_dice_union:.6f}",
                        "fmri_detected_boxdsc": str(fm.detected_boxdsc),
                        "fmri_detected_distance": str(fm.detected_distance),
                        "fmri_detected_ppv50": str(fm.detected_ppv50),
                        "fmri_pinpointed": str(fm.pinpointed),
                        "fmri_n_clusters": str(fm.n_clusters),
                        "fmri_n_fp_clusters": str(fm.n_fp_clusters),
                    }
                )
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1_eval_dir", required=True, type=Path)
    ap.add_argument("--tracka_eval_dir", required=True, type=Path)
    ap.add_argument("--fmri_eval_dir", default=None, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--n_boot", default=10000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--n_perm", default=20000, type=int)
    args = ap.parse_args()

    t1_subject = _load_subject_rows(args.t1_eval_dir)
    t1_cluster = _load_cluster_rows(args.t1_eval_dir)
    track_subject = _load_subject_rows(args.tracka_eval_dir)
    track_cluster = _load_cluster_rows(args.tracka_eval_dir)

    fmri_subject = None
    fmri_cluster = None
    if args.fmri_eval_dir is not None:
        fmri_subject = _load_subject_rows(args.fmri_eval_dir)
        fmri_cluster = _load_cluster_rows(args.fmri_eval_dir)

    subject_ids = sorted(set(t1_subject.keys()))
    if set(track_subject.keys()) != set(subject_ids):
        raise ValueError("T1 and TrackA subject sets differ.")
    if fmri_subject is not None and set(fmri_subject.keys()) != set(subject_ids):
        raise ValueError("fMRI subject set differs from T1.")

    # Point estimates (pooled)
    t1_metrics = _compute_metrics(subject_ids, t1_subject, t1_cluster)
    track_metrics = _compute_metrics(subject_ids, track_subject, track_cluster)
    fmri_metrics = (
        _compute_metrics(subject_ids, fmri_subject, fmri_cluster)
        if (fmri_subject is not None and fmri_cluster is not None)
        else None
    )

    # Bootstrap CIs
    t1_boot = _bootstrap(subject_ids, subject_rows=t1_subject, cluster_rows=t1_cluster, n_boot=args.n_boot, seed=args.seed)
    track_boot = _bootstrap(
        subject_ids,
        subject_rows=track_subject,
        cluster_rows=track_cluster,
        n_boot=args.n_boot,
        seed=args.seed + 1,
    )
    fmri_boot = None
    if fmri_subject is not None and fmri_cluster is not None:
        fmri_boot = _bootstrap(
            subject_ids,
            subject_rows=fmri_subject,
            cluster_rows=fmri_cluster,
            n_boot=args.n_boot,
            seed=args.seed + 2,
        )

    # Paired bootstrap diffs (TrackA - T1) for global metrics
    rng = random.Random(args.seed + 3)
    diffs_boot: Dict[str, List[float]] = {}
    for _ in range(int(args.n_boot)):
        sample = [subject_ids[rng.randrange(len(subject_ids))] for _ in range(len(subject_ids))]
        a = _compute_metrics(sample, track_subject, track_cluster)
        b = _compute_metrics(sample, t1_subject, t1_cluster)
        for k in [
            "vertex_dsc_mean",
            "clusters_per_subject_mean",
            "fp_per_patient_mean",
            "fp_per_patient_ppv50_mean",
            "precision_boxdsc",
            "sensitivity_boxdsc",
            "detection_rate_boxdsc",
            "detection_rate_distance",
            "detection_rate_ppv50",
            "f1_boxdsc",
            "precision_ppv50",
            "sensitivity_ppv50",
            "f1_ppv50",
            "pinpointing_rate",
            "recall_at_0.15",
        ]:
            diffs_boot.setdefault(k, []).append(float(a[k] - b[k]))

    # Paired tests
    dsc_diffs = [
        track_subject[sid].vertex_dsc_lesion_hemi - t1_subject[sid].vertex_dsc_lesion_hemi for sid in subject_ids
    ]
    fp_diffs = [track_subject[sid].n_fp_clusters - t1_subject[sid].n_fp_clusters for sid in subject_ids]
    nclu_diffs = [track_subject[sid].n_clusters - t1_subject[sid].n_clusters for sid in subject_ids]
    p_dsc = _signflip_pvalue(dsc_diffs, n_perm=args.n_perm, seed=args.seed + 10)
    p_fp = _signflip_pvalue(fp_diffs, n_perm=args.n_perm, seed=args.seed + 11)
    p_nclu = _signflip_pvalue(nclu_diffs, n_perm=args.n_perm, seed=args.seed + 12)

    def _fp_ppv50_count(sid: str, clusters: Dict[str, List[ClusterRow]]) -> int:
        cs = clusters.get(sid, [])
        return int(len(cs) - sum(1 for c in cs if c.is_tp_ppv50))

    fp_ppv50_diffs = [_fp_ppv50_count(sid, track_cluster) - _fp_ppv50_count(sid, t1_cluster) for sid in subject_ids]
    p_fp_ppv50 = _signflip_pvalue(fp_ppv50_diffs, n_perm=args.n_perm, seed=args.seed + 13)

    t1_det = [t1_subject[sid].detected_boxdsc for sid in subject_ids]
    tr_det = [track_subject[sid].detected_boxdsc for sid in subject_ids]
    p_det, det_n01, det_n10 = _mcnemar_exact(t1_det, tr_det)

    t1_det_dist = [t1_subject[sid].detected_distance for sid in subject_ids]
    tr_det_dist = [track_subject[sid].detected_distance for sid in subject_ids]
    p_det_dist, detdist_n01, detdist_n10 = _mcnemar_exact(t1_det_dist, tr_det_dist)

    t1_det_ppv50 = [t1_subject[sid].detected_ppv50 for sid in subject_ids]
    tr_det_ppv50 = [track_subject[sid].detected_ppv50 for sid in subject_ids]
    p_det_ppv50, detppv_n01, detppv_n10 = _mcnemar_exact(t1_det_ppv50, tr_det_ppv50)

    t1_pin = [t1_subject[sid].pinpointed for sid in subject_ids]
    tr_pin = [track_subject[sid].pinpointed for sid in subject_ids]
    p_pin, pin_n01, pin_n10 = _mcnemar_exact(t1_pin, tr_pin)

    t1_rec015 = [t1_subject[sid].lesion_dice_union >= 0.15 for sid in subject_ids]
    tr_rec015 = [track_subject[sid].lesion_dice_union >= 0.15 for sid in subject_ids]
    p_rec015, rec015_n01, rec015_n10 = _mcnemar_exact(t1_rec015, tr_rec015)

    # Bootstrap p-values for global precision/F1/sensitivity/pinpoint (TrackA-T1)
    p_prec = _two_sided_boot_pvalue(diffs_boot["precision_boxdsc"])
    p_sens = _two_sided_boot_pvalue(diffs_boot["sensitivity_boxdsc"])
    p_f1 = _two_sided_boot_pvalue(diffs_boot["f1_boxdsc"])
    p_pin_boot = _two_sided_boot_pvalue(diffs_boot["pinpointing_rate"])
    p_det_ppv50_boot = _two_sided_boot_pvalue(diffs_boot["detection_rate_ppv50"])
    p_det_dist_boot = _two_sided_boot_pvalue(diffs_boot["detection_rate_distance"])
    p_rec015_boot = _two_sided_boot_pvalue(diffs_boot["recall_at_0.15"])
    p_prec_ppv50 = _two_sided_boot_pvalue(diffs_boot["precision_ppv50"])
    p_sens_ppv50 = _two_sided_boot_pvalue(diffs_boot["sensitivity_ppv50"])
    p_f1_ppv50 = _two_sided_boot_pvalue(diffs_boot["f1_ppv50"])

    # Compose table rows
    def metric_row(scale: str, metric: str) -> Dict[str, str]:
        return {"Scale": scale, "Metric": metric}

    def add_vals(
        row: Dict[str, str],
        key: str,
        *,
        is_pct: bool = False,
        n_count_key: Optional[str] = None,
    ) -> None:
        # T1
        t1_vals = _extract_series(t1_boot, key)
        tr_vals = _extract_series(track_boot, key)
        t1_lo, t1_hi = _ci95(t1_vals)
        tr_lo, tr_hi = _ci95(tr_vals)
        if n_count_key is not None:
            row["T1-only"] = _format_n_pct_ci(int(t1_metrics[n_count_key]), len(subject_ids), t1_lo, t1_hi)
            row["Track A (T1+fMRI)"] = _format_n_pct_ci(
                int(track_metrics[n_count_key]), len(subject_ids), tr_lo, tr_hi
            )
        elif is_pct:
            row["T1-only"] = _format_pct_ci(float(t1_metrics[key]), t1_lo, t1_hi)
            row["Track A (T1+fMRI)"] = _format_pct_ci(float(track_metrics[key]), tr_lo, tr_hi)
        else:
            row["T1-only"] = _format_mean_ci(float(t1_metrics[key]), t1_lo, t1_hi)
            row["Track A (T1+fMRI)"] = _format_mean_ci(float(track_metrics[key]), tr_lo, tr_hi)

        # fMRI-only (optional)
        if fmri_metrics is not None and fmri_boot is not None:
            fm_vals = _extract_series(fmri_boot, key)
            fm_lo, fm_hi = _ci95(fm_vals)
            if n_count_key is not None:
                row["fMRI-only (fMRIhemi B2a)"] = _format_n_pct_ci(
                    int(fmri_metrics[n_count_key]), len(subject_ids), fm_lo, fm_hi
                )
            elif is_pct:
                row["fMRI-only (fMRIhemi B2a)"] = _format_pct_ci(float(fmri_metrics[key]), fm_lo, fm_hi)
            else:
                row["fMRI-only (fMRIhemi B2a)"] = _format_mean_ci(float(fmri_metrics[key]), fm_lo, fm_hi)
        else:
            row["fMRI-only (fMRIhemi B2a)"] = ""

        # Delta + p (TrackA-T1)
        dvals = diffs_boot[key]
        dlo, dhi = _ci95(dvals)
        if is_pct:
            row["Δ (TrackA − T1)"] = _format_pct_ci(float(track_metrics[key] - t1_metrics[key]), dlo, dhi)
        else:
            row["Δ (TrackA − T1)"] = _format_mean_ci(float(track_metrics[key] - t1_metrics[key]), dlo, dhi)

    rows: List[Dict[str, str]] = []
    rows_ppv50_cluster: List[Dict[str, str]] = []  # moved to Supplementary Table

    # Vertex level
    r = metric_row("Vertex-level", "Dice similarity coefficient (DSC), mean [95% CI]")
    add_vals(r, "vertex_dsc_mean", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_dsc:.4f}"
    rows.append(r)

    # Cluster level (boxDSC>0.22)
    r = metric_row("Cluster-level (boxDSC>0.22)", "Predicted clusters per subject, mean [95% CI]")
    add_vals(r, "clusters_per_subject_mean", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_nclu:.4f}"
    rows.append(r)

    r = metric_row("Cluster-level (boxDSC>0.22)", "False-positive clusters per subject, mean [95% CI]")
    add_vals(r, "fp_per_patient_mean", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_fp:.4f}"
    rows.append(r)

    r = metric_row("Cluster-level (boxDSC>0.22)", "Precision (TP clusters / all clusters), % [95% CI]")
    add_vals(r, "precision_boxdsc", is_pct=True)
    r["p (TrackA vs T1)"] = f"{p_prec:.4f}"
    rows.append(r)

    r = metric_row("Cluster-level (boxDSC>0.22)", "Sensitivity (≥1 TP cluster), % [95% CI]")
    add_vals(r, "sensitivity_boxdsc", is_pct=True)
    r["p (TrackA vs T1)"] = f"{p_sens:.4f}"
    rows.append(r)

    r = metric_row("Cluster-level (boxDSC>0.22)", "F1 score, mean [95% CI]")
    add_vals(r, "f1_boxdsc", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_f1:.4f}"
    rows.append(r)

    # Cluster level (PPV-in-mask ≥ 0.5) — move to Supplementary Table (keep subject-level Det(PPV50) in main Table 2)
    r = metric_row("Cluster-level (PPV-in-mask≥0.5)", "False-positive clusters per subject, mean [95% CI]")
    add_vals(r, "fp_per_patient_ppv50_mean", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_fp_ppv50:.4f}"
    rows_ppv50_cluster.append(r)

    r = metric_row("Cluster-level (PPV-in-mask≥0.5)", "Precision (TP clusters / all clusters), % [95% CI]")
    add_vals(r, "precision_ppv50", is_pct=True)
    r["p (TrackA vs T1)"] = f"{p_prec_ppv50:.4f}"
    rows_ppv50_cluster.append(r)

    r = metric_row("Cluster-level (PPV-in-mask≥0.5)", "Sensitivity (≥1 TP cluster), % [95% CI]")
    add_vals(r, "sensitivity_ppv50", is_pct=True)
    r["p (TrackA vs T1)"] = f"{p_sens_ppv50:.4f}"
    rows_ppv50_cluster.append(r)

    r = metric_row("Cluster-level (PPV-in-mask≥0.5)", "F1 score, mean [95% CI]")
    add_vals(r, "f1_ppv50", is_pct=False)
    r["p (TrackA vs T1)"] = f"{p_f1_ppv50:.4f}"
    rows_ppv50_cluster.append(r)

    # Subject level (primary endpoint)
    r = metric_row("Subject-level (primary)", "Detection rate (Det(boxDSC>0.22)), n/N (%) [95% CI]")
    add_vals(r, "detection_rate_boxdsc", n_count_key="n_detected_boxdsc")
    r["p (TrackA vs T1)"] = f"{p_det:.4f} (McNemar; n01={det_n01}, n10={det_n10})"
    rows.append(r)

    r = metric_row("Subject-level (co-primary)", "Detection rate (Det(PPV-in-mask≥0.5)), n/N (%) [95% CI]")
    add_vals(r, "detection_rate_ppv50", n_count_key="n_detected_ppv50")
    r["p (TrackA vs T1)"] = (
        f"{p_det_ppv50:.4f} (McNemar; n01={detppv_n01}, n10={detppv_n10}; boot-p={p_det_ppv50_boot:.4f})"
    )
    rows.append(r)

    r = metric_row("Subject-level", "Detection rate (Det(distance≤20mm)), n/N (%) [95% CI]")
    add_vals(r, "detection_rate_distance", n_count_key="n_detected_distance")
    r["p (TrackA vs T1)"] = (
        f"{p_det_dist:.4f} (McNemar; n01={detdist_n01}, n10={detdist_n10}; boot-p={p_det_dist_boot:.4f})"
    )
    rows.append(r)

    r = metric_row("Subject-level", "Pinpointing rate (any cluster COM within lesion), n/N (%) [95% CI]")
    add_vals(r, "pinpointing_rate", n_count_key="n_pinpointed")
    r["p (TrackA vs T1)"] = f"{p_pin:.4f} (McNemar; n01={pin_n01}, n10={pin_n10}; boot-p={p_pin_boot:.4f})"
    rows.append(r)

    r = metric_row("Subject-level", "Recall@0.15 (lesion Dice union ≥ 0.15), n/N (%) [95% CI]")
    add_vals(r, "recall_at_0.15", n_count_key="n_recall_at_0.15")
    r["p (TrackA vs T1)"] = (
        f"{p_rec015:.4f} (McNemar; n01={rec015_n01}, n10={rec015_n10}; boot-p={p_rec015_boot:.4f})"
    )
    rows.append(r)

    # Add column order (stable)
    ordered_cols = [
        "Scale",
        "Metric",
        "T1-only",
        "fMRI-only (fMRIhemi B2a)",
        "Track A (T1+fMRI)",
        "Δ (TrackA − T1)",
        "p (TrackA vs T1)",
    ]
    rows = [{k: r.get(k, "") for k in ordered_cols} for r in rows]
    rows_ppv50_cluster = [{k: r.get(k, "") for k in ordered_cols} for r in rows_ppv50_cluster]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table2_tsv(rows, out_dir / "table2_multiscale_performance.tsv")
    _write_table2_md(rows, out_dir / "table2_multiscale_performance.md")
    if rows_ppv50_cluster:
        _write_table2_tsv(rows_ppv50_cluster, out_dir / "tableS_ppv50_cluster_level.tsv")
        _write_table2_md(rows_ppv50_cluster, out_dir / "tableS_ppv50_cluster_level.md")

    # Save metrics JSON for audit
    audit = {
        "inputs": {
            "t1_eval_dir": str(args.t1_eval_dir.resolve()),
            "tracka_eval_dir": str(args.tracka_eval_dir.resolve()),
            "fmri_eval_dir": str(args.fmri_eval_dir.resolve()) if args.fmri_eval_dir else None,
        },
        "params": {"n_boot": args.n_boot, "seed": args.seed, "n_perm": args.n_perm},
        "point_estimates": {"t1": t1_metrics, "trackA": track_metrics, "fmri": fmri_metrics},
        "paired_tests": {
            "DSC_signflip_p": p_dsc,
            "FP_signflip_p": p_fp,
            "clusters_per_subject_signflip_p": p_nclu,
            "FP_PPV50_signflip_p": p_fp_ppv50,
            "Det_boxDSC_mcnemar_p": p_det,
            "Det_boxDSC_n01": det_n01,
            "Det_boxDSC_n10": det_n10,
            "Det_distance_mcnemar_p": p_det_dist,
            "Det_distance_n01": detdist_n01,
            "Det_distance_n10": detdist_n10,
            "Det_PPV50_mcnemar_p": p_det_ppv50,
            "Det_PPV50_n01": detppv_n01,
            "Det_PPV50_n10": detppv_n10,
            "Pinpoint_mcnemar_p": p_pin,
            "Pinpoint_n01": pin_n01,
            "Pinpoint_n10": pin_n10,
            "Recall015_mcnemar_p": p_rec015,
            "Recall015_n01": rec015_n01,
            "Recall015_n10": rec015_n10,
            "Precision_boot_p": p_prec,
            "Sensitivity_boot_p": p_sens,
            "F1_boot_p": p_f1,
            "Pinpoint_boot_p": p_pin_boot,
            "Det_PPV50_boot_p": p_det_ppv50_boot,
            "Det_distance_boot_p": p_det_dist_boot,
            "Recall015_boot_p": p_rec015_boot,
            "Precision_PPV50_boot_p": p_prec_ppv50,
            "Sensitivity_PPV50_boot_p": p_sens_ppv50,
            "F1_PPV50_boot_p": p_f1_ppv50,
        },
    }
    (out_dir / "table2_audit.json").write_text(json.dumps(audit, indent=2))

    # Source data for quick spot-checking / pairing
    _write_source_data(
        subject_ids=subject_ids,
        t1_subject=t1_subject,
        track_subject=track_subject,
        fmri_subject=fmri_subject,
        out_path=out_dir / "table2_source_data.csv",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
