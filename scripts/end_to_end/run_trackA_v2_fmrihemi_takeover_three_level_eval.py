#!/usr/bin/env python3
"""
Deploy-real Track A evaluation: T1-first with fmrihemi(V2) takeover.

Key constraints (no leakage):
  - No oracle lesion hemisphere for gating or fMRI mapping
  - fMRI hemisphere mapping uses an fMRI-only laterality classifier (fold-specific CSV)
  - Gate thresholds are fixed constants (or optionally inner-calibrated in a separate step)

This script writes, per fold:
  - predictions_val.hdf5 (fused: either T1 or V2 per subject)
  - gate_decisions.csv (traceability)
  - deploy_config.json (all parameters)
  - three-level evaluation outputs via meld_graph.three_level_evaluation.ThreeLevelEvaluator

And for all folds:
  - all_folds_{subject,cluster,vertex}_level_results.csv
  - aggregate_val.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse.csgraph

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from meld_fmri.fmri_gcn.atlas import load_atlas_brainnetome
from meld_graph.meld_cohort import MeldCohort
from meld_fmri.three_level_evaluation import ThreeLevelEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("meld_fmri.trackA_v2_fmrihemi_takeover_eval")


# T1 predictions HDF5 (out-of-fold per fold)
T1_PRED_HDF5_BY_FOLD = {
    0: "meld_data/models/25-12-16_MELD_fMRI_fold0/fold_00/results_best_model/predictions_val.hdf5",
    1: "meld_data/models/25-12-15_MELD_fMRI_fold1/fold_01/results_best_model/predictions_val.hdf5",
    2: "meld_data/models/25-12-15_MELD_fMRI_fold2/fold_02/results_best_model/predictions_val.hdf5",
    3: "meld_data/models/25-12-16_MELD_fMRI_fold3/fold_03/results_best_model/predictions_val.hdf5",
    4: "meld_data/models/25-12-16_MELD_fMRI_fold4/fold_04/results_best_model/predictions_val.hdf5",
}

# Fold splits (reuse T1 baseline splits strictly)
SPLIT_JSONS = {
    0: "meld_data/models/25-12-16_MELD_fMRI_fold0/fold_00/data_parameters.json",
    1: "meld_data/models/25-12-15_MELD_fMRI_fold1/fold_01/data_parameters.json",
    2: "meld_data/models/25-12-15_MELD_fMRI_fold2/fold_02/data_parameters.json",
    3: "meld_data/models/25-12-16_MELD_fMRI_fold3/fold_03/data_parameters.json",
    4: "meld_data/models/25-12-16_MELD_fMRI_fold4/fold_04/data_parameters.json",
}

LESION_ROOT_DEFAULT = "meld_data/derived_labels/lesion_main_island/template_fsaverage_sym_xhemi"
DEFAULT_SELECTOR_T1_SUBJECT_CSV = "meld_data/output/three_level_eval/t1_baseline_val52/unbudgeted_ppv/subject_level_results.csv"
DEFAULT_SELECTOR_FMRI_SUBJECT_CSV = (
    "meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80/all_folds_subject_level_results.csv"
)


class NodeLayout:
    """S26/V2 node ordering: ipsi cortex | ipsi subcort | contra cortex | contra subcort | midline."""

    n_parcels: int = 105
    n_sub_hemi: int = 9

    @property
    def idx_ipsi_cortex(self) -> slice:
        return slice(0, self.n_parcels)

    @property
    def idx_contra_cortex(self) -> slice:
        start = self.n_parcels + self.n_sub_hemi
        return slice(start, start + self.n_parcels)


def _resolve_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (PROJECT_DIR / pp).resolve()
    return pp


def _load_fold_path_map_json(path: str | Path) -> Dict[int, Path]:
    p = _resolve_path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (dict) in {p}, got: {type(data).__name__}")
    out: Dict[int, Path] = {}
    for k, v in data.items():
        fold = int(k)
        out[fold] = _resolve_path(str(v))
    return out


def _fold_path_map_from_template(template: str, *, label: str) -> Dict[int, Path]:
    tpl = str(template).strip()
    if not tpl:
        return {}
    out: Dict[int, Path] = {}
    for fold in range(5):
        try:
            s = tpl.format(fold=fold)
        except Exception as e:
            raise ValueError(f"Invalid {label}_template={tpl!r}: {e}") from e
        out[int(fold)] = _resolve_path(s)
    return out


def _build_fold_path_map(
    *,
    label: str,
    json_path: str,
    template: str,
    default_map: Dict[int, str | Path],
) -> Dict[int, Path]:
    if str(json_path).strip():
        mp = _load_fold_path_map_json(str(json_path).strip())
    else:
        mp = _fold_path_map_from_template(template, label=label) if str(template).strip() else {}

    if not mp:
        mp = {int(f): _resolve_path(default_map[int(f)]) for f in range(5)}

    missing = [f for f in range(5) if int(f) not in mp]
    if missing:
        raise ValueError(f"Missing {label} for folds: {missing}. Provide --{label}_by_fold_json or --{label}_template.")

    return {int(f): mp[int(f)] for f in range(5)}


def _load_fold_splits(split_json_by_fold: Dict[int, Path]) -> Dict[int, Dict[str, List[str]]]:
    splits: Dict[int, Dict[str, List[str]]] = {}
    for fold in range(5):
        path = _resolve_path(split_json_by_fold[int(fold)])
        data = json.loads(path.read_text(encoding="utf-8"))
        splits[int(fold)] = {"train_ids": list(data["train_ids"]), "val_ids": list(data["val_ids"])}
    return splits


def _laterality_csv_for_fold(fold: int) -> Path:
    """
    Fold-specific fMRI laterality predictions (deploy-real, fMRI-only).

    NOTE: We must NOT use oracle columns (true_hemi/correct) in this CSV for gating;
    only `prob_right` and `pred_hemi` are used.
    """
    fold = int(fold)
    if fold == 0:
        p = PROJECT_DIR / "meld_data/models/20260106_fmri_laterality_absLR_dualExpert_fold0/val_predictions.csv"
        if p.is_file():
            return p
    p = PROJECT_DIR / f"meld_data/models/20260106_fmri_laterality_absLR_dualExpert_fold{fold}_balacc/val_predictions.csv"
    if p.is_file():
        return p
    # fallback
    p2 = PROJECT_DIR / f"meld_data/models/20260106_fmri_laterality_absLR_dualExpert_fold{fold}/val_predictions.csv"
    return p2


def _load_fmri_pred_hemi_by_sid(path: Path) -> Dict[str, Dict[str, float | str]]:
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing laterality CSV: {path}")
    out: Dict[str, Dict[str, float | str]] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row.get("subject_id", "")).strip()
            if not sid:
                continue
            pred_hemi = str(row.get("pred_hemi", "")).strip()
            prob_right = float(row.get("prob_right", "nan"))
            if pred_hemi not in {"L", "R"}:
                continue
            out[sid] = {
                "pred_hemi": pred_hemi,
                "prob_right": prob_right,
                "source_csv": str(path),
            }
    return out


def _top_cluster_mean_score(pred: np.ndarray, clustered: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    clustered = np.asarray(clustered, dtype=np.int32).reshape(-1)
    if pred.shape != clustered.shape:
        raise ValueError(f"pred/clustered shape mismatch: {pred.shape} vs {clustered.shape}")
    cluster_ids = np.unique(clustered)
    cluster_ids = cluster_ids[cluster_ids > 0]
    if cluster_ids.size == 0:
        return 0.0
    best = 0.0
    for cid in cluster_ids.tolist():
        m = clustered == int(cid)
        if not np.any(m):
            continue
        s = float(pred[m].mean())
        if s > best:
            best = s
    return float(best)


def _count_clusters(*, cl_lh: np.ndarray, cl_rh: np.ndarray) -> int:
    cl_lh = np.asarray(cl_lh, dtype=np.int32).reshape(-1)
    cl_rh = np.asarray(cl_rh, dtype=np.int32).reshape(-1)
    n_lh = int(np.unique(cl_lh[cl_lh > 0]).size)
    n_rh = int(np.unique(cl_rh[cl_rh > 0]).size)
    return int(n_lh + n_rh)


def _build_sid_to_fold(splits: Dict[int, Dict[str, List[str]]]) -> Dict[str, int]:
    sid_to_fold: Dict[str, int] = {}
    for f, rec in splits.items():
        for sid in rec.get("val_ids", []):
            sid_to_fold[str(sid)] = int(f)
    return sid_to_fold


def _load_detected_map(path: Path, *, col: str) -> Dict[str, bool]:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        raise ValueError(f"Expected subject_id column in {path}")
    if col not in df.columns:
        raise ValueError(f"Expected {col} column in {path}")
    return {str(s): bool(v) for s, v in zip(df["subject_id"].astype(str), df[col].astype(bool))}


def _selector_feature_vector(
    *,
    t1_conf: float,
    t1_conf_lh: float,
    t1_conf_rh: float,
    t1_has_cluster: bool,
    t1_n_clusters: int,
    fmri_available: bool,
    fmri_top_score: float,
    fmri_minus_t1: float,
    lat_margin: float,
    hemi_conflict: bool,
    fmri_n_clusters: int,
    t1_top_area_cm2: float,
    fmri_top_area_cm2: float,
    t1_fmri_top_dice: float,
    t1_fmri_top_minfrac: float,
) -> np.ndarray:
    def f(x: float) -> float:
        return float(x) if np.isfinite(x) else 0.0

    return np.asarray(
        [
            f(t1_conf),
            f(t1_conf_lh),
            f(t1_conf_rh),
            1.0 if bool(t1_has_cluster) else 0.0,
            float(int(t1_n_clusters)),
            1.0 if bool(fmri_available) else 0.0,
            f(fmri_top_score),
            f(fmri_minus_t1),
            f(lat_margin),
            1.0 if bool(hemi_conflict) else 0.0,
            float(int(fmri_n_clusters)),
            f(t1_top_area_cm2),
            f(fmri_top_area_cm2),
            f(t1_fmri_top_dice),
            f(t1_fmri_top_minfrac),
        ],
        dtype=np.float32,
    )


def _train_gate_selector(
    *,
    fold: int,
    v2_dir: Path,
    args: argparse.Namespace,
    post: "V2DeployPostprocessor",
    layout: "NodeLayout",
    splits: Dict[int, Dict[str, List[str]]],
    t1_pred_hdf5_by_fold: Dict[int, Path],
    laterality_csv_by_fold: Dict[int, Path],
) -> Tuple[object, float, Dict[str, object]]:
    """
    Train a leakage-free gate selector for fold `fold` using only train_ids.

    Label definition (train-only, GT-derived but no val leakage):
      y=1 iff (T1 baseline misses) AND (fMRIhemi(V2) hits) under the chosen endpoint.

    Inputs are deploy-available scalars derived from prediction maps and laterality probabilities.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    gate_label = str(getattr(args, "selector_label", "boxdsc")).strip().lower()
    if gate_label == "ppv50":
        det_col = "detected_ppv50"
    else:
        det_col = "detected_boxdsc"

    t1_csv = _resolve_path(getattr(args, "selector_t1_subject_level_csv", DEFAULT_SELECTOR_T1_SUBJECT_CSV))
    fmri_csv = _resolve_path(getattr(args, "selector_fmri_subject_level_csv", DEFAULT_SELECTOR_FMRI_SUBJECT_CSV))
    t1_det = _load_detected_map(t1_csv, col=det_col)
    fmri_det = _load_detected_map(fmri_csv, col=det_col)

    sid_to_fold = _build_sid_to_fold(splits)

    fmri_hemi_by_fold = {f: _load_fmri_pred_hemi_by_sid(laterality_csv_by_fold[int(f)]) for f in range(5)}
    t1_h5_by_fold = {f: h5py.File(str(_resolve_path(t1_pred_hdf5_by_fold[int(f)])), "r") for f in range(5)}

    X_train: List[np.ndarray] = []
    y_train: List[int] = []
    skipped = 0
    missing_fmri = 0
    missing_label = 0
    try:
        for sid in splits[int(fold)]["train_ids"]:
            sid = str(sid)
            f_sid = sid_to_fold.get(sid)
            if f_sid is None:
                skipped += 1
                continue

            if sid not in t1_det or sid not in fmri_det:
                missing_label += 1
                continue

            f_sid = int(f_sid)
            f_t1 = t1_h5_by_fold[f_sid]
            if sid not in f_t1:
                skipped += 1
                continue

            pred_t1_lh = np.asarray(f_t1[sid]["lh"]["prediction"][:], dtype=np.float32).reshape(-1)
            pred_t1_rh = np.asarray(f_t1[sid]["rh"]["prediction"][:], dtype=np.float32).reshape(-1)
            if args.t1_cluster_key in f_t1[sid]["lh"]:
                cl_t1_lh = np.asarray(f_t1[sid]["lh"][args.t1_cluster_key][:], dtype=np.int32).reshape(-1)
                cl_t1_rh = np.asarray(f_t1[sid]["rh"][args.t1_cluster_key][:], dtype=np.int32).reshape(-1)
            else:
                cl_t1_lh = np.zeros_like(pred_t1_lh, dtype=np.int32)
                cl_t1_rh = np.zeros_like(pred_t1_rh, dtype=np.int32)

            t1_has_cluster = bool((cl_t1_lh > 0).any() or (cl_t1_rh > 0).any())
            t1_conf_lh = float(_top_cluster_mean_score(pred_t1_lh, cl_t1_lh))
            t1_conf_rh = float(_top_cluster_mean_score(pred_t1_rh, cl_t1_rh))
            t1_conf = float(max(t1_conf_lh, t1_conf_rh))
            t1_pred_hemi = "L" if t1_conf_lh >= t1_conf_rh else "R"
            t1_n_clusters = _count_clusters(cl_lh=cl_t1_lh, cl_rh=cl_t1_rh)
            t1_top_area_cm2 = 0.0
            t1_top_mask_lh = np.zeros_like(pred_t1_lh, dtype=bool)
            t1_top_mask_rh = np.zeros_like(pred_t1_rh, dtype=bool)
            if t1_has_cluster:
                t1_stats = _cluster_stats(
                    pred_lh=pred_t1_lh,
                    pred_rh=pred_t1_rh,
                    cl_lh=cl_t1_lh,
                    cl_rh=cl_t1_rh,
                    score_mode=str(args.cluster_sort),
                )
                if t1_stats:
                    t = t1_stats[0]
                    if str(t["hemi"]) == "lh":
                        t1_top_mask_lh = t["mask"]
                        t1_top_area_cm2 = float(np.sum(post.area_cortex_cm2[t1_top_mask_lh]))
                    else:
                        t1_top_mask_rh = t["mask"]
                        t1_top_area_cm2 = float(np.sum(post.area_cortex_cm2[t1_top_mask_rh]))

            fmri_available = False
            fmri_pred_hemi = ""
            prob_right = float("nan")
            lat_margin = float("nan")
            fmri_top_score = float("nan")
            fmri_minus_t1 = float("nan")
            fmri_n_clusters = 0
            fmri_top_area_cm2 = 0.0
            fmri_top_mask_lh = np.zeros_like(pred_t1_lh, dtype=bool)
            fmri_top_mask_rh = np.zeros_like(pred_t1_rh, dtype=bool)

            if sid not in fmri_hemi_by_fold[f_sid]:
                missing_fmri += 1
            else:
                p_npz = v2_dir / f"fold_{f_sid:02d}" / "val_predictions" / f"{sid}.npz"
                if not p_npz.is_file():
                    missing_fmri += 1
                else:
                    fmri_pred_hemi = str(fmri_hemi_by_fold[f_sid][sid]["pred_hemi"])
                    prob_right = float(fmri_hemi_by_fold[f_sid][sid]["prob_right"])
                    lat_margin = float(abs(prob_right - 0.5))

                    probs = np.asarray(np.load(str(p_npz))["p"], dtype=np.float32).reshape(-1)
                    p_t = _transform_probs(probs, str(args.prob_transform))
                    ipsi = p_t[layout.idx_ipsi_cortex]
                    contra = p_t[layout.idx_contra_cortex]
                    if fmri_pred_hemi == "L":
                        left_parc, right_parc = ipsi, contra
                    else:
                        left_parc, right_parc = contra, ipsi

                    res = post.postprocess(
                        left_parc=left_parc,
                        right_parc=right_parc,
                        area_target_cm2=float(args.area_target_cm2),
                        cluster_sort=str(args.cluster_sort),
                    )
                    cl_v2_lh = res["clustered_lh"].astype(np.int32)
                    cl_v2_rh = res["clustered_rh"].astype(np.int32)
                    fmri_available = bool((cl_v2_lh > 0).any() or (cl_v2_rh > 0).any())
                    fmri_n_clusters = _count_clusters(cl_lh=cl_v2_lh, cl_rh=cl_v2_rh)
                    if fmri_available:
                        pred_v2_lh = res["pred_lh"].astype(np.float32)
                        pred_v2_rh = res["pred_rh"].astype(np.float32)
                        stats = _cluster_stats(
                            pred_lh=pred_v2_lh,
                            pred_rh=pred_v2_rh,
                            cl_lh=cl_v2_lh,
                            cl_rh=cl_v2_rh,
                            score_mode=str(args.cluster_sort),
                        )
                        fmri_top_score = float(stats[0]["score"]) if stats else float("nan")
                        fmri_minus_t1 = float(fmri_top_score - t1_conf) if np.isfinite(fmri_top_score) else float("nan")
                        if stats:
                            top = stats[0]
                            if str(top["hemi"]) == "lh":
                                fmri_top_mask_lh = top["mask"]
                                fmri_top_area_cm2 = float(np.sum(post.area_cortex_cm2[fmri_top_mask_lh]))
                            else:
                                fmri_top_mask_rh = top["mask"]
                                fmri_top_area_cm2 = float(np.sum(post.area_cortex_cm2[fmri_top_mask_rh]))

            hemi_conflict = (t1_pred_hemi != fmri_pred_hemi) if fmri_pred_hemi in {"L", "R"} else False
            inter = float(
                np.sum(post.area_cortex_cm2[t1_top_mask_lh & fmri_top_mask_lh])
                + np.sum(post.area_cortex_cm2[t1_top_mask_rh & fmri_top_mask_rh])
            )
            denom = float(t1_top_area_cm2 + fmri_top_area_cm2)
            t1_fmri_top_dice = float((2.0 * inter / denom) if denom > 0 else 0.0)
            min_area = float(min(t1_top_area_cm2, fmri_top_area_cm2))
            t1_fmri_top_minfrac = float((inter / min_area) if min_area > 0 else 0.0)
            X_train.append(
                _selector_feature_vector(
                    t1_conf=t1_conf,
                    t1_conf_lh=t1_conf_lh,
                    t1_conf_rh=t1_conf_rh,
                    t1_has_cluster=t1_has_cluster,
                    t1_n_clusters=t1_n_clusters,
                    fmri_available=fmri_available,
                    fmri_top_score=fmri_top_score,
                    fmri_minus_t1=fmri_minus_t1,
                    lat_margin=lat_margin,
                    hemi_conflict=hemi_conflict,
                    fmri_n_clusters=fmri_n_clusters,
                    t1_top_area_cm2=t1_top_area_cm2,
                    fmri_top_area_cm2=fmri_top_area_cm2,
                    t1_fmri_top_dice=t1_fmri_top_dice,
                    t1_fmri_top_minfrac=t1_fmri_top_minfrac,
                )
            )
            y_train.append(1 if ((not bool(t1_det[sid])) and bool(fmri_det[sid])) else 0)
    finally:
        for f in range(5):
            try:
                t1_h5_by_fold[int(f)].close()
            except Exception:
                pass

    if not X_train:
        raise RuntimeError(f"[fold={fold}] selector: no training samples (skipped={skipped}, missing_label={missing_label})")

    Xtr = np.stack(X_train, axis=0)
    ytr = np.asarray(y_train, dtype=np.int64)
    selector = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )
    selector.fit(Xtr, ytr)
    train_acc = float(np.mean((selector.predict(Xtr) == ytr).astype(np.float32)))
    pos = int(np.sum(ytr == 1))
    neg = int(np.sum(ytr == 0))

    probs = selector.predict_proba(Xtr)[:, 1].astype(np.float64)
    if pos <= 0:
        thr = 1.0
    else:
        thr = float(max(0.0, min(1.0, float(np.min(probs[ytr == 1]) - 1e-6))))

    meta = {
        "fold": int(fold),
        "label": str(gate_label),
        "det_col": str(det_col),
        "t1_subject_level_csv": str(t1_csv),
        "fmri_subject_level_csv": str(fmri_csv),
        "n_train": int(Xtr.shape[0]),
        "n_pos": int(pos),
        "n_neg": int(neg),
        "train_acc": float(train_acc),
        "threshold_strategy": "min_pos_prob_minus_eps (train-only)",
        "threshold": float(thr),
        "missing_fmri": int(missing_fmri),
        "missing_label": int(missing_label),
        "skipped": int(skipped),
        "feature_names": [
            "t1_conf",
            "t1_conf_lh",
            "t1_conf_rh",
            "t1_has_cluster",
            "t1_n_clusters",
            "fmri_available",
            "fmri_top_score",
            "fmri_minus_t1",
            "lat_margin",
            "hemi_conflict",
            "fmri_n_clusters",
            "t1_top_area_cm2",
            "fmri_top_area_cm2",
            "t1_fmri_top_dice",
            "t1_fmri_top_minfrac",
        ],
    }
    logger.info(
        "[fold=%d] Trained gate selector: train_acc=%.3f n=%d pos=%d neg=%d thr=%.4f (missing_fmri=%d, missing_label=%d, skipped=%d)",
        int(fold),
        float(train_acc),
        int(Xtr.shape[0]),
        int(pos),
        int(neg),
        float(thr),
        int(missing_fmri),
        int(missing_label),
        int(skipped),
    )
    return selector, float(thr), meta


def _cluster_stats(
    *,
    pred_lh: np.ndarray,
    pred_rh: np.ndarray,
    cl_lh: np.ndarray,
    cl_rh: np.ndarray,
    score_mode: str,
) -> List[Dict]:
    """
    Return per-cluster stats across both hemispheres.

    score_mode: 'mean' | 'max'
    """
    out: List[Dict] = []
    for hemi, pred, cl in (("lh", pred_lh, cl_lh), ("rh", pred_rh, cl_rh)):
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        cl = np.asarray(cl, dtype=np.int32).reshape(-1)
        for cid in np.unique(cl):
            cid_i = int(cid)
            if cid_i <= 0:
                continue
            m = cl == cid_i
            if not np.any(m):
                continue
            if score_mode == "max":
                score = float(np.max(pred[m]))
            else:
                score = float(np.mean(pred[m]))
            out.append({"hemi": hemi, "cid": cid_i, "score": score, "mask": m})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _apply_union_fusion(
    *,
    pred_t1_lh: np.ndarray,
    pred_t1_rh: np.ndarray,
    cl_t1_lh: np.ndarray,
    cl_t1_rh: np.ndarray,
    pred_fmri_lh: np.ndarray,
    pred_fmri_rh: np.ndarray,
    cl_fmri_lh: np.ndarray,
    cl_fmri_rh: np.ndarray,
    k_total: int,
    k_fmri: int,
    cluster_sort: str,
    label_components_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Union-style fusion without oracle: keep a limited number of T1 clusters + fMRI clusters.

    Strategy:
      - Keep top `k_fmri` fMRI clusters (by `cluster_sort` score).
      - Keep up to `k_total - k_fmri_kept` T1 clusters (by the same score).
      - Output prediction maps with pred>0.5 exactly on the kept cluster masks.
    """
    k_total = int(k_total)
    k_fmri = int(k_fmri)
    if k_total <= 0:
        raise ValueError("k_total must be > 0")
    if k_fmri < 0:
        raise ValueError("k_fmri must be >= 0")

    fmri_stats = _cluster_stats(
        pred_lh=pred_fmri_lh,
        pred_rh=pred_fmri_rh,
        cl_lh=cl_fmri_lh,
        cl_rh=cl_fmri_rh,
        score_mode=str(cluster_sort),
    )
    fmri_keep = fmri_stats[: min(k_fmri, len(fmri_stats))]

    remaining = max(0, k_total - len(fmri_keep))
    t1_stats = _cluster_stats(
        pred_lh=pred_t1_lh,
        pred_rh=pred_t1_rh,
        cl_lh=cl_t1_lh,
        cl_rh=cl_t1_rh,
        score_mode=str(cluster_sort),
    )
    t1_keep = t1_stats[: min(remaining, len(t1_stats))]

    keep_lh = np.zeros_like(cl_t1_lh, dtype=bool)
    keep_rh = np.zeros_like(cl_t1_rh, dtype=bool)

    for c in t1_keep:
        if str(c["hemi"]) == "lh":
            keep_lh |= c["mask"]
        else:
            keep_rh |= c["mask"]

    for c in fmri_keep:
        if str(c["hemi"]) == "lh":
            keep_lh |= c["mask"]
        else:
            keep_rh |= c["mask"]

    pred_out_lh = np.zeros_like(pred_t1_lh, dtype=np.float32)
    pred_out_rh = np.zeros_like(pred_t1_rh, dtype=np.float32)

    if np.any(keep_lh):
        t1_scaled = 0.51 + 0.49 * np.clip(pred_t1_lh, 0.0, 1.0)
        pred_out_lh[keep_lh] = np.maximum(t1_scaled[keep_lh], pred_fmri_lh[keep_lh])
    if np.any(keep_rh):
        t1_scaled = 0.51 + 0.49 * np.clip(pred_t1_rh, 0.0, 1.0)
        pred_out_rh[keep_rh] = np.maximum(t1_scaled[keep_rh], pred_fmri_rh[keep_rh])

    cl_out_lh = label_components_fn(keep_lh).astype(np.int32)
    cl_out_rh = label_components_fn(keep_rh).astype(np.int32)

    return pred_out_lh, pred_out_rh, cl_out_lh, cl_out_rh


def _transform_probs(probs: np.ndarray, mode: str) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if mode == "raw":
        return probs
    if mode == "centered":
        eps = 1e-6
        p = np.clip(probs, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        logit_centered = logit - np.median(logit)
        return (1.0 / (1.0 + np.exp(-logit_centered))).astype(np.float32)
    if mode == "rank":
        ranks = np.argsort(np.argsort(probs))
        return (ranks / float(len(ranks))).astype(np.float32)
    return probs


class V2DeployPostprocessor:
    """V2 parcel probs (L/R cortex 105) -> vertex prediction + clusters under deploy constraints."""

    def __init__(
        self,
        *,
        area_max_cluster_cm2: float,
        area_max_total_cm2: float,
        k_clusters: int,
        min_vertices: int,
    ) -> None:
        self.area_max_cluster_cm2 = float(area_max_cluster_cm2)
        self.area_max_total_cm2 = float(area_max_total_cm2)
        self.k_clusters = int(k_clusters)
        self.min_vertices = int(min_vertices)

        cohort = MeldCohort()
        self.cortex_mask = cohort.cortex_mask.astype(bool)
        self.adj_mat = cohort.adj_mat  # full sparse adjacency (fsaverage_sym)
        self.surf_area = cohort.surf_area.astype(np.float64)  # mm^2

        self.area_cortex_cm2 = (self.surf_area[self.cortex_mask] / 100.0).astype(np.float64)  # cortex only
        # Subject-level budgets are applied across both hemispheres.
        self.area_cm2 = np.concatenate([self.area_cortex_cm2, self.area_cortex_cm2])  # (2*n_cortex,)
        self.n_cortex = int(self.cortex_mask.sum())

        atlas = load_atlas_brainnetome()
        self.parcel_index = atlas.fsaverage_sym_lh_parcel_index  # length NVERT, 0..104 on cortex

        self.parcel_area_cm2 = np.zeros(105, dtype=np.float64)
        for p in range(105):
            self.parcel_area_cm2[p] = float(
                np.sum((self.surf_area / 100.0)[(self.parcel_index == p) & self.cortex_mask])
            )

        self.adj_mat_cortex = self.adj_mat[self.cortex_mask][:, self.cortex_mask]

    def _cluster_and_filter(self, mask_full: np.ndarray, island_count: int) -> Tuple[np.ndarray, int]:
        islands = np.zeros(mask_full.shape[0], dtype=np.int32)
        if int(mask_full.sum()) <= 0:
            return islands, island_count
        n_comp, labels = scipy.sparse.csgraph.connected_components(self.adj_mat[mask_full][:, mask_full])
        for comp_idx in range(int(n_comp)):
            include_vec = labels == comp_idx
            if int(np.sum(include_vec)) < int(self.min_vertices):
                continue
            island_count += 1
            island_mask = mask_full.copy()
            island_mask[mask_full] = include_vec
            islands[island_mask] = island_count
        return islands, island_count

    def _label_connected_components_cortex(self, mask: np.ndarray) -> np.ndarray:
        labels_out = np.zeros(mask.shape[0], dtype=np.int32)
        if int(mask.sum()) <= 0:
            return labels_out
        n_comp, labels = scipy.sparse.csgraph.connected_components(self.adj_mat_cortex[mask][:, mask])
        cluster_id = 0
        for comp_idx in range(int(n_comp)):
            include_vec = labels == comp_idx
            if int(np.sum(include_vec)) < int(self.min_vertices):
                continue
            cluster_id += 1
            cc_mask = mask.copy()
            cc_mask[mask] = include_vec
            labels_out[cc_mask] = cluster_id
        return labels_out

    def _trim_cluster_by_probability(self, cluster_mask: np.ndarray, pred: np.ndarray, target_area: float) -> np.ndarray:
        if not np.any(cluster_mask):
            return cluster_mask
        current_area = float(np.sum(self.area_cm2[cluster_mask]))
        if current_area <= target_area + 1e-12:
            return cluster_mask

        cluster_indices = np.where(cluster_mask)[0]
        cluster_probs = pred[cluster_indices]
        sorted_order = np.argsort(cluster_probs)[::-1]
        sorted_indices = cluster_indices[sorted_order]

        trimmed_mask = np.zeros_like(cluster_mask)
        accumulated_area = 0.0
        for idx in sorted_indices:
            vertex_area = float(self.area_cm2[idx])
            if accumulated_area + vertex_area > target_area + 1e-12:
                break
            trimmed_mask[idx] = True
            accumulated_area += vertex_area
        return trimmed_mask

    def postprocess(
        self,
        *,
        left_parc: np.ndarray,
        right_parc: np.ndarray,
        area_target_cm2: float,
        cluster_sort: str,
    ) -> Dict[str, np.ndarray]:
        left_parc = np.asarray(left_parc, dtype=np.float32).reshape(-1)
        right_parc = np.asarray(right_parc, dtype=np.float32).reshape(-1)
        if left_parc.shape[0] != 105 or right_parc.shape[0] != 105:
            raise ValueError(f"Expected (105,) left/right parcel probs; got {left_parc.shape} {right_parc.shape}")

        # Area-target parcel selection (by pooled score across both hemis).
        scores = np.concatenate([left_parc, right_parc])
        order = np.argsort(scores)[::-1]

        selected_idx: List[int] = []
        area_sum = 0.0
        for idx in order:
            selected_idx.append(int(idx))
            p = int(idx if idx < 105 else idx - 105)
            area_sum += float(self.parcel_area_cm2[p])
            if area_sum >= float(area_target_cm2):
                break

        left_sel = set(i for i in selected_idx if i < 105)
        right_sel = set(i - 105 for i in selected_idx if i >= 105)

        mask_lh = np.zeros_like(self.cortex_mask, dtype=bool)
        mask_rh = np.zeros_like(self.cortex_mask, dtype=bool)
        m = (self.parcel_index >= 0) & self.cortex_mask
        if left_sel:
            mask_lh[m] = np.isin(self.parcel_index[m], np.array(sorted(left_sel), dtype=np.int32))
        if right_sel:
            mask_rh[m] = np.isin(self.parcel_index[m], np.array(sorted(right_sel), dtype=np.int32))

        pred_lh = np.zeros_like(self.cortex_mask, dtype=np.float32)
        pred_rh = np.zeros_like(self.cortex_mask, dtype=np.float32)
        pred_lh[m] = left_parc[self.parcel_index[m]]
        pred_rh[m] = right_parc[self.parcel_index[m]]

        island_count = 0
        cl_lh, island_count = self._cluster_and_filter(mask_lh, island_count)
        cl_rh, island_count = self._cluster_and_filter(mask_rh, island_count)

        score_lh = pred_lh[self.cortex_mask]
        score_rh = pred_rh[self.cortex_mask]
        clustered = np.concatenate([cl_lh[self.cortex_mask], cl_rh[self.cortex_mask]])
        pred = np.concatenate([score_lh, score_rh])

        # Score clusters.
        clusters = []
        for cid in np.unique(clustered):
            if int(cid) <= 0:
                continue
            cm = clustered == int(cid)
            if not np.any(cm):
                continue
            if str(cluster_sort) == "max":
                score = float(np.max(pred[cm]))
            else:
                score = float(np.mean(pred[cm]))
            clusters.append((score, int(cid), cm))
        clusters.sort(key=lambda x: x[0], reverse=True)

        keep = np.zeros_like(clustered, dtype=bool)
        total_area = 0.0
        kept = 0
        for _score, _cid, cm in clusters:
            if kept >= int(self.k_clusters):
                break
            area = float(np.sum(self.area_cm2[cm]))
            remaining = float(self.area_max_total_cm2) - total_area
            if remaining <= 0:
                break

            if area <= float(self.area_max_cluster_cm2) + 1e-12 and area <= remaining + 1e-12:
                keep |= cm
                total_area += area
                kept += 1
                continue

            trim_target = min(float(self.area_max_cluster_cm2), float(remaining))
            trimmed = self._trim_cluster_by_probability(cm, pred, trim_target)
            if np.any(trimmed):
                keep |= trimmed
                total_area += float(np.sum(self.area_cm2[trimmed]))
                kept += 1

        keep_lh = keep[: self.n_cortex]
        keep_rh = keep[self.n_cortex :]

        pred_out_lh = np.zeros(self.n_cortex, dtype=np.float32)
        pred_out_rh = np.zeros(self.n_cortex, dtype=np.float32)
        # Ensure pred>0.5 matches final mask.
        pred_out_lh[keep_lh] = 0.51 + 0.49 * score_lh[keep_lh]
        pred_out_rh[keep_rh] = 0.51 + 0.49 * score_rh[keep_rh]

        clustered_lh = self._label_connected_components_cortex(keep_lh)
        clustered_rh = self._label_connected_components_cortex(keep_rh)

        return {
            "pred_lh": pred_out_lh,
            "pred_rh": pred_out_rh,
            "clustered_lh": clustered_lh,
            "clustered_rh": clustered_rh,
        }


def _concat_csvs(out_base: Path, *, out_name: str) -> None:
    parts = []
    for fold in range(5):
        p = out_base / f"fold{fold}" / "val" / out_name
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "fold" not in df.columns:
            df.insert(1, "fold", int(fold))
        parts.append(df)
    if not parts:
        return
    out = pd.concat(parts, axis=0, ignore_index=True)
    out.to_csv(out_base / f"all_folds_{out_name}", index=False)


def _aggregate_and_save(*, name: str, fold_summaries: List[Dict], out_base: Path, meta: Dict) -> Dict:
    def mean_std(xs: List[float]) -> Tuple[float, float]:
        return float(np.mean(xs)) if xs else 0.0, float(np.std(xs)) if xs else 0.0

    metrics = {
        "vertex_dsc": [s["vertex_level"]["DSC_mean"] for s in fold_summaries],
        "vertex_auprc": [s["vertex_level"]["AUPRC"] for s in fold_summaries],
        "det_boxdsc": [s["subject_level"]["detection_rate_boxdsc"] for s in fold_summaries],
        "det_dist": [s["subject_level"]["detection_rate_distance"] for s in fold_summaries],
        "det_ppv50": [s["subject_level"].get("detection_rate_ppv50", 0.0) for s in fold_summaries],
        "pinpoint": [s["subject_level"]["pinpointing_rate"] for s in fold_summaries],
        "recall015": [s["subject_level"]["recall_at_0.15"] for s in fold_summaries],
        "recall020": [s["subject_level"]["recall_at_0.20"] for s in fold_summaries],
        "f1_boxdsc": [s["cluster_level"]["F1_boxdsc"] for s in fold_summaries],
        "f1_dist": [s["cluster_level"]["F1_distance"] for s in fold_summaries],
        "f1_ppv50": [s["cluster_level"].get("F1_ppv50", 0.0) for s in fold_summaries],
        "prec_boxdsc": [s["cluster_level"]["precision_boxdsc"] for s in fold_summaries],
        "prec_dist": [s["cluster_level"]["precision_distance"] for s in fold_summaries],
        "prec_ppv50": [s["cluster_level"].get("precision_ppv50", 0.0) for s in fold_summaries],
        "fp_per_patient": [s["cluster_level"]["FP_per_patient_mean"] for s in fold_summaries],
        "fp_ppv50_per_patient": [s["cluster_level"].get("FP_per_patient_ppv50_mean", 0.0) for s in fold_summaries],
    }

    agg = {
        "model_type": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_folds": len(fold_summaries),
        "total_subjects": int(sum(int(s.get("n_subjects", 0)) for s in fold_summaries)),
        "meta": meta,
        "vertex_level": {
            "DSC_mean": mean_std(metrics["vertex_dsc"])[0],
            "DSC_std": mean_std(metrics["vertex_dsc"])[1],
            "AUPRC_mean": mean_std(metrics["vertex_auprc"])[0],
            "AUPRC_std": mean_std(metrics["vertex_auprc"])[1],
        },
        "cluster_level": {
            "F1_boxdsc_mean": mean_std(metrics["f1_boxdsc"])[0],
            "F1_boxdsc_std": mean_std(metrics["f1_boxdsc"])[1],
            "F1_distance_mean": mean_std(metrics["f1_dist"])[0],
            "F1_distance_std": mean_std(metrics["f1_dist"])[1],
            "F1_ppv50_mean": mean_std(metrics["f1_ppv50"])[0],
            "F1_ppv50_std": mean_std(metrics["f1_ppv50"])[1],
            "precision_boxdsc_mean": mean_std(metrics["prec_boxdsc"])[0],
            "precision_boxdsc_std": mean_std(metrics["prec_boxdsc"])[1],
            "precision_distance_mean": mean_std(metrics["prec_dist"])[0],
            "precision_distance_std": mean_std(metrics["prec_dist"])[1],
            "precision_ppv50_mean": mean_std(metrics["prec_ppv50"])[0],
            "precision_ppv50_std": mean_std(metrics["prec_ppv50"])[1],
            "FP_per_patient_mean": mean_std(metrics["fp_per_patient"])[0],
            "FP_per_patient_std": mean_std(metrics["fp_per_patient"])[1],
            "FP_per_patient_ppv50_mean": mean_std(metrics["fp_ppv50_per_patient"])[0],
            "FP_per_patient_ppv50_std": mean_std(metrics["fp_ppv50_per_patient"])[1],
        },
        "subject_level": {
            "detection_rate_boxdsc_mean": mean_std(metrics["det_boxdsc"])[0],
            "detection_rate_boxdsc_std": mean_std(metrics["det_boxdsc"])[1],
            "detection_rate_distance_mean": mean_std(metrics["det_dist"])[0],
            "detection_rate_distance_std": mean_std(metrics["det_dist"])[1],
            "detection_rate_ppv50_mean": mean_std(metrics["det_ppv50"])[0],
            "detection_rate_ppv50_std": mean_std(metrics["det_ppv50"])[1],
            "pinpointing_rate_mean": mean_std(metrics["pinpoint"])[0],
            "pinpointing_rate_std": mean_std(metrics["pinpoint"])[1],
            "recall_at_0.15_mean": mean_std(metrics["recall015"])[0],
            "recall_at_0.20_mean": mean_std(metrics["recall020"])[0],
        },
        "per_fold": fold_summaries,
    }

    (out_base / "aggregate_val.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg


def _evaluate_fold(
    *,
    name: str,
    fold: int,
    v2_dir: Path,
    out_base: Path,
    lesion_root: Path,
    args: argparse.Namespace,
    splits: Dict[int, Dict[str, List[str]]],
    t1_pred_hdf5_by_fold: Dict[int, Path],
    laterality_csv_by_fold: Dict[int, Path],
) -> Dict:
    subject_ids = [str(s) for s in splits[int(fold)]["val_ids"]]
    gate_mode = str(getattr(args, "gate_mode", "threshold")).strip().lower()

    # Load laterality predictions (deploy-real; do NOT use oracle columns).
    fmri_hemi = _load_fmri_pred_hemi_by_sid(laterality_csv_by_fold[int(fold)])

    # Prepare I/O.
    out_dir = out_base / f"fold{fold}" / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_hdf5 = out_dir / "predictions_val.hdf5"

    t1_path = _resolve_path(t1_pred_hdf5_by_fold[int(fold)])
    if not t1_path.is_file():
        raise FileNotFoundError(f"Missing T1 predictions for fold {fold}: {t1_path}")

    pred_dir = v2_dir / f"fold_{fold:02d}" / "val_predictions"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing V2 val_predictions dir: {pred_dir}")

    post = V2DeployPostprocessor(
        area_max_cluster_cm2=float(args.area_max_cluster_cm2),
        area_max_total_cm2=float(args.area_max_total_cm2),
        k_clusters=int(args.k_clusters),
        min_vertices=int(args.min_vertices),
    )
    layout = NodeLayout()

    selector = None
    selector_thr = float("nan")
    selector_meta: Dict[str, object] = {}
    if gate_mode == "selector":
        selector, selector_thr, selector_meta = _train_gate_selector(
            fold=int(fold),
            v2_dir=v2_dir,
            args=args,
            post=post,
            layout=layout,
            splits=splits,
            t1_pred_hdf5_by_fold=t1_pred_hdf5_by_fold,
            laterality_csv_by_fold=laterality_csv_by_fold,
        )
        (out_dir / "selector_meta.json").write_text(json.dumps(selector_meta, indent=2), encoding="utf-8")

    gate_rows = []
    wrote: List[str] = []
    t1_used = 0
    fmri_used = 0

    with h5py.File(str(t1_path), "r") as f_t1, h5py.File(str(out_hdf5), "w") as f_out:
        for sid in subject_ids:
            if sid not in f_t1:
                logger.warning("Missing T1 subject in fold %d: %s (skip)", fold, sid)
                continue

            # ---- Load T1 ----
            pred_t1_lh = np.asarray(f_t1[sid]["lh"]["prediction"][:], dtype=np.float32).reshape(-1)
            pred_t1_rh = np.asarray(f_t1[sid]["rh"]["prediction"][:], dtype=np.float32).reshape(-1)

            if args.t1_cluster_key in f_t1[sid]["lh"]:
                cl_t1_lh = np.asarray(f_t1[sid]["lh"][args.t1_cluster_key][:], dtype=np.int32).reshape(-1)
                cl_t1_rh = np.asarray(f_t1[sid]["rh"][args.t1_cluster_key][:], dtype=np.int32).reshape(-1)
            else:
                cl_t1_lh = np.zeros_like(pred_t1_lh, dtype=np.int32)
                cl_t1_rh = np.zeros_like(pred_t1_rh, dtype=np.int32)

            t1_has_cluster = bool((cl_t1_lh > 0).any() or (cl_t1_rh > 0).any())
            t1_conf_lh = float(_top_cluster_mean_score(pred_t1_lh, cl_t1_lh))
            t1_conf_rh = float(_top_cluster_mean_score(pred_t1_rh, cl_t1_rh))
            t1_conf = float(max(t1_conf_lh, t1_conf_rh))
            t1_pred_hemi = "L" if t1_conf_lh >= t1_conf_rh else "R"
            t1_n_clusters = _count_clusters(cl_lh=cl_t1_lh, cl_rh=cl_t1_rh)
            t1_top_area_cm2 = 0.0
            t1_top_mask_lh = np.zeros_like(pred_t1_lh, dtype=bool)
            t1_top_mask_rh = np.zeros_like(pred_t1_rh, dtype=bool)
            if t1_has_cluster:
                t1_stats = _cluster_stats(
                    pred_lh=pred_t1_lh,
                    pred_rh=pred_t1_rh,
                    cl_lh=cl_t1_lh,
                    cl_rh=cl_t1_rh,
                    score_mode=str(args.cluster_sort),
                )
                if t1_stats:
                    t = t1_stats[0]
                    if str(t["hemi"]) == "lh":
                        t1_top_mask_lh = t["mask"]
                        t1_top_area_cm2 = float(np.sum(post.area_cortex_cm2[t1_top_mask_lh]))
                    else:
                        t1_top_mask_rh = t["mask"]
                        t1_top_area_cm2 = float(np.sum(post.area_cortex_cm2[t1_top_mask_rh]))

            # Base gate: switch to fMRI (deploy-real).
            # - threshold mode: classic TrackA (no cluster OR low confidence)
            # - selector mode: only force fMRI if T1 has no usable clusters; selector decides injection
            if gate_mode == "selector":
                use_fmri = not t1_has_cluster
                use_fmri_reason = "t1_no_cluster" if (not t1_has_cluster) else "selector"
            else:
                use_fmri = (not t1_has_cluster) or (t1_conf < float(args.t1_conf_threshold))
                use_fmri_reason = "t1_no_cluster" if (not t1_has_cluster) else "t1_low_conf"
            use_union_injection = False
            union_reason = ""

            # ---- Build fMRIhemi(V2) candidate (only if needed) ----
            fmri_available = False
            fmri_pred_hemi = ""
            prob_right = float("nan")
            lat_margin = float("nan")
            fmri_top_score = float("nan")
            fmri_minus_t1 = float("nan")
            fmri_n_clusters = 0
            fmri_top_area_cm2 = 0.0
            fmri_top_mask_lh = np.zeros_like(pred_t1_lh, dtype=bool)
            fmri_top_mask_rh = np.zeros_like(pred_t1_rh, dtype=bool)
            pred_v2_lh = pred_v2_rh = None
            cl_v2_lh = cl_v2_rh = None
            selector_prob = float("nan")

            if use_fmri or bool(args.enable_injection) or gate_mode == "selector":
                if sid not in fmri_hemi:
                    logger.warning("Missing fMRI laterality for fold %d: %s (fallback to T1)", fold, sid)
                    use_fmri = False
                    use_fmri_reason = "missing_laterality"
                else:
                    p_npz = pred_dir / f"{sid}.npz"
                    if not p_npz.is_file():
                        logger.warning("Missing V2 val_prediction: %s (fallback to T1)", str(p_npz))
                        use_fmri = False
                        use_fmri_reason = "missing_v2_npz"
                    else:
                        fmri_pred_hemi = str(fmri_hemi[sid]["pred_hemi"])
                        prob_right = float(fmri_hemi[sid]["prob_right"])
                        lat_margin = float(abs(prob_right - 0.5))

                        probs = np.asarray(np.load(str(p_npz))["p"], dtype=np.float32).reshape(-1)
                        p_t = _transform_probs(probs, str(args.prob_transform))
                        ipsi = p_t[layout.idx_ipsi_cortex]
                        contra = p_t[layout.idx_contra_cortex]
                        if fmri_pred_hemi == "L":
                            left_parc, right_parc = ipsi, contra
                        else:
                            left_parc, right_parc = contra, ipsi

                        res = post.postprocess(
                            left_parc=left_parc,
                            right_parc=right_parc,
                            area_target_cm2=float(args.area_target_cm2),
                            cluster_sort=str(args.cluster_sort),
                        )
                        pred_v2_lh = res["pred_lh"].astype(np.float32)
                        pred_v2_rh = res["pred_rh"].astype(np.float32)
                        cl_v2_lh = res["clustered_lh"].astype(np.int32)
                        cl_v2_rh = res["clustered_rh"].astype(np.int32)
                        fmri_available = bool((cl_v2_lh > 0).any() or (cl_v2_rh > 0).any())
                        fmri_n_clusters = _count_clusters(cl_lh=cl_v2_lh, cl_rh=cl_v2_rh)
                        if fmri_available:
                            stats = _cluster_stats(
                                pred_lh=pred_v2_lh,
                                pred_rh=pred_v2_rh,
                                cl_lh=cl_v2_lh,
                                cl_rh=cl_v2_rh,
                                score_mode=str(args.cluster_sort),
                            )
                            fmri_top_score = float(stats[0]["score"]) if stats else float("nan")
                            fmri_minus_t1 = float(fmri_top_score - t1_conf) if np.isfinite(fmri_top_score) else float("nan")
                            if stats:
                                top = stats[0]
                                if str(top["hemi"]) == "lh":
                                    fmri_top_mask_lh = top["mask"]
                                    fmri_top_area_cm2 = float(np.sum(post.area_cortex_cm2[fmri_top_mask_lh]))
                                else:
                                    fmri_top_mask_rh = top["mask"]
                                    fmri_top_area_cm2 = float(np.sum(post.area_cortex_cm2[fmri_top_mask_rh]))

                        if not fmri_available:
                            # Do not switch to empty fMRI output.
                            use_fmri = False
                            use_fmri_reason = "fmri_empty"
                        elif (not use_fmri) and bool(args.enable_injection) and gate_mode != "selector":
                            # Injection mode: keep T1, but add limited fMRI clusters when deploy-real signals suggest
                            # fMRI is worth considering (no oracle).
                            conflict = (t1_pred_hemi != fmri_pred_hemi) and (lat_margin >= float(args.inject_conflict_margin_threshold))
                            diff = float(fmri_minus_t1) if np.isfinite(fmri_minus_t1) else -1e9
                            high_thr = float(args.inject_diff_threshold)
                            low_thr = float(getattr(args, "inject_diff_low_threshold", -1.0))
                            low_conf_max = float(getattr(args, "inject_lowdiff_t1conf_max", 1.0))
                            low_require_same = bool(getattr(args, "inject_lowdiff_require_same_hemi", False))
                            low_enabled = (low_thr > 0) and (low_thr < high_thr)

                            if conflict or (diff >= high_thr):
                                use_union_injection = True
                                if conflict and (diff >= high_thr):
                                    union_reason = "inject_conflict_and_diff"
                                elif conflict:
                                    union_reason = "inject_conflict"
                                else:
                                    union_reason = "inject_diff"
                            elif low_enabled:
                                same_hemi = t1_pred_hemi == fmri_pred_hemi
                                if (diff >= low_thr) and (t1_conf <= low_conf_max + 1e-12) and ((not low_require_same) or same_hemi):
                                    use_union_injection = True
                                    union_reason = "inject_lowdiff"
                        elif (not use_fmri) and gate_mode == "selector" and selector is not None and fmri_available:
                            hemi_conflict = (t1_pred_hemi != fmri_pred_hemi) if fmri_pred_hemi in {"L", "R"} else False
                            inter = float(
                                np.sum(post.area_cortex_cm2[t1_top_mask_lh & fmri_top_mask_lh])
                                + np.sum(post.area_cortex_cm2[t1_top_mask_rh & fmri_top_mask_rh])
                            )
                            denom = float(t1_top_area_cm2 + fmri_top_area_cm2)
                            t1_fmri_top_dice = float((2.0 * inter / denom) if denom > 0 else 0.0)
                            min_area = float(min(t1_top_area_cm2, fmri_top_area_cm2))
                            t1_fmri_top_minfrac = float((inter / min_area) if min_area > 0 else 0.0)
                            feat = _selector_feature_vector(
                                t1_conf=t1_conf,
                                t1_conf_lh=t1_conf_lh,
                                t1_conf_rh=t1_conf_rh,
                                t1_has_cluster=t1_has_cluster,
                                t1_n_clusters=t1_n_clusters,
                                fmri_available=fmri_available,
                                fmri_top_score=fmri_top_score,
                                fmri_minus_t1=fmri_minus_t1,
                                lat_margin=lat_margin,
                                hemi_conflict=hemi_conflict,
                                fmri_n_clusters=fmri_n_clusters,
                                t1_top_area_cm2=t1_top_area_cm2,
                                fmri_top_area_cm2=fmri_top_area_cm2,
                                t1_fmri_top_dice=t1_fmri_top_dice,
                                t1_fmri_top_minfrac=t1_fmri_top_minfrac,
                            ).reshape(1, -1)
                            selector_prob = float(selector.predict_proba(feat)[0, 1])
                            if np.isfinite(selector_prob) and (selector_prob >= float(selector_thr)):
                                use_union_injection = True
                                union_reason = "selector_inject"

            # ---- Write fused ----
            union_payload = None
            if use_union_injection:
                union_payload = _apply_union_fusion(
                    pred_t1_lh=pred_t1_lh,
                    pred_t1_rh=pred_t1_rh,
                    cl_t1_lh=cl_t1_lh,
                    cl_t1_rh=cl_t1_rh,
                    pred_fmri_lh=pred_v2_lh,
                    pred_fmri_rh=pred_v2_rh,
                    cl_fmri_lh=cl_v2_lh,
                    cl_fmri_rh=cl_v2_rh,
                    k_total=int(args.k_clusters),
                    k_fmri=int(args.inject_k_fmri),
                    cluster_sort=str(args.cluster_sort),
                    label_components_fn=post._label_connected_components_cortex,
                )

            g = f_out.create_group(sid)
            for hemi_name in ("lh", "rh"):
                h = g.create_group(hemi_name)
                if use_union_injection:
                    pred_u_lh, pred_u_rh, cl_u_lh, cl_u_rh = union_payload
                    pred = pred_u_lh if hemi_name == "lh" else pred_u_rh
                    cl = cl_u_lh if hemi_name == "lh" else cl_u_rh
                elif use_fmri:
                    pred = pred_v2_lh if hemi_name == "lh" else pred_v2_rh
                    cl = cl_v2_lh if hemi_name == "lh" else cl_v2_rh
                else:
                    pred = pred_t1_lh if hemi_name == "lh" else pred_t1_rh
                    cl = cl_t1_lh if hemi_name == "lh" else cl_t1_rh
                h.create_dataset("prediction", data=pred.astype(np.float32), dtype=np.float32, compression="gzip")
                h.create_dataset("prediction_clustered", data=cl.astype(np.int32), dtype=np.int32, compression="gzip")

            wrote.append(sid)
            if use_union_injection:
                # Count injection under fMRI usage as well (fMRI contributes candidates).
                fmri_used += 1
            elif use_fmri:
                fmri_used += 1
            else:
                t1_used += 1

            gate_rows.append(
                {
                    "subject_id": sid,
                    "fold": int(fold),
                    "source": "T1+fMRI" if use_union_injection else ("fMRI" if use_fmri else "T1"),
                    "use_fmri_reason": str(union_reason if use_union_injection else use_fmri_reason),
                    "t1_conf": float(t1_conf),
                    "t1_conf_lh": float(t1_conf_lh),
                    "t1_conf_rh": float(t1_conf_rh),
                    "t1_pred_hemi": str(t1_pred_hemi),
                    "t1_has_cluster": bool(t1_has_cluster),
                    "t1_n_clusters": int(t1_n_clusters),
                    "t1_conf_threshold": float(args.t1_conf_threshold),
                    "gate_mode": str(gate_mode),
                    "fmri_available": bool(fmri_available),
                    "fmri_pred_hemi": str(fmri_pred_hemi),
                    "prob_right": float(prob_right) if np.isfinite(prob_right) else np.nan,
                    "lat_margin": float(lat_margin) if np.isfinite(lat_margin) else np.nan,
                    "fmri_top_score": float(fmri_top_score) if np.isfinite(fmri_top_score) else np.nan,
                    "fmri_minus_t1": float(fmri_minus_t1) if np.isfinite(fmri_minus_t1) else np.nan,
                    "fmri_n_clusters": int(fmri_n_clusters),
                    "selector_prob": float(selector_prob) if np.isfinite(selector_prob) else np.nan,
                    "selector_threshold": float(selector_thr) if np.isfinite(selector_thr) else np.nan,
                    "area_target_cm2": float(args.area_target_cm2),
                    "prob_transform": str(args.prob_transform),
                    "cluster_sort": str(args.cluster_sort),
                }
            )

    pd.DataFrame(gate_rows).to_csv(out_dir / "gate_decisions.csv", index=False)
    (out_dir / "deploy_config.json").write_text(
        json.dumps(
            {
                "gate": {
                    "t1_conf_feature": "top_cluster_mean",
                    "t1_conf_threshold": float(args.t1_conf_threshold),
                    "use_fmri_if_no_t1_cluster": True,
                    "gate_mode": str(gate_mode),
                },
                "selector": selector_meta if selector_meta else None,
                "fmrihemi_v2": {
                    "v2_dir": str(v2_dir),
                    "laterality_csv": str(_laterality_csv_for_fold(int(fold))),
                    "area_target_cm2": float(args.area_target_cm2),
                    "prob_transform": str(args.prob_transform),
                    "cluster_sort": str(args.cluster_sort),
                },
                "budgets": {
                    "AREA_MAX_CLUSTER_CM2": float(args.area_max_cluster_cm2),
                    "AREA_MAX_TOTAL_CM2": float(args.area_max_total_cm2),
                    "K_CLUSTERS": int(args.k_clusters),
                    "MIN_VERTICES": int(args.min_vertices),
                },
                "t1_inputs": {
                    "t1_predictions_hdf5": str(t1_path),
                    "t1_cluster_key": str(args.t1_cluster_key),
                },
                "write_stats": {"T1": int(t1_used), "fMRI": int(fmri_used), "n_subjects": int(len(wrote))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cohort = MeldCohort(hdf5_file_root="{site_code}_{group}_featurematrix_combat.hdf5", dataset=None)
    evaluator = ThreeLevelEvaluator(
        predictions_hdf5=out_hdf5,
        cohort=cohort,
        subject_ids=wrote,
        lesion_file_root=lesion_root,
        lesion_feature="lesion_main",
        split="val",
        model_name=f"{name}_fold{fold}",
    )
    summary = evaluator.evaluate_all(verbose=True)
    evaluator.save_results(out_dir)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy-real TrackA: T1 gate + fmrihemi(V2) takeover (no leakage).")
    ap.add_argument("--name", default="trackA_v2_fmrihemi_takeover", help="Output label/name")
    ap.add_argument("--v2_dir", default="meld_data/models/v2_gat3_diceproxy", help="V2 model dir (has fold_00/.../val_predictions)")
    ap.add_argument("--out_base", default="", help="Output base dir (default auto under meld_data/output/three_level_eval/)")
    ap.add_argument("--lesion_root", default=LESION_ROOT_DEFAULT, help="Derived lesion label root (fsaverage_sym_xhemi)")

    # Fold-aware inputs (override for public end-to-end use)
    ap.add_argument(
        "--t1_pred_hdf5_by_fold_json",
        default="",
        help="Optional JSON mapping fold->T1 predictions HDF5 path (out-of-fold predictions_val.hdf5).",
    )
    ap.add_argument(
        "--t1_pred_hdf5_template",
        default="",
        help="Optional template for fold->T1 predictions HDF5 path. Use {fold} or {fold:02d}.",
    )
    ap.add_argument(
        "--split_json_by_fold_json",
        default="",
        help="Optional JSON mapping fold->split json path (data_parameters.json with train_ids/val_ids).",
    )
    ap.add_argument(
        "--split_json_template",
        default="",
        help="Optional template for fold->split json path. Use {fold} or {fold:02d}.",
    )
    ap.add_argument(
        "--laterality_csv_by_fold_json",
        default="",
        help="Optional JSON mapping fold->laterality CSV path (fMRI-only; columns: subject_id, prob_right, pred_hemi).",
    )
    ap.add_argument(
        "--laterality_csv_template",
        default="",
        help="Optional template for fold->laterality CSV path. Use {fold} or {fold:02d}.",
    )

    # Gate
    ap.add_argument("--gate_mode", choices=["threshold", "selector"], default="threshold", help="Gating mode. 'selector' trains a per-fold train-only classifier (no leakage) to decide fMRI injection.")
    ap.add_argument("--t1_conf_threshold", type=float, default=0.5773, help="Fixed T1 confidence threshold (top_cluster_mean).")
    ap.add_argument("--t1_cluster_key", default="prediction_clustered", help="T1 clustered key to reuse from predictions HDF5.")
    ap.add_argument("--enable_injection", action="store_true", help="Enable union-style fMRI injection when T1 is high-confidence but fMRI signals suggest benefit (deploy-real).")
    ap.add_argument("--inject_diff_threshold", type=float, default=0.15, help="Inject if (fmri_top_score - t1_conf) >= this.")
    ap.add_argument("--inject_diff_low_threshold", type=float, default=-1.0, help="Optional lower diff threshold to use only with additional guards (disabled by default).")
    ap.add_argument("--inject_lowdiff_t1conf_max", type=float, default=1.0, help="Only allow low-diff injection when t1_conf <= this.")
    ap.add_argument("--inject_lowdiff_require_same_hemi", action="store_true", help="Only allow low-diff injection when T1 hemi == fMRI hemi.")
    ap.add_argument("--inject_conflict_margin_threshold", type=float, default=0.25, help="Inject if (T1 hemi != fMRI hemi) and laterality margin >= this.")
    ap.add_argument("--inject_k_fmri", type=int, default=2, help="How many fMRI clusters to include in union injection (0-2 recommended).")
    ap.add_argument("--selector_label", choices=["boxdsc", "ppv50"], default="boxdsc", help="Selector training label endpoint (train-only, GT-derived).")
    ap.add_argument("--selector_t1_subject_level_csv", default=DEFAULT_SELECTOR_T1_SUBJECT_CSV, help="T1 baseline subject-level CSV used to define selector labels (must be out-of-fold).")
    ap.add_argument("--selector_fmri_subject_level_csv", default=DEFAULT_SELECTOR_FMRI_SUBJECT_CSV, help="fMRIhemi(V2) subject-level CSV used to define selector labels (must be out-of-fold).")

    # V2 deploy postprocess config (fixed; do NOT use outer-val tuned best_config here)
    ap.add_argument("--area_target_cm2", type=float, default=60.0, help="V2 area-target parcel selection threshold (cm^2).")
    ap.add_argument("--prob_transform", default="raw", choices=["raw", "centered", "rank"])
    ap.add_argument("--cluster_sort", default="mean", choices=["mean", "max"])

    # Budgets (deployment)
    ap.add_argument("--area_max_cluster_cm2", type=float, default=30.0)
    ap.add_argument("--area_max_total_cm2", type=float, default=80.0)
    ap.add_argument("--k_clusters", type=int, default=3)
    ap.add_argument("--min_vertices", type=int, default=100)

    ap.add_argument("--fold", default="all", help="Fold index (0-4) or 'all'")
    args = ap.parse_args()

    name = str(args.name)
    v2_dir = _resolve_path(args.v2_dir)
    lesion_root = _resolve_path(args.lesion_root)

    if not v2_dir.is_dir():
        raise FileNotFoundError(f"V2 dir not found: {v2_dir}")
    if not lesion_root.is_dir():
        raise FileNotFoundError(f"Lesion root not found: {lesion_root}")

    t1_pred_hdf5_by_fold = _build_fold_path_map(
        label="t1_pred_hdf5",
        json_path=str(getattr(args, "t1_pred_hdf5_by_fold_json", "")),
        template=str(getattr(args, "t1_pred_hdf5_template", "")),
        default_map=T1_PRED_HDF5_BY_FOLD,
    )
    split_json_by_fold = _build_fold_path_map(
        label="split_json",
        json_path=str(getattr(args, "split_json_by_fold_json", "")),
        template=str(getattr(args, "split_json_template", "")),
        default_map=SPLIT_JSONS,
    )
    default_laterality = {int(f): _laterality_csv_for_fold(int(f)) for f in range(5)}
    laterality_csv_by_fold = _build_fold_path_map(
        label="laterality_csv",
        json_path=str(getattr(args, "laterality_csv_by_fold_json", "")),
        template=str(getattr(args, "laterality_csv_template", "")),
        default_map=default_laterality,
    )
    splits = _load_fold_splits(split_json_by_fold)

    if args.out_base:
        out_base = _resolve_path(args.out_base)
    else:
        suffix = f"_thr{args.t1_conf_threshold:.4f}_area{args.area_target_cm2:.0f}_a{args.area_max_cluster_cm2:.0f}_t{args.area_max_total_cm2:.0f}"
        out_base = _resolve_path(f"meld_data/output/three_level_eval/{name}{suffix}")
    out_base.mkdir(parents=True, exist_ok=True)

    folds = list(range(5)) if str(args.fold) == "all" else [int(args.fold)]
    fold_summaries: List[Dict] = []
    for fold in folds:
        fold_summaries.append(
            _evaluate_fold(
                name=name,
                fold=int(fold),
                v2_dir=v2_dir,
                out_base=out_base,
                lesion_root=lesion_root,
                args=args,
                splits=splits,
                t1_pred_hdf5_by_fold=t1_pred_hdf5_by_fold,
                laterality_csv_by_fold=laterality_csv_by_fold,
            )
        )

    if len(folds) == 5:
        _concat_csvs(out_base, out_name="subject_level_results.csv")
        _concat_csvs(out_base, out_name="cluster_level_results.csv")
        _concat_csvs(out_base, out_name="vertex_level_results.csv")
        _concat_csvs(out_base, out_name="gate_decisions.csv")
        meta = {
            "gate": {"t1_conf_feature": "top_cluster_mean", "t1_conf_threshold": float(args.t1_conf_threshold)},
            "injection": {
                "enabled": bool(args.enable_injection),
                "diff_threshold": float(args.inject_diff_threshold),
                "conflict_margin_threshold": float(args.inject_conflict_margin_threshold),
                "k_fmri": int(args.inject_k_fmri),
                "k_total": int(args.k_clusters),
            },
            "fmrihemi_v2": {
                "v2_dir": str(v2_dir),
                "area_target_cm2": float(args.area_target_cm2),
                "prob_transform": str(args.prob_transform),
                "cluster_sort": str(args.cluster_sort),
                "laterality_csv_by_fold": {f: str(laterality_csv_by_fold[int(f)]) for f in range(5)},
            },
            "budgets": {
                "AREA_MAX_CLUSTER_CM2": float(args.area_max_cluster_cm2),
                "AREA_MAX_TOTAL_CM2": float(args.area_max_total_cm2),
                "K_CLUSTERS": int(args.k_clusters),
                "MIN_VERTICES": int(args.min_vertices),
            },
            "t1_inputs": {"t1_cluster_key": str(args.t1_cluster_key), "t1_pred_hdf5_by_fold": {f: str(t1_pred_hdf5_by_fold[int(f)]) for f in range(5)}},
            "splits": {"split_json_by_fold": {f: str(split_json_by_fold[int(f)]) for f in range(5)}},
            "no_leakage": True,
        }
        agg = _aggregate_and_save(name=name, fold_summaries=fold_summaries, out_base=out_base, meta=meta)
        logger.info("Wrote aggregate: %s", str(out_base / "aggregate_val.json"))
        logger.info("  Det(boxDSC) mean=%.4f", float(agg["subject_level"]["detection_rate_boxdsc_mean"]))
        logger.info("  Det(PPV50)  mean=%.4f", float(agg["subject_level"]["detection_rate_ppv50_mean"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
