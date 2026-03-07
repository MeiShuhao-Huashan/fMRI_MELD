#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from meld_fmri.fmri_gcn.model import (
    DeepEZConfig,
    DeepEZDualExpertGCN,
    DeepEZDualExpertGAT,
    DeepEZGAT,
    DeepEZGATFiLM,
    DeepEZGATFiLMHemiSSDB,
    DeepEZGATFiLMSubGate,
    DeepEZGCN,
    DeepEZGCNFiLM,
    DeepEZGCNFiLMHemiSSDB,
    DeepEZGCNFiLMSubGate,
)


def _configure_logging(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fmri_gcn.train")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "train_deepez_gcn.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _load_split_ids(split_json: Path) -> Tuple[List[str], List[str]]:
    with open(split_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    train_ids = [str(x) for x in d.get("train_ids", [])]
    val_ids = [str(x) for x in d.get("val_ids", [])]
    if not train_ids or not val_ids:
        raise ValueError(f"split_json missing train_ids/val_ids: {split_json}")
    return train_ids, val_ids


def _default_split_json_for_fold(project_dir: Path, fold: int) -> Optional[Path]:
    # Use the T1-only anchor runs as default split definitions for fold0/fold4.
    candidates = [
        project_dir / f"meld_data/models/25-12-16_MELD_fMRI_fold{fold}/fold_{fold:02d}/data_parameters.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_cache(cache_dir: Path, subject_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    p = cache_dir / f"{subject_id}.npz"
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    L_loc = z["L_loc"].astype(np.float32) if "L_loc" in z.files else None
    y = z["y"].astype(np.int64)
    meta = {
        "hemi": str(z.get("hemi", "")),
        "t": int(z.get("t", 0)),
        "path": str(p),
    }
    return X, y, L_loc, meta


def _compute_class_weights(y_all: np.ndarray) -> torch.Tensor:
    y = y_all.reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos <= 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    w1 = float(n_neg) / float(n_pos)
    return torch.tensor([1.0, w1], dtype=torch.float32)


def _compute_sample_per_class(y_all: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    y = y_all.reshape(-1).astype(np.int64)
    counts = [int(np.sum(y == c)) for c in range(int(n_classes))]
    return torch.tensor(counts, dtype=torch.float32)


def _balanced_softmax_loss(logits: torch.Tensor, y: torch.Tensor, sample_per_class: torch.Tensor) -> torch.Tensor:
    """
    Balanced Softmax = CrossEntropy(logits + log(n_c)).
    Ref: Balanced Meta-Softmax / BalancedSoftmax.
    """
    spc = sample_per_class.to(dtype=logits.dtype, device=logits.device).clamp(min=1.0)
    logits_adj = logits + spc.log().view(1, -1)
    return F.cross_entropy(logits_adj, y)


def _robust_balanced_softmax_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    sample_per_class: torch.Tensor,
    beta: float,
    n_classes: int = 2,
) -> torch.Tensor:
    """
    Robust Balanced Softmax = BalancedSoftmax + beta * RCE (Reverse Cross Entropy) term.
    Ref: IceBerg's RobustBalancedSoftmax (used for pseudo-label training).
    """
    loss = _balanced_softmax_loss(logits, y, sample_per_class)
    if beta <= 0:
        return loss

    spc = sample_per_class.to(dtype=logits.dtype, device=logits.device).clamp(min=1.0)
    logits_adj = logits + spc.log().view(1, -1)
    pred = F.softmax(logits_adj, dim=1).clamp(min=1e-7, max=1.0)
    label_one_hot = F.one_hot(y, int(n_classes)).to(dtype=pred.dtype, device=pred.device).clamp(min=1e-4, max=1.0)
    rce = (-1.0 * torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
    return loss + float(beta) * rce


@dataclass(frozen=True)
class Metrics:
    loss: float
    auc: float
    ap: float
    prevalence: float
    ap_random: float
    ap_lift_ratio: float
    ap_lift_diff: float
    macro_auc_valid: float
    macro_ap_valid: float
    macro_auc_all: float
    macro_ap_all: float
    hit_k: int
    hit_rate_valid: float
    hit_rate_all: float


def _eval_epoch(
    model: nn.Module,
    adj: torch.Tensor,
    samples: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]],
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hit_k: int,
    use_local: bool,
) -> Metrics:
    model.eval()
    losses: List[float] = []
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    sub_aucs: List[float] = []
    sub_aps: List[float] = []
    sub_hits: List[float] = []
    sub_has_pos: List[bool] = []
    sub_has_both: List[bool] = []
    with torch.no_grad():
        for X_np, y_np, L_np, _sid in samples:
            X = torch.from_numpy(X_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            if use_local:
                if L_np is None:
                    raise ValueError(f"Missing L_loc for subject={_sid} but use_local=True")
                L = torch.from_numpy(L_np).to(device)
                logits, _bias = model(X, adj, L)
            else:
                logits, _bias = model(X, adj)
            loss = loss_fn(logits, y)
            losses.append(float(loss.detach().cpu().item()))
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            ys.append(y_np.astype(np.int64))
            ps.append(p.astype(np.float64))

            y_sub = y_np.astype(np.int64)
            p_sub = p.astype(np.float64)
            n_pos = int(np.sum(y_sub == 1))
            n_neg = int(np.sum(y_sub == 0))
            sub_has_pos.append(n_pos > 0)
            sub_has_both.append(n_pos > 0 and n_neg > 0)

            try:
                auc_sub = float(roc_auc_score(y_sub, p_sub)) if (n_pos > 0 and n_neg > 0) else float("nan")
            except Exception:
                auc_sub = float("nan")
            try:
                ap_sub = float(average_precision_score(y_sub, p_sub)) if (n_pos > 0) else float("nan")
            except Exception:
                ap_sub = float("nan")
            sub_aucs.append(auc_sub)
            sub_aps.append(ap_sub)

            k = int(hit_k)
            if k <= 0:
                k = 1
            k = min(k, int(y_sub.shape[0]))
            topk = np.argsort(p_sub)[-k:]
            sub_hits.append(float(np.any(y_sub[topk] == 1)))

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    n_pos_all = int(np.sum(y_all == 1))
    n_all = int(y_all.shape[0])
    prevalence = float(n_pos_all) / float(n_all) if n_all > 0 else float("nan")
    # Guard: if all labels are the same, roc_auc is undefined.
    try:
        auc = float(roc_auc_score(y_all, p_all))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_all, p_all))
    except Exception:
        ap = float("nan")

    ap_random = prevalence
    ap_lift_ratio = float(ap) / float(prevalence) if (not np.isnan(ap) and prevalence > 0) else float("nan")
    ap_lift_diff = float(ap) - float(prevalence) if (not np.isnan(ap) and not np.isnan(prevalence)) else float("nan")

    sub_aucs_np = np.asarray(sub_aucs, dtype=np.float64)
    sub_aps_np = np.asarray(sub_aps, dtype=np.float64)
    sub_hits_np = np.asarray(sub_hits, dtype=np.float64)
    has_pos_np = np.asarray(sub_has_pos, dtype=bool)
    has_both_np = np.asarray(sub_has_both, dtype=bool)

    macro_auc_valid = float(np.nanmean(sub_aucs_np)) if np.any(has_both_np) else float("nan")
    macro_ap_valid = float(np.nanmean(sub_aps_np)) if np.any(has_pos_np) else float("nan")
    macro_auc_all = float(np.mean(np.nan_to_num(sub_aucs_np, nan=0.5))) if sub_aucs_np.size else float("nan")
    macro_ap_all = float(np.mean(np.nan_to_num(sub_aps_np, nan=0.0))) if sub_aps_np.size else float("nan")

    hit_k_eff = int(hit_k) if int(hit_k) > 0 else 1
    hit_rate_valid = float(np.mean(sub_hits_np[has_pos_np])) if np.any(has_pos_np) else float("nan")
    hit_rate_all = float(np.mean(sub_hits_np)) if sub_hits_np.size else float("nan")

    return Metrics(
        loss=float(np.mean(losses)) if losses else float("nan"),
        auc=auc,
        ap=ap,
        prevalence=prevalence,
        ap_random=ap_random,
        ap_lift_ratio=ap_lift_ratio,
        ap_lift_diff=ap_lift_diff,
        macro_auc_valid=macro_auc_valid,
        macro_ap_valid=macro_ap_valid,
        macro_auc_all=macro_auc_all,
        macro_ap_all=macro_ap_all,
        hit_k=hit_k_eff,
        hit_rate_valid=hit_rate_valid,
        hit_rate_all=hit_rate_all,
    )


def _eval_per_subject(
    model: nn.Module,
    adj: torch.Tensor,
    samples: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]],
    device: torch.device,
    hit_k: int,
    use_local: bool,
    out_dir: Optional[Path] = None,
    pred_subdir: str = "val_predictions",
) -> pd.DataFrame:
    """
    Evaluate per-subject AUC/AP and optionally save per-subject prediction arrays.
    """
    model.eval()
    rows: List[Dict[str, object]] = []
    pred_dir: Optional[Path] = None
    if out_dir is not None:
        pred_dir = out_dir / str(pred_subdir)
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for X_np, y_np, L_np, sid in samples:
            X = torch.from_numpy(X_np).to(device)
            y = y_np.astype(np.int64)
            if use_local:
                if L_np is None:
                    raise ValueError(f"Missing L_loc for subject={sid} but use_local=True")
                L = torch.from_numpy(L_np).to(device)
                logits, _bias = model(X, adj, L)
            else:
                logits, _bias = model(X, adj)
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)

            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            try:
                auc = float(roc_auc_score(y, p)) if (n_pos > 0 and n_neg > 0) else float("nan")
            except Exception:
                auc = float("nan")
            try:
                ap = float(average_precision_score(y, p)) if (n_pos > 0) else float("nan")
            except Exception:
                ap = float("nan")

            k = int(hit_k) if int(hit_k) > 0 else 1
            k = min(k, int(y.shape[0]))
            topk = np.argsort(p.astype(np.float64))[-k:]
            hit_at_k = float(np.any(y[topk] == 1)) if y.size else float("nan")
            prevalence = float(n_pos) / float(y.shape[0]) if y.size else float("nan")
            ap_lift_ratio = float(ap) / float(prevalence) if (not np.isnan(ap) and prevalence > 0) else float("nan")
            ap_lift_diff = float(ap) - float(prevalence) if (not np.isnan(ap) and not np.isnan(prevalence)) else float("nan")

            if pred_dir is not None:
                np.savez_compressed(pred_dir / f"{sid}.npz", p=p, y=y.astype(np.uint8))

            rows.append(
                {
                    "subject_id": sid,
                    "n_nodes": int(y.shape[0]),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "prevalence": prevalence,
                    "auc": auc,
                    "ap": ap,
                    "ap_random": prevalence,
                    "ap_lift_ratio": ap_lift_ratio,
                    "ap_lift_diff": ap_lift_diff,
                    "hit_k": int(k),
                    "hit_at_k": hit_at_k,
                }
            )

    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train DeepEZ-style GCN on precomputed FC caches (fMRI_GCN branch).")
    ap.add_argument("--cache_root", default="meld_data/derivatives/fmri_gcn/aparc_v0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split_json", default=None, help="Path to a data_parameters.json containing train_ids/val_ids.")
    ap.add_argument("--arch", default="gcn", choices=["gcn", "gat"], help="Model architecture (gcn or gat).")
    ap.add_argument(
        "--fusion",
        default="none",
        choices=["none", "film", "film_subgate", "film_hemi_ssdb", "dual_expert"],
        help="Fusion mode: none (FC only), film (FiLM local residual), film_subgate (FiLM + subject gate), film_hemi_ssdb (FiLM + hemi SSDB), dual_expert (FC+local experts).",
    )
    ap.add_argument("--init_from", default=None, help="Optional path to a checkpoint (best_model.pt) or model dir to initialise from (S4 baseline).")
    ap.add_argument("--freeze_trunk", action="store_true", help="Freeze trunk layers (gc1/gc2 or gat1/gat2) during training.")
    ap.add_argument(
        "--freeze_classifier_epochs",
        type=int,
        default=80,
        help="If >0, freeze fc_class for the first N epochs (FiLM fusion only).",
    )
    ap.add_argument("--film_z", type=int, default=32, help="Local-branch latent dim for FiLM (default 32).")
    ap.add_argument("--film_gate_bias", type=float, default=-2.0, help="Initial bias for the FiLM gate sigmoid (default -2).")
    ap.add_argument("--film_cov_power", type=float, default=1.0, help="Multiply gate by coverage**power (default 1).")
    ap.add_argument("--subgate_bias", type=float, default=-2.0, help="Init bias for subject gate (film_subgate).")
    ap.add_argument("--subgate_floor", type=float, default=0.0, help="Lower bound for subject gate (film_subgate).")
    ap.add_argument("--hemi_ssdb_n_mid", type=int, default=None, help="Override n_mid for hemi SSDB (film_hemi_ssdb).")
    ap.add_argument("--dual_gate_hidden", type=int, default=16, help="Hidden dim for dual_expert subject gate MLP.")
    ap.add_argument("--dual_gate_bias", type=float, default=-2.0, help="Init bias for dual_expert subject gate.")
    ap.add_argument("--dual_gate_floor", type=float, default=0.0, help="Lower bound for dual_expert subject gate w_sub.")
    ap.add_argument(
        "--dual_use_ssdb1_stats",
        action="store_true",
        help="For dual_expert: include ssdb_fc1 hidden energy stats from both experts in the subject gate input.",
    )
    ap.add_argument("--dual_cov_power", type=float, default=1.0, help="Coverage power for dual_expert node-wise attenuation.")
    ap.add_argument(
        "--dual_bias_mode",
        default="shared",
        choices=["shared", "none", "mean_gnode", "square", "mlp"],
        help="For dual_expert: how to scale the local expert SSDB bias term. shared=w (default); none=0; mean_gnode=w*mean(g_node); square=w^2; mlp=independent bias gate MLP.",
    )
    ap.add_argument(
        "--dual_bias_gate_hidden",
        type=int,
        default=16,
        help="For dual_expert+bias_mode=mlp: hidden dim for the bias gate MLP (default 16).",
    )
    ap.add_argument(
        "--dual_bias_gate_bias",
        type=float,
        default=-4.0,
        help="For dual_expert+bias_mode=mlp: init bias for bias gate sigmoid (default -4.0).",
    )
    ap.add_argument(
        "--dual_bias_gate_floor",
        type=float,
        default=0.0,
        help="For dual_expert+bias_mode=mlp: lower bound for w_bias (default 0).",
    )
    ap.add_argument(
        "--dual_node_gate",
        default="coverage",
        choices=["coverage", "learned"],
        help="For dual_expert: node gate mode. coverage = g_node=coverage^p (default); learned = sigmoid(linear(local))*coverage^p.",
    )
    ap.add_argument("--dual_node_gate_bias", type=float, default=2.0, help="Init bias for dual_expert learned node-gate sigmoid.")
    ap.add_argument("--dual_node_gate_floor", type=float, default=0.0, help="Lower bound for dual_expert learned node-gate sigmoid.")
    ap.add_argument("--init_from_fc", default=None, help="For dual_expert: init FC expert from checkpoint or model dir.")
    ap.add_argument("--init_from_loc", default=None, help="For dual_expert: init local expert from checkpoint or model dir.")
    ap.add_argument("--dual_train_experts", action="store_true", help="For dual_expert: train experts (default: freeze experts, train gate only).")
    ap.add_argument("--dual_freeze_fc_expert", action="store_true", help="For dual_expert: freeze FC expert params (even if --dual_train_experts).")
    ap.add_argument("--dual_freeze_loc_expert", action="store_true", help="For dual_expert: freeze local expert params (even if --dual_train_experts).")
    ap.add_argument(
        "--loss",
        default="weighted_ce",
        choices=["weighted_ce", "balanced_softmax", "robust_balanced_softmax"],
        help="Loss for imbalanced node classification (S1 uses balanced_softmax).",
    )
    ap.add_argument(
        "--rbs_beta",
        type=float,
        default=0.0,
        help="Beta for robust_balanced_softmax (0 disables the robustness term).",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--alpha_risk", type=float, default=0.017, help="DeepEZ risk-sensitive positive term weight.")
    ap.add_argument(
        "--select_metric",
        default="macro_ap_valid",
        choices=["macro_ap_valid", "macro_ap_all", "ap", "auc", "hit_rate_valid", "hit_rate_all"],
        help="Validation metric to select best checkpoint (default: macro_ap_valid).",
    )
    ap.add_argument(
        "--dice_proxy_every",
        type=int,
        default=0,
        help=(
            "If >0, evaluate a fixed-TopK lesion_dice proxy every N epochs on val and save "
            "best_model_diceproxy.pt (+ val_predictions_diceproxy/). Intended to align checkpoint "
            "selection with the final MELD-style post-processing (TopK parcels + area/K constraints)."
        ),
    )
    ap.add_argument("--dice_proxy_topk", type=int, default=8, help="Fixed TopK parcels for dice-proxy evaluation.")
    ap.add_argument("--dice_proxy_k", type=int, default=3, help="Dice-proxy: keep up to K clusters (default 3).")
    ap.add_argument("--dice_proxy_min_vertices", type=int, default=100, help="Dice-proxy: min vertices per cluster (default 100).")
    ap.add_argument("--dice_proxy_area_max_factor", type=float, default=1.5, help="Dice-proxy: A_max factor * max train label area.")
    ap.add_argument("--dice_proxy_area_max_total_cap_cm2", type=float, default=100.0, help="Dice-proxy: hard cap for A_max (cm^2).")
    ap.add_argument("--dice_proxy_area_max_cluster_cm2", type=float, default=40.0, help="Dice-proxy: per-cluster max area (cm^2).")
    ap.add_argument(
        "--dice_proxy_pred_subdir",
        default="val_predictions_diceproxy",
        help="Where to write per-subject val predictions for best_model_diceproxy.pt (relative to out_dir).",
    )
    ap.add_argument(
        "--hit_k",
        type=int,
        default=None,
        help="K for Hit@K on nodes (overrides --hit_frac if set).",
    )
    ap.add_argument(
        "--hit_frac",
        type=float,
        default=0.01,
        help="Fraction of nodes for Hit@K when --hit_k is not set (default 0.01 = top 1 percent).",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args(argv)

    project_dir = Path(__file__).resolve().parents[2]
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = (project_dir / cache_root).resolve()
    cache_dir = cache_root / "cache"
    adj_path = cache_root / "adjacency.npy"
    if not cache_dir.is_dir() or not adj_path.is_file():
        raise FileNotFoundError(f"Missing cache_root artifacts: {cache_root} (need cache/ and adjacency.npy)")

    meta: Dict[str, object] = {}
    atlas_name = "atlas"
    meta_path = cache_root / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            atlas_name = str(meta.get("atlas", atlas_name))
        except Exception:
            atlas_name = "atlas"

    split_json = Path(args.split_json) if args.split_json else _default_split_json_for_fold(project_dir, int(args.fold))
    if split_json is None:
        raise FileNotFoundError("split_json not provided and no default anchor split found for this fold.")
    if not split_json.is_absolute():
        split_json = (project_dir / split_json).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (project_dir / "meld_data/models" / f"{ts}_fmri_gcn_deepez_{args.arch}_{atlas_name}_fold{int(args.fold)}")
    )
    if not out_dir.is_absolute():
        out_dir = (project_dir / out_dir).resolve()
    logger = _configure_logging(out_dir, level=args.log_level)

    logger.info("cache_root=%s", cache_root)
    if meta_path.is_file():
        logger.info("meta.json=%s", meta_path)
    logger.info("split_json=%s", split_json)
    logger.info("out_dir=%s", out_dir)

    train_ids, val_ids = _load_split_ids(split_json)
    logger.info("train_ids=%d val_ids=%d", len(train_ids), len(val_ids))

    # Load caches into memory (small dataset).
    train_samples: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]] = []
    val_samples: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]] = []

    def load_samples(ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]]:
        out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]] = []
        for sid in ids:
            X, y, L_loc, _meta = _load_cache(cache_dir, sid)
            out.append((X, y, L_loc, sid))
        return out

    train_samples = load_samples(train_ids)
    val_samples = load_samples(val_ids)

    # Infer shapes from first sample.
    n_nodes = int(train_samples[0][0].shape[0])
    in_features = int(train_samples[0][0].shape[1])
    fusion = str(args.fusion).strip().lower()
    use_local = fusion in {"film", "film_subgate", "film_hemi_ssdb", "dual_expert"}
    use_film = fusion in {"film", "film_subgate", "film_hemi_ssdb"}
    use_dual = fusion == "dual_expert"
    local_dim = None
    if use_local:
        if train_samples[0][2] is None:
            raise ValueError(f"--fusion={fusion} requires cache files to include L_loc (missing for {train_samples[0][3]})")
        local_dim = int(train_samples[0][2].shape[1])
        if use_film:
            logger.info("fusion=%s local_dim=%d film_z=%d cov_power=%.3f", fusion, int(local_dim), int(args.film_z), float(args.film_cov_power))
        else:
            logger.info("fusion=%s local_dim=%d", fusion, int(local_dim))
    if args.hit_k is not None:
        hit_k = int(args.hit_k)
    else:
        hit_k = int(np.ceil(float(args.hit_frac) * float(n_nodes)))
    hit_k = max(1, min(hit_k, n_nodes))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = DeepEZConfig()
    if not use_local:
        if args.arch == "gcn":
            model = DeepEZGCN(n_nodes=n_nodes, in_features=in_features, cfg=cfg).to(device)
        elif args.arch == "gat":
            model = DeepEZGAT(n_nodes=n_nodes, in_features=in_features, cfg=cfg).to(device)
        else:
            raise ValueError(f"Unknown arch={args.arch}")
    else:
        assert local_dim is not None
        if use_film:
            if fusion == "film":
                if args.arch == "gcn":
                    model = DeepEZGCNFiLM(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                    ).to(device)
                elif args.arch == "gat":
                    model = DeepEZGATFiLM(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                    ).to(device)
                else:
                    raise ValueError(f"Unknown arch={args.arch}")
            elif fusion == "film_subgate":
                if args.arch == "gcn":
                    model = DeepEZGCNFiLMSubGate(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                        subgate_bias=float(args.subgate_bias),
                        subgate_floor=float(args.subgate_floor),
                    ).to(device)
                elif args.arch == "gat":
                    model = DeepEZGATFiLMSubGate(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                        subgate_bias=float(args.subgate_bias),
                        subgate_floor=float(args.subgate_floor),
                    ).to(device)
                else:
                    raise ValueError(f"Unknown arch={args.arch}")
            elif fusion == "film_hemi_ssdb":
                if args.hemi_ssdb_n_mid is not None:
                    n_mid = int(args.hemi_ssdb_n_mid)
                else:
                    n_mid = int(meta.get("n_subcortical_midline", 0) or 0)
                    if n_mid <= 0 and (n_nodes % 2 == 1):
                        n_mid = 1
                if args.arch == "gcn":
                    model = DeepEZGCNFiLMHemiSSDB(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        n_mid=int(n_mid),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                    ).to(device)
                elif args.arch == "gat":
                    model = DeepEZGATFiLMHemiSSDB(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=int(local_dim),
                        n_mid=int(n_mid),
                        cfg=cfg,
                        film_z=int(args.film_z),
                        gate_bias=float(args.film_gate_bias),
                        coverage_power=float(args.film_cov_power),
                    ).to(device)
                else:
                    raise ValueError(f"Unknown arch={args.arch}")
            else:
                raise ValueError(f"Unsupported film fusion={fusion}")
        elif use_dual:
            if args.arch == "gcn":
                model = DeepEZDualExpertGCN(
                    n_nodes=n_nodes,
                    in_features_fc=in_features,
                    in_features_loc=int(local_dim),
                    cfg=cfg,
                    gate_hidden=int(args.dual_gate_hidden),
                    gate_bias=float(args.dual_gate_bias),
                    gate_floor=float(args.dual_gate_floor),
                    gate_use_ssdb_hidden_stats=bool(args.dual_use_ssdb1_stats),
                    coverage_power=float(args.dual_cov_power),
                    bias_mode=str(args.dual_bias_mode),
                    bias_gate_hidden=int(args.dual_bias_gate_hidden),
                    bias_gate_bias=float(args.dual_bias_gate_bias),
                    bias_gate_floor=float(args.dual_bias_gate_floor),
                    node_gate=str(args.dual_node_gate),
                    node_gate_bias=float(args.dual_node_gate_bias),
                    node_gate_floor=float(args.dual_node_gate_floor),
                ).to(device)
            elif args.arch == "gat":
                model = DeepEZDualExpertGAT(
                    n_nodes=n_nodes,
                    in_features_fc=in_features,
                    in_features_loc=int(local_dim),
                    cfg=cfg,
                    gate_hidden=int(args.dual_gate_hidden),
                    gate_bias=float(args.dual_gate_bias),
                    gate_floor=float(args.dual_gate_floor),
                    gate_use_ssdb_hidden_stats=bool(args.dual_use_ssdb1_stats),
                    coverage_power=float(args.dual_cov_power),
                    bias_mode=str(args.dual_bias_mode),
                    bias_gate_hidden=int(args.dual_bias_gate_hidden),
                    bias_gate_bias=float(args.dual_bias_gate_bias),
                    bias_gate_floor=float(args.dual_bias_gate_floor),
                    node_gate=str(args.dual_node_gate),
                    node_gate_bias=float(args.dual_node_gate_bias),
                    node_gate_floor=float(args.dual_node_gate_floor),
                ).to(device)
            else:
                raise ValueError(f"Unknown arch={args.arch}")
        else:
            raise ValueError(f"Unsupported fusion={fusion}")
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)

    def resolve_init_path(x: str) -> Path:
        p = Path(str(x)).expanduser()
        if not p.is_absolute():
            p = (project_dir / p).resolve()
        if p.is_dir():
            p = p / "best_model.pt"
        return p

    # Optional init from a baseline checkpoint (e.g., S4 best_model.pt).
    if args.init_from:
        init_path = resolve_init_path(str(args.init_from))
        if not init_path.is_file():
            raise FileNotFoundError(f"init_from not found: {init_path}")
        sd = torch.load(init_path, map_location=device)
        res = model.load_state_dict(sd, strict=False)
        logger.info("init_from=%s loaded (strict=False): missing=%d unexpected=%d", str(init_path), len(res.missing_keys), len(res.unexpected_keys))

    if use_dual:
        if args.init_from_fc:
            p_fc = resolve_init_path(str(args.init_from_fc))
            if not p_fc.is_file():
                raise FileNotFoundError(f"init_from_fc not found: {p_fc}")
            sd_fc = torch.load(p_fc, map_location=device)
            res_fc = model.fc_expert.load_state_dict(sd_fc, strict=False)
            logger.info(
                "init_from_fc=%s loaded into fc_expert (strict=False): missing=%d unexpected=%d",
                str(p_fc),
                len(res_fc.missing_keys),
                len(res_fc.unexpected_keys),
            )
        if args.init_from_loc:
            p_loc = resolve_init_path(str(args.init_from_loc))
            if not p_loc.is_file():
                raise FileNotFoundError(f"init_from_loc not found: {p_loc}")
            sd_loc = torch.load(p_loc, map_location=device)
            res_loc = model.loc_expert.load_state_dict(sd_loc, strict=False)
            logger.info(
                "init_from_loc=%s loaded into loc_expert (strict=False): missing=%d unexpected=%d",
                str(p_loc),
                len(res_loc.missing_keys),
                len(res_loc.unexpected_keys),
            )
        if not bool(args.dual_train_experts):
            frozen = 0
            for n, p in model.named_parameters():
                if n.startswith("fc_expert.") or n.startswith("loc_expert."):
                    p.requires_grad = False
                    frozen += 1
            logger.info("dual_train_experts=0 froze %d expert params (train gate only)", int(frozen))
        else:
            if bool(args.dual_freeze_fc_expert):
                frozen = 0
                for n, p in model.named_parameters():
                    if n.startswith("fc_expert."):
                        p.requires_grad = False
                        frozen += 1
                logger.info("dual_freeze_fc_expert=1 froze %d fc_expert params", int(frozen))
            if bool(args.dual_freeze_loc_expert):
                frozen = 0
                for n, p in model.named_parameters():
                    if n.startswith("loc_expert."):
                        p.requires_grad = False
                        frozen += 1
                logger.info("dual_freeze_loc_expert=1 froze %d loc_expert params", int(frozen))

    # Freeze trunk if requested (useful when initialised from S4 baseline).
    if bool(args.freeze_trunk):
        trunk_prefixes = ("gc1.", "gc2.", "gat1.", "gat2.")
        frozen = 0
        for n, p in model.named_parameters():
            if n.startswith(trunk_prefixes) or (".gc1." in n) or (".gc2." in n) or (".gat1." in n) or (".gat2." in n):
                p.requires_grad = False
                frozen += 1
        logger.info("freeze_trunk=1 froze %d params (gc/gat trunk)", int(frozen))

    # Class weights from training set.
    y_train_all = np.concatenate([y for _X, y, _L, _sid in train_samples])
    class_w = _compute_class_weights(y_train_all).to(device)
    spc = _compute_sample_per_class(y_train_all, n_classes=2).to(device)

    if args.loss == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_w)

        def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return criterion(logits, y)

    elif args.loss == "balanced_softmax":
        criterion = None

        def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return _balanced_softmax_loss(logits, y, spc)

    elif args.loss == "robust_balanced_softmax":
        criterion = None

        def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return _robust_balanced_softmax_loss(logits, y, spc, beta=float(args.rbs_beta), n_classes=2)

    else:
        raise ValueError(f"Unknown loss={args.loss}")

    # Optionally freeze classifier early for FiLM-style fusion (keeps behaviour close to baseline).
    freeze_classifier_epochs = int(args.freeze_classifier_epochs) if use_film else 0
    if freeze_classifier_epochs > 0:
        for n, p in model.named_parameters():
            if n.startswith("fc_class.") or n.endswith(".fc_class.weight") or n.endswith(".fc_class.bias"):
                p.requires_grad = False
        logger.info("freeze_classifier_epochs=%d (fc_class frozen initially)", freeze_classifier_epochs)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best = {"epoch": -1, "select_metric": str(args.select_metric), "val_select": -1.0, "val_auc": -1.0, "val_ap": -1.0}
    hist_rows: List[Dict[str, object]] = []

    logger.info("n_nodes=%d in_features=%d", n_nodes, in_features)
    if meta:
        if "x_mode" in meta:
            logger.info("x_mode=%s", str(meta.get("x_mode")))
        if "fc_estimator" in meta:
            logger.info("fc_estimator=%s", str(meta.get("fc_estimator")))
        if "fc_transform" in meta:
            logger.info("fc_transform=%s", str(meta.get("fc_transform")))
        if "eb_params" in meta and str(meta.get("eb_params")):
            logger.info("eb_params=%s", str(meta.get("eb_params")))
    logger.info("loss=%s", args.loss)
    logger.info("class_weight=%s", class_w.detach().cpu().numpy().tolist())
    logger.info("sample_per_class=%s", spc.detach().cpu().numpy().astype(int).tolist())
    if args.loss == "robust_balanced_softmax":
        logger.info("rbs_beta=%.6f", float(args.rbs_beta))
    logger.info("arch=%s", args.arch)
    logger.info("fusion=%s", str(args.fusion))
    logger.info("select_metric=%s hit_k=%d (hit_frac=%.4f)", str(args.select_metric), int(hit_k), float(args.hit_frac))
    logger.info(
        "lr=%.6f weight_decay=%.6f alpha_risk=%.6f",
        float(args.lr),
        float(args.weight_decay),
        float(args.alpha_risk),
    )

    # Optional: fixed-TopK lesion_dice proxy for checkpoint selection (P3).
    dice_proxy_every = int(args.dice_proxy_every)
    best_diceproxy: Dict[str, object] = {"epoch": -1, "topk": int(args.dice_proxy_topk), "mean_dice": -1.0}
    dice_proxy_ctx: Optional[Dict[str, object]] = None
    if dice_proxy_every > 0:
        # Lazy imports so the default training path stays lightweight.
        from scripts.fmri_gcn.sweep_deepez_dice import SubjectData, _eval_topk as _eval_topk_fixed
        from scripts.fmri_gcn.evaluate_deepez_candidate_metrics import NodeLayout, _compute_train_a_max_cm2, _load_mgh_vector, _load_subject_hemi
        from meld_graph.fmri_gcn.atlas import load_atlas_brainnetome
        from meld_graph.meld_cohort import MeldCohort

        def _resolve_path(path_str: str, default_rel: str) -> Path:
            if not path_str:
                path_str = default_rel
            p = Path(str(path_str)).expanduser()
            if not p.is_absolute():
                p = (project_dir / p).resolve()
            return p

        derived_dir = _resolve_path(
            str(meta.get("derived_lesion_dir", "")),
            "meld_data/derived_labels/lesion_main_island/template_fsaverage_sym_xhemi",
        )
        demo_csv = _resolve_path(str(meta.get("demographics_csv", "")), "meld_data/demographics.csv")
        hemi_by_id = _load_subject_hemi(demo_csv)

        atlas = load_atlas_brainnetome()
        parcel_index = atlas.fsaverage_sym_lh_parcel_index

        cohort = MeldCohort()
        cortex_mask = cohort.cortex_mask.astype(bool)
        adj_mat = cohort.adj_mat
        area_cortex_cm2 = (cohort.surf_area[cortex_mask].astype(np.float64) / 100.0)  # mm^2 -> cm^2
        area_cm2 = np.concatenate([area_cortex_cm2, area_cortex_cm2])

        layout = NodeLayout(
            n_parcels=int(meta.get("n_parcels_per_hemi", 0) or 0),
            n_sub_hemi=int(meta.get("n_subcortical_paired_per_hemi", 0) or 0),
            n_mid=int(meta.get("n_subcortical_midline", 0) or 0),
        )
        if int(layout.n_parcels) <= 0:
            raise ValueError(f"dice-proxy requires n_parcels_per_hemi in cache meta.json (got {layout.n_parcels})")

        a_max_cm2 = _compute_train_a_max_cm2(
            train_ids=train_ids,
            derived_dir=derived_dir,
            cortex_mask=cortex_mask,
            area_cortex_cm2=area_cortex_cm2,
            area_max_factor=float(args.dice_proxy_area_max_factor),
            logger=logger,
        )
        cap_total = float(args.dice_proxy_area_max_total_cap_cm2)
        if cap_total > 0:
            a_max_cm2 = min(float(a_max_cm2), cap_total)
            logger.info("dice-proxy A_max capped to %.3f cm^2 => %.6f", cap_total, float(a_max_cm2))

        a_max_cluster_cm2: float | None = None
        if float(args.dice_proxy_area_max_cluster_cm2) > 0:
            a_max_cluster_cm2 = float(args.dice_proxy_area_max_cluster_cm2)

        # Preload val labels (vertex-space) once.
        val_static: List[Tuple[str, str, np.ndarray]] = []
        missing: List[str] = []
        for sid in val_ids:
            hemi = hemi_by_id.get(sid)
            if hemi is None:
                missing.append(sid)
                continue
            subdir = derived_dir / sid
            lh_lab = subdir / "lh.on_lh.lesion_main.mgh"
            rh_lab = subdir / "rh.on_lh.lesion_main.mgh"
            if not lh_lab.is_file() or not rh_lab.is_file():
                missing.append(sid)
                continue
            y_lh = _load_mgh_vector(lh_lab)
            y_rh = _load_mgh_vector(rh_lab)
            m_lh = (y_lh > 0.5) & cortex_mask
            m_rh = (y_rh > 0.5) & cortex_mask
            label = np.concatenate([m_lh[cortex_mask], m_rh[cortex_mask]]).astype(bool)
            val_static.append((sid, hemi, label))
        if missing:
            logger.warning(
                "dice-proxy: missing hemi/labels for %d/%d val subjects (first 10): %s",
                len(missing),
                len(val_ids),
                missing[:10],
            )

        val_by_id = {sid: (X_np, y_np, L_np) for X_np, y_np, L_np, sid in val_samples}

        def _eval_dice_proxy(model0: nn.Module) -> Dict[str, float]:
            model0.eval()
            subjects: List[SubjectData] = []
            with torch.no_grad():
                for sid, hemi, label in val_static:
                    rec = val_by_id.get(sid)
                    if rec is None:
                        continue
                    X_np, _y_np, L_np = rec
                    X = torch.from_numpy(X_np).to(device)
                    if use_local:
                        if L_np is None:
                            continue
                        L = torch.from_numpy(L_np).to(device)
                        logits, _bias = model0(X, adj, L)
                    else:
                        logits, _bias = model0(X, adj)
                    p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32).reshape(-1)
                    subjects.append(SubjectData(subject_id=sid, hemi=hemi, p=p, label=label))

            return _eval_topk_fixed(
                subjects=subjects,
                topk_parcels=int(args.dice_proxy_topk),
                layout=layout,
                parcel_index=parcel_index,
                cortex_mask=cortex_mask,
                adj_mat=adj_mat,
                area_cm2=area_cm2,
                a_max_cm2=float(a_max_cm2),
                a_max_cluster_cm2=a_max_cluster_cm2,
                min_vertices=int(args.dice_proxy_min_vertices),
                k_clusters=int(args.dice_proxy_k),
                iou_threshold=0.15,
                dice_threshold=0.15,
            )

        dice_proxy_ctx = {"eval_fn": _eval_dice_proxy}
        logger.info(
            "dice-proxy enabled: every=%d topk=%d k=%d min_vertices=%d A_max=%.3fcm^2 cap_total=%.3f cluster_cap=%.3f",
            dice_proxy_every,
            int(args.dice_proxy_topk),
            int(args.dice_proxy_k),
            int(args.dice_proxy_min_vertices),
            float(a_max_cm2),
            float(args.dice_proxy_area_max_total_cap_cm2),
            float(args.dice_proxy_area_max_cluster_cm2),
        )

    for epoch in range(int(args.epochs)):
        if freeze_classifier_epochs > 0 and epoch == freeze_classifier_epochs:
            for n, p in model.named_parameters():
                if n.startswith("fc_class.") or n.endswith(".fc_class.weight") or n.endswith(".fc_class.bias"):
                    p.requires_grad = True
            logger.info("Unfroze fc_class at epoch=%d", int(epoch))

        model.train()
        losses: List[float] = []
        for X_np, y_np, L_np, sid in train_samples:
            X = torch.from_numpy(X_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            opt.zero_grad(set_to_none=True)

            if use_local:
                if L_np is None:
                    raise ValueError(f"Missing L_loc for subject={sid} but fusion={fusion}")
                L = torch.from_numpy(L_np).to(device)
                logits, _bias = model(X, adj, L)
            else:
                logits, _bias = model(X, adj)
            loss = loss_fn(logits, y)
            # risk-sensitive positive encouragement (DeepEZ-style)
            if float(args.alpha_risk) > 0:
                pos_mask = y == 1
                if torch.any(pos_mask):
                    loss = loss + float(args.alpha_risk) * (-logits[pos_mask, 1].mean())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        val_m = _eval_epoch(model, adj, val_samples, device=device, loss_fn=loss_fn, hit_k=hit_k, use_local=use_local)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_m.loss,
            "val_auc": val_m.auc,
            "val_ap": val_m.ap,
            "val_prevalence": val_m.prevalence,
            "val_ap_random": val_m.ap_random,
            "val_ap_lift_ratio": val_m.ap_lift_ratio,
            "val_ap_lift_diff": val_m.ap_lift_diff,
            "val_macro_auc_valid": val_m.macro_auc_valid,
            "val_macro_ap_valid": val_m.macro_ap_valid,
            "val_macro_auc_all": val_m.macro_auc_all,
            "val_macro_ap_all": val_m.macro_ap_all,
            "val_hit_k": val_m.hit_k,
            "val_hit_rate_valid": val_m.hit_rate_valid,
            "val_hit_rate_all": val_m.hit_rate_all,
        }

        # Optional dice-proxy evaluation (fixed TopK + MELD-style post-processing).
        if dice_proxy_ctx is not None and (epoch % dice_proxy_every == 0 or epoch == int(args.epochs) - 1):
            dp = dice_proxy_ctx["eval_fn"](model)  # type: ignore[index]
            dp_mean = float(dp.get("mean_dice", float("nan")))
            dp_median = float(dp.get("median_dice", float("nan")))
            dp_recall = float(dp.get("recall_dice", float("nan")))
            row.update(
                {
                    "val_dice_proxy_topk": int(args.dice_proxy_topk),
                    "val_dice_proxy_mean": dp_mean,
                    "val_dice_proxy_median": dp_median,
                    "val_dice_proxy_recall0p15": dp_recall,
                }
            )
            if (not np.isnan(dp_mean)) and (dp_mean > float(best_diceproxy.get("mean_dice", -1.0))):
                best_diceproxy = {
                    "epoch": int(epoch),
                    "topk": int(args.dice_proxy_topk),
                    "mean_dice": float(dp_mean),
                    "median_dice": float(dp_median),
                    "recall0p15": float(dp_recall),
                }
                torch.save(model.state_dict(), out_dir / "best_model_diceproxy.pt")
                logger.info(
                    "dice-proxy *best epoch=%d topk=%d mean_dice=%.4f recall0p15=%.3f",
                    int(epoch),
                    int(args.dice_proxy_topk),
                    float(dp_mean),
                    float(dp_recall),
                )

        hist_rows.append(row)

        sel = float(getattr(val_m, str(args.select_metric)))
        improved = (not np.isnan(sel)) and (sel > float(best.get("val_select", -1.0)))
        if improved:
            best = {
                "epoch": epoch,
                "select_metric": str(args.select_metric),
                "val_select": float(sel),
                "val_auc": float(val_m.auc),
                "val_ap": float(val_m.ap),
                "val_macro_ap_valid": float(val_m.macro_ap_valid),
                "val_macro_ap_all": float(val_m.macro_ap_all),
                "val_hit_rate_valid": float(val_m.hit_rate_valid),
                "val_hit_rate_all": float(val_m.hit_rate_all),
                "val_prevalence": float(val_m.prevalence),
            }
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0 or improved or epoch == int(args.epochs) - 1:
            logger.info(
                "epoch=%d train_loss=%.4f val_auc=%.4f val_ap=%.4f prev=%.5f macro_ap=%.4f hit@%d=%.3f sel(%s)=%.4f%s",
                epoch,
                tr_loss,
                val_m.auc,
                val_m.ap,
                val_m.prevalence,
                val_m.macro_ap_valid,
                int(val_m.hit_k),
                val_m.hit_rate_valid,
                str(args.select_metric),
                float(sel),
                " *best" if improved else "",
            )

    # Save final and history.
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    pd.DataFrame(hist_rows).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    if dice_proxy_ctx is not None:
        (out_dir / "best_diceproxy.json").write_text(json.dumps(best_diceproxy, indent=2), encoding="utf-8")

    config = {
        "cache_root": str(cache_root),
        "meta": meta,
        "split_json": str(split_json),
        "fold": int(args.fold),
        "arch": str(args.arch),
        "fusion": str(args.fusion),
        "init_from": str(args.init_from) if args.init_from else None,
        "init_from_fc": str(args.init_from_fc) if args.init_from_fc else None,
        "init_from_loc": str(args.init_from_loc) if args.init_from_loc else None,
        "dual_train_experts": bool(args.dual_train_experts),
        "dual_freeze_fc_expert": bool(args.dual_freeze_fc_expert),
        "dual_freeze_loc_expert": bool(args.dual_freeze_loc_expert),
        "freeze_trunk": bool(args.freeze_trunk),
        "freeze_classifier_epochs": int(freeze_classifier_epochs),
        "film_z": int(args.film_z) if use_film else None,
        "film_gate_bias": float(args.film_gate_bias) if use_film else None,
        "film_cov_power": float(args.film_cov_power) if use_film else None,
        "subgate_bias": float(args.subgate_bias) if fusion == "film_subgate" else None,
        "subgate_floor": float(args.subgate_floor) if fusion == "film_subgate" else None,
        "hemi_ssdb_n_mid": int(args.hemi_ssdb_n_mid) if args.hemi_ssdb_n_mid is not None else None,
        "dual_gate_hidden": int(args.dual_gate_hidden) if use_dual else None,
        "dual_gate_bias": float(args.dual_gate_bias) if use_dual else None,
        "dual_gate_floor": float(args.dual_gate_floor) if use_dual else None,
        "dual_use_ssdb1_stats": bool(args.dual_use_ssdb1_stats) if use_dual else None,
        "dual_cov_power": float(args.dual_cov_power) if use_dual else None,
        "dual_bias_mode": str(args.dual_bias_mode) if use_dual else None,
        "dual_bias_gate_hidden": int(args.dual_bias_gate_hidden) if use_dual else None,
        "dual_bias_gate_bias": float(args.dual_bias_gate_bias) if use_dual else None,
        "dual_bias_gate_floor": float(args.dual_bias_gate_floor) if use_dual else None,
        "dual_node_gate": str(args.dual_node_gate) if use_dual else None,
        "dual_node_gate_bias": float(args.dual_node_gate_bias) if use_dual else None,
        "dual_node_gate_floor": float(args.dual_node_gate_floor) if use_dual else None,
        "loss": str(args.loss),
        "rbs_beta": float(args.rbs_beta),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "alpha_risk": float(args.alpha_risk),
        "select_metric": str(args.select_metric),
        "dice_proxy_every": int(args.dice_proxy_every),
        "dice_proxy_topk": int(args.dice_proxy_topk),
        "dice_proxy_k": int(args.dice_proxy_k),
        "dice_proxy_min_vertices": int(args.dice_proxy_min_vertices),
        "dice_proxy_area_max_factor": float(args.dice_proxy_area_max_factor),
        "dice_proxy_area_max_total_cap_cm2": float(args.dice_proxy_area_max_total_cap_cm2),
        "dice_proxy_area_max_cluster_cm2": float(args.dice_proxy_area_max_cluster_cm2),
        "dice_proxy_pred_subdir": str(args.dice_proxy_pred_subdir),
        "hit_k": int(hit_k),
        "hit_frac": float(args.hit_frac),
        "seed": int(args.seed),
        "n_nodes": n_nodes,
        "class_weight": class_w.detach().cpu().numpy().tolist(),
        "sample_per_class": spc.detach().cpu().numpy().astype(int).tolist(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Save per-subject val metrics for the best checkpoint (helps diagnose outliers).
    best_path = out_dir / "best_model.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device))
        df_sub = _eval_per_subject(
            model,
            adj,
            val_samples,
            device=device,
            hit_k=hit_k,
            use_local=use_local,
            out_dir=out_dir,
            pred_subdir="val_predictions",
        )
        df_sub.to_csv(out_dir / "val_subject_metrics.csv", index=False)

    best_dp_path = out_dir / "best_model_diceproxy.pt"
    if best_dp_path.is_file():
        model.load_state_dict(torch.load(best_dp_path, map_location=device))
        df_dp = _eval_per_subject(
            model,
            adj,
            val_samples,
            device=device,
            hit_k=hit_k,
            use_local=use_local,
            out_dir=out_dir,
            pred_subdir=str(args.dice_proxy_pred_subdir),
        )
        df_dp.to_csv(out_dir / "val_subject_metrics_diceproxy.csv", index=False)

    logger.info("Saved: %s", out_dir)
    logger.info("Best: %s", best)
    if dice_proxy_ctx is not None:
        logger.info("Best dice-proxy: %s", best_diceproxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
