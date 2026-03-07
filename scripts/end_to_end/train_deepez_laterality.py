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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from meld_fmri.fmri_gcn.model import DeepEZConfig, DeepEZDualExpertGCN, DeepEZDualExpertGAT, DeepEZGCN, DeepEZGAT
from meld_fmri.fmri_gcn.laterality import DeepEZDualExpertLateralityHead, _TrunkFCOnly, _TrunkLocOnly


def _configure_logging(out_dir: Path, level: str = "INFO") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fmri_gcn.train_laterality")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "train_deepez_laterality.log", encoding="utf-8")
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
    candidates = [
        project_dir / f"meld_data/models/25-12-16_MELD_fMRI_fold{fold}/fold_{fold:02d}/data_parameters.json",
        project_dir / f"meld_data/models/25-12-15_MELD_fMRI_fold{fold}/fold_{fold:02d}/data_parameters.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_cache(cache_dir: Path, subject_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], str, Dict[str, object]]:
    p = cache_dir / f"{subject_id}.npz"
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    L_loc = z["L_loc"].astype(np.float32) if "L_loc" in z.files else None
    hemi = str(z.get("hemi", "")).strip().upper()
    if hemi not in {"L", "R"}:
        raise ValueError(f"Invalid hemi in cache {p}: {hemi!r}")
    meta = {
        "path": str(p),
        "t": int(z.get("t", 0)),
    }
    return X, L_loc, hemi, meta


def _hemi_to_label(hemi: str) -> int:
    h = str(hemi).strip().upper()
    if h == "L":
        return 0
    if h == "R":
        return 1
    raise ValueError(f"Invalid hemi={hemi!r} (expected L/R)")


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    acc: float
    bal_acc: float
    auc: float
    ap: float


def _eval_epoch(
    model: nn.Module,
    adj: torch.Tensor,
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[EpochMetrics, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    losses: List[float] = []
    ys: List[float] = []
    ps: List[float] = []
    sids: List[str] = []
    with torch.no_grad():
        for X_np, L_np, y_np, sid in samples:
            X = torch.from_numpy(X_np).to(device)
            L = torch.from_numpy(L_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            logit = model(X, adj, L)
            loss = loss_fn(logit.view(-1), y.view(-1))
            losses.append(float(loss.detach().cpu().item()))
            p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
            ys.append(float(y_np.reshape(-1)[0]))
            ps.append(float(p[0]))
            sids.append(str(sid))

    y_all = np.asarray(ys, dtype=np.float64)
    p_all = np.asarray(ps, dtype=np.float64)
    pred = (p_all >= 0.5).astype(np.int64)
    acc = float(np.mean(pred == y_all.astype(np.int64))) if y_all.size else float("nan")

    # Balanced accuracy (mean recall for L and R).
    y_i = y_all.astype(np.int64)
    tp = int(np.sum((pred == 1) & (y_i == 1)))
    tn = int(np.sum((pred == 0) & (y_i == 0)))
    fp = int(np.sum((pred == 1) & (y_i == 0)))
    fn = int(np.sum((pred == 0) & (y_i == 1)))
    tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else float("nan")  # recall_R
    tnr = float(tn) / float(tn + fp) if (tn + fp) > 0 else float("nan")  # recall_L
    if np.isfinite(tpr) and np.isfinite(tnr):
        bal_acc = 0.5 * (tpr + tnr)
    else:
        bal_acc = float("nan")

    # Guard: AUC undefined if a split has only one class.
    try:
        auc = float(roc_auc_score(y_all, p_all)) if (np.unique(y_all).size == 2) else float("nan")
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_all, p_all)) if (np.sum(y_all == 1) > 0) else float("nan")
    except Exception:
        ap = float("nan")

    m = EpochMetrics(loss=float(np.mean(losses)) if losses else float("nan"), acc=acc, bal_acc=bal_acc, auc=auc, ap=ap)
    return m, y_all, p_all, sids


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a DeepEZ dual-expert laterality (L/R) classifier on absolute-order caches.")
    ap.add_argument("--cache_root", required=True, help="cache_root with cache/*.npz + adjacency.npy + meta.json (prefer node_order=lh_rh).")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split_json", default=None)
    ap.add_argument(
        "--trunk_mode",
        choices=["dual", "fc_only", "loc_only"],
        default="dual",
        help="Which node-logit trunk to use: dual (FC+local experts), fc_only, or loc_only.",
    )
    ap.add_argument("--arch", choices=["gcn", "gat"], default="gat")
    ap.add_argument("--topk", type=int, default=20, help="TopK used in the laterality pooling head (default 20).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args(argv)

    project_dir = Path(__file__).resolve().parents[2]
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = (project_dir / cache_root).resolve()
    cache_dir = cache_root / "cache"
    adj_path = cache_root / "adjacency.npy"
    meta_path = cache_root / "meta.json"
    if not cache_dir.is_dir() or not adj_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing cache_root artifacts: {cache_root} (need cache/, adjacency.npy, meta.json)")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_parcels = int(meta.get("n_parcels_per_hemi", 0) or 0)
    n_sub = int(meta.get("n_subcortical_paired_per_hemi", 0) or 0)
    n_mid = int(meta.get("n_subcortical_midline", 0) or 0)
    n_hemi = int(n_parcels) + int(n_sub)
    n_nodes_expected = 2 * n_hemi + int(n_mid)

    split_json = Path(args.split_json) if args.split_json else _default_split_json_for_fold(project_dir, int(args.fold))
    if split_json is None:
        raise FileNotFoundError("split_json not provided and no default anchor split found for this fold.")
    if not split_json.is_absolute():
        split_json = (project_dir / split_json).resolve()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (project_dir / "meld_data/models" / f"{ts}_fmri_gcn_deepez_laterality_{args.arch}_fold{int(args.fold)}")
    )
    if not out_dir.is_absolute():
        out_dir = (project_dir / out_dir).resolve()
    logger = _configure_logging(out_dir, level=args.log_level)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "adjacency_npy": str(adj_path),
                "meta_json": str(meta_path),
                "split_json": str(split_json),
                "fold": int(args.fold),
                "trunk_mode": str(args.trunk_mode),
                "arch": str(args.arch),
                "topk": int(args.topk),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "device": str(args.device),
                "n_parcels_per_hemi": int(n_parcels),
                "n_subcortical_paired_per_hemi": int(n_sub),
                "n_subcortical_midline": int(n_mid),
                "n_hemi": int(n_hemi),
                "n_nodes_expected": int(n_nodes_expected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("cache_root=%s", cache_root)
    logger.info("meta.json=%s", meta_path)
    logger.info("split_json=%s", split_json)
    logger.info("out_dir=%s", out_dir)
    logger.info("n_parcels=%d n_sub=%d n_mid=%d n_hemi=%d n_nodes_expected=%d", n_parcels, n_sub, n_mid, n_hemi, n_nodes_expected)

    train_ids, val_ids = _load_split_ids(split_json)
    logger.info("train_ids=%d val_ids=%d", len(train_ids), len(val_ids))

    # Load caches into memory (tiny dataset).
    train_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    val_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    def load_samples(ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for sid in ids:
            X, L, hemi, _m = _load_cache(cache_dir, sid)
            if L is None:
                raise ValueError(f"Missing L_loc for subject={sid} (laterality head expects dual-expert cache with L_loc)")
            if int(X.shape[0]) != int(n_nodes_expected):
                raise ValueError(f"Node mismatch for {sid}: X={X.shape} expected_n_nodes={n_nodes_expected}")
            y = np.asarray([_hemi_to_label(hemi)], dtype=np.float32)
            out.append((X, L, y, sid))
        return out

    train_samples = load_samples(train_ids)
    val_samples = load_samples(val_ids)

    # Shapes from cache.
    in_features = int(train_samples[0][0].shape[1])
    local_dim = int(train_samples[0][1].shape[1])
    logger.info("in_features=%d local_dim=%d", in_features, local_dim)

    # Reproducibility.
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(str(args.device))
    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)

    cfg = DeepEZConfig()
    trunk_mode = str(args.trunk_mode).strip().lower()
    arch = str(args.arch).strip().lower()
    if trunk_mode == "dual":
        if arch == "gcn":
            trunk = DeepEZDualExpertGCN(n_nodes=n_nodes_expected, in_features_fc=in_features, in_features_loc=local_dim, cfg=cfg).to(device)
        else:
            trunk = DeepEZDualExpertGAT(n_nodes=n_nodes_expected, in_features_fc=in_features, in_features_loc=local_dim, cfg=cfg).to(device)
    elif trunk_mode == "fc_only":
        if arch == "gcn":
            trunk0 = DeepEZGCN(n_nodes=n_nodes_expected, in_features=in_features, cfg=cfg).to(device)
        else:
            trunk0 = DeepEZGAT(n_nodes=n_nodes_expected, in_features=in_features, cfg=cfg).to(device)
        trunk = _TrunkFCOnly(trunk0).to(device)
    else:  # loc_only
        if arch == "gcn":
            trunk0 = DeepEZGCN(n_nodes=n_nodes_expected, in_features=local_dim, cfg=cfg).to(device)
        else:
            trunk0 = DeepEZGAT(n_nodes=n_nodes_expected, in_features=local_dim, cfg=cfg).to(device)
        trunk = _TrunkLocOnly(trunk0).to(device)

    model = DeepEZDualExpertLateralityHead(trunk, n_hemi=n_hemi, topk=int(args.topk)).to(device)

    # Class imbalance handling (pos_weight for BCE).
    y_train = np.array([float(s[2].reshape(-1)[0]) for s in train_samples], dtype=np.float32)
    n_pos = float(np.sum(y_train == 1.0))
    n_neg = float(np.sum(y_train == 0.0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("train class balance: n_pos=%.0f n_neg=%.0f pos_weight=%.3f", n_pos, n_neg, float(pos_weight.item()))

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "val_bal_acc": -1.0,
        "val_auc": float("nan"),
        "val_ap": float("nan"),
        "val_loss": float("inf"),
    }
    best_path = out_dir / "best_model.pt"

    # Training loop (batch size = 1 subject).
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        # Shuffle subjects each epoch.
        order = np.random.permutation(len(train_samples))
        for idx in order:
            X_np, L_np, y_np, sid = train_samples[int(idx)]
            X = torch.from_numpy(X_np).to(device)
            L = torch.from_numpy(L_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            opt.zero_grad(set_to_none=True)
            logit = model(X, adj, L)
            loss = loss_fn(logit.view(-1), y.view(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_m, y_val, p_val, sids_val = _eval_epoch(model, adj, val_samples, device=device, loss_fn=loss_fn)

        logger.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f val_acc=%.3f val_bal_acc=%s val_auc=%s val_ap=%s",
            epoch,
            train_loss,
            val_m.loss,
            val_m.acc,
            f"{val_m.bal_acc:.3f}" if np.isfinite(val_m.bal_acc) else "nan",
            f"{val_m.auc:.3f}" if np.isfinite(val_m.auc) else "nan",
            f"{val_m.ap:.3f}" if np.isfinite(val_m.ap) else "nan",
        )

        # Select best by balanced accuracy (preferred), else accuracy; break ties by loss.
        improved = False
        if np.isfinite(val_m.bal_acc) and np.isfinite(float(best.get("val_bal_acc", float("nan")))):
            if val_m.bal_acc > float(best["val_bal_acc"]) + 1e-6:
                improved = True
            elif abs(val_m.bal_acc - float(best["val_bal_acc"])) <= 1e-6 and val_m.loss < float(best["val_loss"]) - 1e-6:
                improved = True
        elif np.isfinite(val_m.bal_acc) and not np.isfinite(float(best.get("val_bal_acc", float("nan")))):
            improved = True
        else:
            if val_m.acc > float(best["val_acc"]) + 1e-6:
                improved = True
            elif abs(val_m.acc - float(best["val_acc"])) <= 1e-6 and val_m.loss < float(best["val_loss"]) - 1e-6:
                improved = True
        if improved:
            best = {
                "epoch": int(epoch),
                "val_acc": float(val_m.acc),
                "val_bal_acc": float(val_m.bal_acc),
                "val_auc": float(val_m.auc),
                "val_ap": float(val_m.ap),
                "val_loss": float(val_m.loss),
            }
            torch.save(model.state_dict(), best_path)

            # Dump per-subject val predictions.
            pred_label = (p_val >= 0.5).astype(np.int64)
            out_csv = out_dir / "val_predictions.csv"
            rows = []
            for sid, y0, p0, pl in zip(sids_val, y_val.tolist(), p_val.tolist(), pred_label.tolist()):
                rows.append(
                    {
                        "subject_id": str(sid),
                        "true_hemi": "R" if int(y0) == 1 else "L",
                        "prob_right": float(p0),
                        "pred_hemi": "R" if int(pl) == 1 else "L",
                        "correct": bool(int(pl) == int(y0)),
                    }
                )
            import pandas as pd

            pd.DataFrame(rows).to_csv(out_csv, index=False)
            (out_dir / "best_meta.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
            logger.info(
                "saved best_model.pt (epoch=%d val_acc=%.3f val_bal_acc=%s)",
                int(best["epoch"]),
                float(best["val_acc"]),
                f"{float(best['val_bal_acc']):.3f}" if np.isfinite(float(best["val_bal_acc"])) else "nan",
            )

    logger.info("done. best=%s", json.dumps(best, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
