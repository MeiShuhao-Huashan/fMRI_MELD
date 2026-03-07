#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from meld_fmri.fmri_gcn.laterality import (  # noqa: E402
    DeepEZDualExpertLateralityHead,
    _TrunkFCOnly,
    _TrunkLocOnly,
)
from meld_fmri.fmri_gcn.model import (  # noqa: E402
    DeepEZConfig,
    DeepEZDualExpertGCN,
    DeepEZDualExpertGAT,
    DeepEZGCN,
    DeepEZGAT,
)


def _resolve_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if not pp.is_absolute():
        pp = (PROJECT_DIR / pp).resolve()
    return pp


def _load_ids_from_txt(p: Path) -> List[str]:
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def _load_ids_from_split_json(p: Path, split: str) -> List[str]:
    d = json.loads(p.read_text(encoding="utf-8"))
    key = f"{split}_ids"
    if key not in d:
        raise ValueError(f"Missing {key} in split_json: {p}")
    return [str(x) for x in d[key]]


def _load_cache_npz(cache_dir: Path, sid: str) -> Tuple[np.ndarray, np.ndarray]:
    p = cache_dir / f"{sid}.npz"
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    if "L_loc" not in z.files:
        raise ValueError(f"Missing L_loc in cache: {p}")
    L = z["L_loc"].astype(np.float32)
    return X, L


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run inference for a trained DeepEZ laterality (L/R) classifier on absolute-order caches.")
    ap.add_argument("--cache_root", required=True, help="cache_root with cache/*.npz + adjacency.npy + meta.json.")
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt from train_deepez_laterality.py.")
    ap.add_argument("--trunk_mode", choices=["dual", "fc_only", "loc_only"], default="dual")
    ap.add_argument("--arch", choices=["gcn", "gat"], default="gat")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")

    # Subject selection
    ap.add_argument("--subject_ids_txt", default=None, help="Text file with one subject_id per line.")
    ap.add_argument("--split_json", default=None, help="Split JSON with train_ids/val_ids (data_parameters.json).")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--all_cache", action="store_true", help="Predict all cache/*.npz subjects.")

    ap.add_argument("--out_csv", required=True, help="Output CSV with columns: subject_id, prob_right, pred_hemi.")
    args = ap.parse_args(argv)

    cache_root = _resolve_path(args.cache_root)
    cache_dir = cache_root / "cache"
    adj_path = cache_root / "adjacency.npy"
    meta_path = cache_root / "meta.json"
    if not cache_dir.is_dir() or not adj_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Missing cache_root artifacts: {cache_root} (need cache/, adjacency.npy, meta.json)")

    ckpt_path = _resolve_path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_parcels = int(meta.get("n_parcels_per_hemi", 0) or 0)
    n_sub = int(meta.get("n_subcortical_paired_per_hemi", 0) or 0)
    n_mid = int(meta.get("n_subcortical_midline", 0) or 0)
    n_hemi = int(n_parcels) + int(n_sub)
    n_nodes_expected = 2 * n_hemi + int(n_mid)

    # Subject ids
    if args.subject_ids_txt:
        subject_ids = _load_ids_from_txt(_resolve_path(args.subject_ids_txt))
    elif args.split_json:
        subject_ids = _load_ids_from_split_json(_resolve_path(args.split_json), str(args.split))
    elif args.all_cache:
        subject_ids = sorted(p.stem for p in cache_dir.glob("*.npz"))
    else:
        raise ValueError("Provide --subject_ids_txt, --split_json (+--split), or --all_cache.")
    if not subject_ids:
        raise ValueError("No subject_ids selected.")

    # Infer dims from first cache.
    X0, L0 = _load_cache_npz(cache_dir, subject_ids[0])
    if int(X0.shape[0]) != int(n_nodes_expected):
        raise ValueError(f"Node mismatch: X0={X0.shape} expected_n_nodes={n_nodes_expected}")
    in_features = int(X0.shape[1])
    local_dim = int(L0.shape[1])

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for sid in subject_ids:
            X_np, L_np = _load_cache_npz(cache_dir, str(sid))
            X = torch.from_numpy(X_np).to(device)
            L = torch.from_numpy(L_np).to(device)
            logit = model(X, adj, L).view(-1)
            prob_right = float(torch.sigmoid(logit)[0].detach().cpu().item())
            pred_hemi = "R" if prob_right >= 0.5 else "L"
            rows.append({"subject_id": str(sid), "prob_right": prob_right, "pred_hemi": pred_hemi})

    out_csv = _resolve_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"ok: wrote {len(rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

