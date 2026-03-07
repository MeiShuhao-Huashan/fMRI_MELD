#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from meld_fmri.fmri_gcn.model import (  # noqa: E402
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


def _discover_ids_from_cache(cache_dir: Path) -> List[str]:
    out = []
    for p in sorted(cache_dir.glob("*.npz")):
        out.append(p.stem)
    return out


def _load_cache_npz(cache_dir: Path, sid: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    p = cache_dir / f"{sid}.npz"
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    L_loc = z["L_loc"].astype(np.float32) if "L_loc" in z.files else None
    y = z["y"].astype(np.int64) if "y" in z.files else None
    return X, L_loc, y


def _infer_n_mid_from_meta(*, cache_root: Path, n_nodes: int) -> int:
    meta_path = cache_root / "meta.json"
    n_mid = 0
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n_mid = int(meta.get("n_subcortical_midline", 0) or 0)
        except Exception:
            n_mid = 0
    if n_mid <= 0 and (int(n_nodes) % 2 == 1):
        n_mid = 1
    return int(n_mid)


def _build_model_from_train_config(
    *,
    config: Dict[str, object],
    n_nodes: int,
    in_features: int,
    local_dim: Optional[int],
    cache_root: Path,
) -> Tuple[torch.nn.Module, bool]:
    cfg = DeepEZConfig()
    arch = str(config.get("arch", "gcn")).strip().lower()
    fusion = str(config.get("fusion", "none")).strip().lower()
    use_local = fusion in {"film", "film_subgate", "film_hemi_ssdb", "dual_expert"}
    use_film = fusion in {"film", "film_subgate", "film_hemi_ssdb"}
    use_dual = fusion == "dual_expert"

    if use_local and local_dim is None:
        raise ValueError(f"Config requires fusion={fusion} but caches do not contain L_loc.")

    if not use_local:
        if arch == "gcn":
            return DeepEZGCN(n_nodes=n_nodes, in_features=in_features, cfg=cfg), False
        if arch == "gat":
            return DeepEZGAT(n_nodes=n_nodes, in_features=in_features, cfg=cfg), False
        raise ValueError(f"Unknown arch={arch!r}")

    assert local_dim is not None
    if use_film:
        film_z = int(config.get("film_z", 32) or 32)
        gate_bias = float(config.get("film_gate_bias", -2.0) or -2.0)
        cov_power = float(config.get("film_cov_power", 1.0) or 1.0)
        if fusion == "film":
            if arch == "gcn":
                return (
                    DeepEZGCNFiLM(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                    ),
                    True,
                )
            if arch == "gat":
                return (
                    DeepEZGATFiLM(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                    ),
                    True,
                )
            raise ValueError(f"Unknown arch={arch!r}")

        if fusion == "film_subgate":
            subgate_bias = float(config.get("subgate_bias", -2.0) or -2.0)
            subgate_floor = float(config.get("subgate_floor", 0.0) or 0.0)
            if arch == "gcn":
                return (
                    DeepEZGCNFiLMSubGate(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                        subgate_bias=subgate_bias,
                        subgate_floor=subgate_floor,
                    ),
                    True,
                )
            if arch == "gat":
                return (
                    DeepEZGATFiLMSubGate(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                        subgate_bias=subgate_bias,
                        subgate_floor=subgate_floor,
                    ),
                    True,
                )
            raise ValueError(f"Unknown arch={arch!r}")

        if fusion == "film_hemi_ssdb":
            n_mid_cfg = config.get("hemi_ssdb_n_mid", None)
            n_mid = int(n_mid_cfg) if n_mid_cfg is not None else _infer_n_mid_from_meta(cache_root=cache_root, n_nodes=n_nodes)
            if arch == "gcn":
                return (
                    DeepEZGCNFiLMHemiSSDB(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        n_mid=n_mid,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                    ),
                    True,
                )
            if arch == "gat":
                return (
                    DeepEZGATFiLMHemiSSDB(
                        n_nodes=n_nodes,
                        in_features=in_features,
                        local_dim=local_dim,
                        n_mid=n_mid,
                        cfg=cfg,
                        film_z=film_z,
                        gate_bias=gate_bias,
                        coverage_power=cov_power,
                    ),
                    True,
                )
            raise ValueError(f"Unknown arch={arch!r}")

        raise ValueError(f"Unsupported fusion={fusion!r}")

    if use_dual:
        gate_hidden = int(config.get("dual_gate_hidden", 16) or 16)
        gate_bias = float(config.get("dual_gate_bias", -2.0) or -2.0)
        gate_floor = float(config.get("dual_gate_floor", 0.0) or 0.0)
        use_ssdb1_stats = bool(config.get("dual_use_ssdb1_stats", False))
        cov_power = float(config.get("dual_cov_power", 1.0) or 1.0)
        bias_mode = str(config.get("dual_bias_mode", "shared")).strip().lower()
        bias_gate_hidden = int(config.get("dual_bias_gate_hidden", 16) or 16)
        bias_gate_bias = float(config.get("dual_bias_gate_bias", -4.0) or -4.0)
        bias_gate_floor = float(config.get("dual_bias_gate_floor", 0.0) or 0.0)
        node_gate = str(config.get("dual_node_gate", "coverage")).strip().lower()
        node_gate_bias = float(config.get("dual_node_gate_bias", 2.0) or 2.0)
        node_gate_floor = float(config.get("dual_node_gate_floor", 0.0) or 0.0)

        if arch == "gcn":
            return (
                DeepEZDualExpertGCN(
                    n_nodes=n_nodes,
                    in_features_fc=in_features,
                    in_features_loc=local_dim,
                    cfg=cfg,
                    gate_hidden=gate_hidden,
                    gate_bias=gate_bias,
                    gate_floor=gate_floor,
                    use_ssdb1_stats=use_ssdb1_stats,
                    coverage_power=cov_power,
                    bias_mode=bias_mode,
                    bias_gate_hidden=bias_gate_hidden,
                    bias_gate_bias=bias_gate_bias,
                    bias_gate_floor=bias_gate_floor,
                    node_gate_mode=node_gate,
                    node_gate_bias=node_gate_bias,
                    node_gate_floor=node_gate_floor,
                ),
                True,
            )
        if arch == "gat":
            return (
                DeepEZDualExpertGAT(
                    n_nodes=n_nodes,
                    in_features_fc=in_features,
                    in_features_loc=local_dim,
                    cfg=cfg,
                    gate_hidden=gate_hidden,
                    gate_bias=gate_bias,
                    gate_floor=gate_floor,
                    use_ssdb1_stats=use_ssdb1_stats,
                    coverage_power=cov_power,
                    bias_mode=bias_mode,
                    bias_gate_hidden=bias_gate_hidden,
                    bias_gate_bias=bias_gate_bias,
                    bias_gate_floor=bias_gate_floor,
                    node_gate_mode=node_gate,
                    node_gate_bias=node_gate_bias,
                    node_gate_floor=node_gate_floor,
                ),
                True,
            )
        raise ValueError(f"Unknown arch={arch!r}")

    raise ValueError(f"Unsupported fusion={fusion!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run inference for a trained DeepEZ-style GCN/GAT fMRI model on precomputed caches.")
    ap.add_argument("--cache_root", required=True, help="cache_root with cache/*.npz and adjacency.npy (and optional meta.json).")
    ap.add_argument("--model_dir", default=None, help="Model dir from train_deepez_gcn.py (contains config.json + best_model.pt).")
    ap.add_argument("--checkpoint", default=None, help="Path to a checkpoint (.pt). If omitted, uses <model_dir>/best_model.pt.")
    ap.add_argument("--config_json", default=None, help="Path to config.json. If omitted, uses <model_dir>/config.json.")
    ap.add_argument("--device", default="cuda:0")

    # Subject selection
    ap.add_argument("--subject_ids_txt", default=None, help="Text file with one subject_id per line.")
    ap.add_argument("--split_json", default=None, help="Split JSON with train_ids/val_ids (data_parameters.json).")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--all_cache", action="store_true", help="Predict all cache/*.npz subjects.")

    # Output
    ap.add_argument("--out_dir", default=None, help="Where to write per-subject .npz predictions (default: <model_dir>/val_predictions).")
    args = ap.parse_args(argv)

    cache_root = _resolve_path(args.cache_root)
    cache_dir = cache_root / "cache"
    adj_path = cache_root / "adjacency.npy"
    if not cache_dir.is_dir() or not adj_path.is_file():
        raise FileNotFoundError(f"Missing cache_root artifacts: {cache_root} (need cache/ and adjacency.npy)")

    model_dir = _resolve_path(args.model_dir) if args.model_dir else None
    config_path = _resolve_path(args.config_json) if args.config_json else (model_dir / "config.json" if model_dir else None)
    ckpt_path = _resolve_path(args.checkpoint) if args.checkpoint else (model_dir / "best_model.pt" if model_dir else None)
    if config_path is None or ckpt_path is None:
        raise ValueError("Provide --model_dir or both --checkpoint and --config_json.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config_json: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Subject ids
    if args.subject_ids_txt:
        subject_ids = _load_ids_from_txt(_resolve_path(args.subject_ids_txt))
    elif args.split_json:
        subject_ids = _load_ids_from_split_json(_resolve_path(args.split_json), str(args.split))
    elif args.all_cache:
        subject_ids = _discover_ids_from_cache(cache_dir)
    else:
        raise ValueError("Provide --subject_ids_txt, --split_json (+--split), or --all_cache.")

    if not subject_ids:
        raise ValueError("No subject_ids selected.")

    out_dir = _resolve_path(args.out_dir) if args.out_dir else (_resolve_path(model_dir) / "val_predictions" if model_dir else None)
    if out_dir is None:
        raise ValueError("Could not infer out_dir. Provide --out_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Determine dims from first cache.
    X0, L0, _y0 = _load_cache_npz(cache_dir, subject_ids[0])
    n_nodes = int(X0.shape[0])
    in_features = int(X0.shape[1])
    local_dim = int(L0.shape[1]) if L0 is not None else None

    model, use_local = _build_model_from_train_config(
        config=config,
        n_nodes=n_nodes,
        in_features=in_features,
        local_dim=local_dim,
        cache_root=cache_root,
    )

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    adj = torch.from_numpy(np.load(adj_path).astype(np.float32)).to(device)

    wrote = 0
    with torch.no_grad():
        for sid in subject_ids:
            X_np, L_np, y_np = _load_cache_npz(cache_dir, str(sid))
            X = torch.from_numpy(X_np).to(device)
            if use_local:
                if L_np is None:
                    raise ValueError(f"Missing L_loc for subject={sid} but model requires local features.")
                L = torch.from_numpy(L_np).to(device)
                logits, _bias = model(X, adj, L)
            else:
                logits, _bias = model(X, adj)
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)

            out_npz = out_dir / f"{sid}.npz"
            if y_np is None:
                np.savez_compressed(out_npz, p=p)
            else:
                np.savez_compressed(out_npz, p=p, y=y_np.astype(np.uint8))
            wrote += 1

    print(f"ok: wrote {wrote} subjects -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

