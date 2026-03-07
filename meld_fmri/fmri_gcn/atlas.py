from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_label
from scipy.spatial import cKDTree

from meld_graph.paths import MELD_PARAMS_PATH, NVERT


@dataclass(frozen=True)
class AtlasMapping:
    name: str
    parcel_names: List[str]  # canonical order (no hemi prefix), length N
    fsaverage_sym_lh_parcel_index: np.ndarray  # (NVERT,) int32 in [-1..N-1], canonical LH vertices
    fslr32k_lh_parcel_index: np.ndarray  # (32492,) int32 in [-1..N-1]
    fslr32k_rh_parcel_index: np.ndarray  # (32492,) int32 in [-1..N-1]
    parcel_centroids_fsavg_sym_lh: np.ndarray  # (N,3) float32
    cortex_mask_fsavg_sym_lh: np.ndarray  # (NVERT,) bool


def _as_abs(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_fsaverage_sym_sphere_coords(hemi: str) -> np.ndarray:
    path = Path(MELD_PARAMS_PATH) / "fsaverage_sym" / "surf" / f"{hemi}.sphere"
    coords, _faces = nib.freesurfer.read_geometry(str(path))
    return coords.astype(np.float32)


def _load_gifti_sphere_coords(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI sphere surface: {path}")
    return np.asarray(img.darrays[0].data, dtype=np.float32)


def _build_nn_index(src_coords: np.ndarray, dst_coords: np.ndarray) -> np.ndarray:
    tree = cKDTree(src_coords.astype(np.float64))
    _d, idx = tree.query(dst_coords.astype(np.float64), k=1)
    return idx.astype(np.int32)


def _load_fsaverage_sym_cortex_mask(hemi: str) -> np.ndarray:
    label_path = Path(MELD_PARAMS_PATH) / "fsaverage_sym" / "label" / f"{hemi}.cortex.label"
    idx = np.sort(read_label(str(label_path))).astype(int)
    mask = np.zeros(NVERT, dtype=bool)
    mask[idx] = True
    return mask


def _read_annot(annot_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
    names2 = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
    return labels.astype(np.int32), ctab.astype(np.int32), names2


def _code_to_name(ctab: np.ndarray, names: List[str]) -> Dict[int, str]:
    # nibabel stores the code in the last column of ctab
    return {int(ctab[i, -1]): str(names[i]) for i in range(len(names))}

def _read_gifti_label(path: Path) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Read a GIFTI label file (*.label.gii).

    Returns:
      labels: (NVERT,) int32
      code_to_name: mapping from integer label value -> label name
    """
    img = nib.load(str(path))
    if not isinstance(img, nib.gifti.GiftiImage):
        raise ValueError(f"Expected GIFTI label file: {path}")
    if len(img.darrays) != 1:
        raise ValueError(f"Expected 1 data array in label.gii: {path} (got {len(img.darrays)})")
    labels = np.asarray(img.darrays[0].data, dtype=np.int32).reshape(-1)
    if labels.shape[0] != NVERT:
        raise ValueError(f"Unexpected label length={labels.shape[0]} for {path} (expected {NVERT})")
    # nibabel returns a dict-like label table (keys may be int or str depending on version)
    lt = img.labeltable.get_labels_as_dict()
    code_to_name: Dict[int, str] = {}
    for k, v in lt.items():
        try:
            code = int(k)
        except Exception:
            continue
        code_to_name[code] = str(v)
    return labels, code_to_name


def load_atlas_aparc(
    templateflow_fsLR_dir: str = "templateflow/tpl-fsLR",
    exclude_names: Tuple[str, ...] = ("unknown",),
) -> AtlasMapping:
    """
    v0 atlas for DeepEZ-style GCN: FreeSurfer aparc (Desikan-Killiany), N=34 per hemi.

    Rationale:
      - Present locally via `meld_params/fsaverage_sym` (no template downloads).
      - Symmetric names across hemispheres, enabling ipsi/contra re-ordering.
    """
    project_root = Path(__file__).resolve().parents[2]
    tpl_dir = Path(templateflow_fsLR_dir)
    if not tpl_dir.is_absolute():
        tpl_dir = (project_root / tpl_dir).resolve()

    fsLR_L = tpl_dir / "tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii"
    fsLR_R = tpl_dir / "tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii"
    if not fsLR_L.is_file() or not fsLR_R.is_file():
        raise FileNotFoundError(f"Missing fsLR<->fsaverage spheres under: {tpl_dir}")

    # Read canonical annotation on fsaverage_sym (per hemi).
    lh_annot = Path(MELD_PARAMS_PATH) / "fsaverage_sym" / "label" / "lh.aparc.annot"
    rh_annot = Path(MELD_PARAMS_PATH) / "fsaverage_sym" / "label" / "rh.aparc.annot"
    lh_labels, lh_ctab, lh_names = _read_annot(lh_annot)
    rh_labels, rh_ctab, rh_names = _read_annot(rh_annot)

    lh_code2name = _code_to_name(lh_ctab, lh_names)
    rh_code2name = _code_to_name(rh_ctab, rh_names)

    # Canonical parcel order (no hemi prefix; consistent across hemis).
    parcel_names = [n for n in lh_names if n not in exclude_names]
    name2idx = {n: i for i, n in enumerate(parcel_names)}

    def labels_to_parcel_index(labels: np.ndarray, code2name: Dict[int, str]) -> np.ndarray:
        out = np.full(labels.shape[0], -1, dtype=np.int32)
        for code in np.unique(labels):
            name = code2name.get(int(code), None)
            if name is None or name in exclude_names:
                continue
            out[labels == code] = int(name2idx[name])
        return out

    fsavg_sym_lh_parcel_index = labels_to_parcel_index(lh_labels, lh_code2name)

    # Centroids on fsaverage_sym LH sphere (canonical geometry).
    coords_lh = _load_fsaverage_sym_sphere_coords("lh")
    centroids = np.zeros((len(parcel_names), 3), dtype=np.float32)
    for i in range(len(parcel_names)):
        m = fsavg_sym_lh_parcel_index == i
        if not np.any(m):
            continue
        centroids[i] = np.mean(coords_lh[m], axis=0).astype(np.float32)

    cortex_mask_lh = _load_fsaverage_sym_cortex_mask("lh")

    # Map fsaverage_sym parcels -> fsLR32k parcels by nearest-neighbour on sphere coords.
    fsavg_sym_coords_lh = coords_lh
    fsavg_sym_coords_rh = _load_fsaverage_sym_sphere_coords("rh")
    fslr_coords_lh = _load_gifti_sphere_coords(fsLR_L)
    fslr_coords_rh = _load_gifti_sphere_coords(fsLR_R)

    idx_lh = _build_nn_index(fsavg_sym_coords_lh, fslr_coords_lh)
    idx_rh = _build_nn_index(fsavg_sym_coords_rh, fslr_coords_rh)

    fslr_lh_codes = lh_labels[idx_lh]
    fslr_rh_codes = rh_labels[idx_rh]

    fslr32k_lh_parcel_index = labels_to_parcel_index(fslr_lh_codes, lh_code2name)
    fslr32k_rh_parcel_index = labels_to_parcel_index(fslr_rh_codes, rh_code2name)

    return AtlasMapping(
        name="aparc",
        parcel_names=parcel_names,
        fsaverage_sym_lh_parcel_index=fsavg_sym_lh_parcel_index,
        fslr32k_lh_parcel_index=fslr32k_lh_parcel_index,
        fslr32k_rh_parcel_index=fslr32k_rh_parcel_index,
        parcel_centroids_fsavg_sym_lh=centroids,
        cortex_mask_fsavg_sym_lh=cortex_mask_lh,
    )

def load_atlas_brainnetome(
    templateflow_fsaverage_dir: str = "templateflow/tpl-fsaverage",
    templateflow_fsLR_dir: str = "templateflow/tpl-fsLR",
    exclude_codes: Tuple[int, ...] = (0,),
) -> AtlasMapping:
    """
    Brainnetome (surface cortical) labels on fsaverage 164k, mapped to fsLR32k via spheres.

    Notes:
      - In the public repo, we ship a small, precomputed mapping under
        `meld_fmri/resources/atlas/brainnetome/` to avoid requiring a TemplateFlow checkout.
      - If the resource files are missing, we fall back to a TemplateFlow-based build.
    """
    # `.../meld_fmri/fmri_gcn/atlas.py` -> parents[1] is the `meld_fmri/` package dir.
    res_dir = Path(__file__).resolve().parents[1] / "resources" / "atlas" / "brainnetome"
    res_files = {
        "parcel_names": res_dir / "parcel_names.json",
        "fsavg_sym_lh": res_dir / "fsaverage_sym_lh_parcel_index.npy",
        "fslr32k_lh": res_dir / "fslr32k_lh_parcel_index.npy",
        "fslr32k_rh": res_dir / "fslr32k_rh_parcel_index.npy",
        "centroids": res_dir / "parcel_centroids_fsavg_sym_lh.npy",
        "cortex_mask": res_dir / "cortex_mask_fsavg_sym_lh.npy",
    }
    if all(p.is_file() for p in res_files.values()):
        parcel_names = json.loads(res_files["parcel_names"].read_text(encoding="utf-8"))
        fsavg_sym_lh_parcel_index = np.load(res_files["fsavg_sym_lh"]).astype(np.int32)
        fslr32k_lh_parcel_index = np.load(res_files["fslr32k_lh"]).astype(np.int32)
        fslr32k_rh_parcel_index = np.load(res_files["fslr32k_rh"]).astype(np.int32)
        centroids = np.load(res_files["centroids"]).astype(np.float32)
        cortex_mask_lh = np.load(res_files["cortex_mask"]).astype(bool)

        if fsavg_sym_lh_parcel_index.shape != (NVERT,):
            raise ValueError(
                f"Unexpected brainnetome resource shape: fsaverage_sym_lh_parcel_index={fsavg_sym_lh_parcel_index.shape} "
                f"(expected {(NVERT,)})"
            )
        if cortex_mask_lh.shape != (NVERT,):
            raise ValueError(
                f"Unexpected brainnetome resource shape: cortex_mask_fsavg_sym_lh={cortex_mask_lh.shape} (expected {(NVERT,)})"
            )

        return AtlasMapping(
            name="brainnetome",
            parcel_names=[str(x) for x in parcel_names],
            fsaverage_sym_lh_parcel_index=fsavg_sym_lh_parcel_index,
            fslr32k_lh_parcel_index=fslr32k_lh_parcel_index,
            fslr32k_rh_parcel_index=fslr32k_rh_parcel_index,
            parcel_centroids_fsavg_sym_lh=centroids,
            cortex_mask_fsavg_sym_lh=cortex_mask_lh,
        )

    project_root = Path(__file__).resolve().parents[2]

    tpl_fsavg = Path(templateflow_fsaverage_dir)
    if not tpl_fsavg.is_absolute():
        tpl_fsavg = (project_root / tpl_fsavg).resolve()
    tpl_fslr = Path(templateflow_fsLR_dir)
    if not tpl_fslr.is_absolute():
        tpl_fslr = (project_root / tpl_fslr).resolve()

    fsLR_L = tpl_fslr / "tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii"
    fsLR_R = tpl_fslr / "tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii"
    if not fsLR_L.is_file() or not fsLR_R.is_file():
        raise FileNotFoundError(f"Missing fsLR<->fsaverage spheres under: {tpl_fslr}")

    lh_gii = tpl_fsavg / "tpl-fsaverage_hemi-L_den-164k_atlas-brainnetome_dseg.label.gii"
    rh_gii = tpl_fsavg / "tpl-fsaverage_hemi-R_den-164k_atlas-brainnetome_dseg.label.gii"
    if not lh_gii.is_file() or not rh_gii.is_file():
        raise FileNotFoundError(f"Missing Brainnetome label files under: {tpl_fsavg}")

    lh_labels, lh_code2name = _read_gifti_label(lh_gii)
    rh_labels, rh_code2name = _read_gifti_label(rh_gii)

    # Canonical parcel order by integer code, excluding unknown.
    codes = sorted(set(lh_code2name.keys()) & set(rh_code2name.keys()))
    codes = [c for c in codes if c not in exclude_codes]
    parcel_names = [lh_code2name[c] for c in codes]
    code_to_idx = {int(code): int(i) for i, code in enumerate(codes)}

    fsavg_sym_lh_parcel_index = np.full((NVERT,), -1, dtype=np.int32)
    for code, idx in code_to_idx.items():
        fsavg_sym_lh_parcel_index[lh_labels == code] = idx

    # Centroids on fsaverage_sym LH sphere (canonical geometry).
    coords_lh = _load_fsaverage_sym_sphere_coords("lh")
    centroids = np.zeros((len(parcel_names), 3), dtype=np.float32)
    for i in range(len(parcel_names)):
        m = fsavg_sym_lh_parcel_index == i
        if not np.any(m):
            continue
        centroids[i] = np.mean(coords_lh[m], axis=0).astype(np.float32)

    cortex_mask_lh = _load_fsaverage_sym_cortex_mask("lh")

    # Map fsaverage_sym parcels -> fsLR32k parcels by nearest-neighbour on sphere coords.
    fsavg_sym_coords_lh = coords_lh
    fsavg_sym_coords_rh = _load_fsaverage_sym_sphere_coords("rh")
    fslr_coords_lh = _load_gifti_sphere_coords(fsLR_L)
    fslr_coords_rh = _load_gifti_sphere_coords(fsLR_R)

    idx_lh = _build_nn_index(fsavg_sym_coords_lh, fslr_coords_lh)
    idx_rh = _build_nn_index(fsavg_sym_coords_rh, fslr_coords_rh)

    fslr_lh_codes = lh_labels[idx_lh]
    fslr_rh_codes = rh_labels[idx_rh]

    fslr32k_lh_parcel_index = np.full((fslr_lh_codes.shape[0],), -1, dtype=np.int32)
    fslr32k_rh_parcel_index = np.full((fslr_rh_codes.shape[0],), -1, dtype=np.int32)
    for code, idx in code_to_idx.items():
        fslr32k_lh_parcel_index[fslr_lh_codes == code] = idx
        fslr32k_rh_parcel_index[fslr_rh_codes == code] = idx

    return AtlasMapping(
        name="brainnetome",
        parcel_names=parcel_names,
        fsaverage_sym_lh_parcel_index=fsavg_sym_lh_parcel_index,
        fslr32k_lh_parcel_index=fslr32k_lh_parcel_index,
        fslr32k_rh_parcel_index=fslr32k_rh_parcel_index,
        parcel_centroids_fsavg_sym_lh=centroids,
        cortex_mask_fsavg_sym_lh=cortex_mask_lh,
    )


def build_knn_adjacency(
    centroids: np.ndarray,
    k: int = 8,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build a simple symmetric kNN adjacency (dense float32, values in {0,1}).
    """
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids as (N,3), got {centroids.shape}")
    n = int(centroids.shape[0])
    if n == 0:
        raise ValueError("No parcels to build adjacency.")
    if k < 1:
        raise ValueError("k must be >= 1")

    # Pairwise squared distances
    diff = centroids[:, None, :] - centroids[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    # argsort each row
    nn = np.argsort(d2, axis=1)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neigh = nn[i, : (k + 1)]  # includes self at rank 0
        for j in neigh:
            if i == j and not include_self:
                continue
            adj[i, int(j)] = 1.0
            adj[int(j), i] = 1.0
    if include_self:
        np.fill_diagonal(adj, 1.0)
    return adj


def build_partial_corr_adjacency(
    fc_or_timeseries: np.ndarray,
    top_pct: float = 0.10,
    method: str = "ledoit_wolf",
    fallback_k: int = 3,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build a sparse adjacency matrix based on partial correlation.

    Motivation (BrainGNN):
    1. Avoid dense graph over-smoothing by keeping only top k% positive edges
    2. Use partial corr for edges + Pearson for node features = multi-graph fusion
       (fusing two different fMRI connectivity measures)

    Args:
        fc_or_timeseries: Either (N, N) FC matrix or (T, N) timeseries
        top_pct: Fraction of positive edges to keep (default 0.10 = 10%)
        method: "ledoit_wolf" (shrinkage) or "graphical_lasso" (sparse precision)
        fallback_k: For isolated nodes, connect to top-k strongest partial corr neighbors
        include_self: Whether to include self-loops

    Returns:
        adj: (N, N) float32 binary adjacency matrix

    References:
        - BrainGNN: https://pmc.ncbi.nlm.nih.gov/articles/PMC9916535/
    """
    from sklearn.covariance import LedoitWolf, GraphicalLassoCV

    # Handle input: timeseries (T, N) or FC matrix (N, N)
    if fc_or_timeseries.ndim == 2:
        if fc_or_timeseries.shape[0] == fc_or_timeseries.shape[1]:
            # Assume it's an FC matrix, estimate partial corr from it
            fc = fc_or_timeseries.astype(np.float64)
            n = fc.shape[0]
            # Use FC as covariance proxy and estimate precision
            if method == "ledoit_wolf":
                # Ledoit-Wolf shrinkage on FC treated as covariance
                model = LedoitWolf()
                model.covariance_ = fc
                # Estimate precision via pseudo-inverse
                precision = np.linalg.pinv(fc + 1e-6 * np.eye(n))
            else:
                # For graphical lasso, we need actual samples
                raise ValueError("graphical_lasso requires timeseries, not FC matrix")
        else:
            # Timeseries: (T, N)
            timeseries = fc_or_timeseries.T.astype(np.float64)  # (N, T)
            n = timeseries.shape[0]
            if method == "ledoit_wolf":
                model = LedoitWolf()
                model.fit(timeseries.T)  # fit expects (T, N)
                precision = model.precision_
            elif method == "graphical_lasso":
                model = GraphicalLassoCV(cv=3, max_iter=200)
                model.fit(timeseries.T)
                precision = model.precision_
            else:
                raise ValueError(f"Unknown method: {method}")
    else:
        raise ValueError(f"Expected 2D array, got shape {fc_or_timeseries.shape}")

    # Convert precision to partial correlation
    # partial_corr[i,j] = -P[i,j] / sqrt(P[i,i] * P[j,j])
    d = np.sqrt(np.diag(precision) + 1e-10)
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    # Keep only positive partial correlations
    partial_corr_pos = np.maximum(partial_corr, 0)
    np.fill_diagonal(partial_corr_pos, 0)  # Remove diagonal for threshold calculation

    # Compute threshold for top k%
    upper_tri = partial_corr_pos[np.triu_indices(n, k=1)]
    positive_vals = upper_tri[upper_tri > 0]
    if len(positive_vals) == 0:
        # No positive edges, use fallback kNN on partial_corr
        threshold = -np.inf
    else:
        threshold = np.percentile(positive_vals, 100 - top_pct * 100)

    # Create adjacency
    adj = np.zeros((n, n), dtype=np.float32)
    adj[partial_corr_pos >= threshold] = 1.0

    # Symmetrize
    adj = np.maximum(adj, adj.T)

    # Handle isolated nodes (degree = 0)
    degrees = np.sum(adj, axis=1)
    isolated = np.where(degrees == 0)[0]
    for i in isolated:
        # Connect to top-k strongest partial corr neighbors
        row = partial_corr_pos[i].copy()
        row[i] = -np.inf  # Exclude self
        top_neighbors = np.argsort(row)[::-1][:fallback_k]
        for j in top_neighbors:
            if row[j] > 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Self-loops
    if include_self:
        np.fill_diagonal(adj, 1.0)
    else:
        np.fill_diagonal(adj, 0.0)

    return adj


def build_group_fc_adjacency(
    fc_matrices: list,
    top_pct: float = 0.10,
    method: str = "knn",
    k: int = 8,
    fallback_k: int = 3,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build a shared adjacency matrix from group-level FC (cGCN style).

    This creates a stable, shared graph across all subjects by averaging
    FC matrices from training set.

    Args:
        fc_matrices: List of (N, N) FC matrices from training subjects
        top_pct: For top_pct method, fraction of positive edges to keep
        method: "knn" (top-k neighbors) or "top_pct" (top percentage)
        k: For knn method, number of neighbors per node
        fallback_k: For isolated nodes handling
        include_self: Whether to include self-loops

    Returns:
        adj: (N, N) float32 binary adjacency matrix

    References:
        - cGCN: https://pmc.ncbi.nlm.nih.gov/articles/PMC7935029/
    """
    if len(fc_matrices) == 0:
        raise ValueError("No FC matrices provided")

    # Compute group-level FC (mean across subjects)
    fc_group = np.mean(np.stack(fc_matrices, axis=0), axis=0).astype(np.float32)
    n = fc_group.shape[0]

    # Only consider positive correlations
    fc_pos = np.maximum(fc_group, 0)
    np.fill_diagonal(fc_pos, 0)

    if method == "knn":
        # Top-k neighbors per node
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            row = fc_pos[i].copy()
            top_neighbors = np.argsort(row)[::-1][:k]
            for j in top_neighbors:
                if row[j] > 0:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
    elif method == "top_pct":
        # Global top percentage
        upper_tri = fc_pos[np.triu_indices(n, k=1)]
        positive_vals = upper_tri[upper_tri > 0]
        if len(positive_vals) == 0:
            threshold = -np.inf
        else:
            threshold = np.percentile(positive_vals, 100 - top_pct * 100)
        adj = np.zeros((n, n), dtype=np.float32)
        adj[fc_pos >= threshold] = 1.0
        adj = np.maximum(adj, adj.T)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle isolated nodes
    degrees = np.sum(adj, axis=1)
    isolated = np.where(degrees == 0)[0]
    for i in isolated:
        row = fc_pos[i].copy()
        row[i] = -np.inf
        top_neighbors = np.argsort(row)[::-1][:fallback_k]
        for j in top_neighbors:
            if row[j] > 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Self-loops
    if include_self:
        np.fill_diagonal(adj, 1.0)
    else:
        np.fill_diagonal(adj, 0.0)

    return adj


def normalize_adjacency(adj: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected square adjacency, got {adj.shape}")
    a = adj.astype(np.float32)
    deg = np.sum(a, axis=1).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, eps))
    d = inv_sqrt[:, None] * inv_sqrt[None, :]
    return a * d


def load_yeo2011_17net_labels(
    templateflow_fsaverage_dir: str = "templateflow/tpl-fsaverage",
    density: str = "164k",
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load Yeo2011 17-network labels on fsaverage as GIFTI label files.

    Expected templateflow files (in this repo):
      - tpl-fsaverage_hemi-L_den-<density>_atlas-Yeo2011_seg-17n_dseg.label.gii
      - tpl-fsaverage_hemi-R_den-<density>_atlas-Yeo2011_seg-17n_dseg.label.gii

    Returns:
      lh_labels: (NVERT,) int32 in [0..17] where 0 is medial wall / unknown.
      rh_labels: (NVERT,) int32 in [0..17]
      code_to_name: dict label_value -> label_name
    """
    project_root = Path(__file__).resolve().parents[2]
    tpl_fsavg = Path(templateflow_fsaverage_dir)
    if not tpl_fsavg.is_absolute():
        tpl_fsavg = (project_root / tpl_fsavg).resolve()

    lh = tpl_fsavg / f"tpl-fsaverage_hemi-L_den-{density}_atlas-Yeo2011_seg-17n_dseg.label.gii"
    rh = tpl_fsavg / f"tpl-fsaverage_hemi-R_den-{density}_atlas-Yeo2011_seg-17n_dseg.label.gii"
    if not lh.is_file() or not rh.is_file():
        raise FileNotFoundError(f"Missing Yeo2011 17-network label files under: {tpl_fsavg}")

    lh_labels, lh_code2name = _read_gifti_label(lh)
    rh_labels, rh_code2name = _read_gifti_label(rh)

    if set(lh_code2name.keys()) != set(rh_code2name.keys()):
        raise ValueError("Yeo2011 label tables differ between hemispheres; expected identical codes.")
    return lh_labels, rh_labels, lh_code2name


def load_atlas_schaefer100(
    templateflow_fsaverage_dir: str = "templateflow/tpl-fsaverage",
    templateflow_fsLR_dir: str = "templateflow/tpl-fsLR",
    scale: int = 100,
    networks: str = "7n",
    exclude_codes: Tuple[int, ...] = (0,),
) -> AtlasMapping:
    """
    Schaefer 2018 cortical parcellation (100 parcels: 50 per hemisphere).

    This atlas is designed to match the BrainMass pretrained encoder which expects
    100 ROIs (n_rois=100).

    Args:
        templateflow_fsaverage_dir: Path to templateflow fsaverage directory
        templateflow_fsLR_dir: Path to templateflow fsLR directory
        scale: Number of parcels (100, 200, 300, etc.)
        networks: Network version ("7n" for Yeo 7-network, "17n" for Yeo 17-network)
        exclude_codes: Label codes to exclude (0 = background/medial wall)

    Returns:
        AtlasMapping with 50 parcels per hemisphere (100 total cortical ROIs)

    Notes:
        - Schaefer 100 = 50 parcels/hemi × 2 = 100 cortical ROIs
        - Label files use codes 0-50 (0=background, 1-50=parcels)
        - Parcel names follow pattern: "7Networks_LH_Vis_1", "7Networks_RH_Vis_1", etc.
    """
    project_root = Path(__file__).resolve().parents[2]

    tpl_fsavg = Path(templateflow_fsaverage_dir)
    if not tpl_fsavg.is_absolute():
        tpl_fsavg = (project_root / tpl_fsavg).resolve()
    tpl_fslr = Path(templateflow_fsLR_dir)
    if not tpl_fslr.is_absolute():
        tpl_fslr = (project_root / tpl_fslr).resolve()

    fsLR_L = tpl_fslr / "tpl-fsLR_space-fsaverage_hemi-L_den-32k_sphere.surf.gii"
    fsLR_R = tpl_fslr / "tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii"
    if not fsLR_L.is_file() or not fsLR_R.is_file():
        raise FileNotFoundError(f"Missing fsLR<->fsaverage spheres under: {tpl_fslr}")

    n_parcels_per_hemi = scale // 2  # 100 -> 50 per hemi

    lh_gii = tpl_fsavg / f"tpl-fsaverage_hemi-L_den-164k_atlas-Schaefer2018_seg-{networks}_scale-{scale}_dseg.label.gii"
    rh_gii = tpl_fsavg / f"tpl-fsaverage_hemi-R_den-164k_atlas-Schaefer2018_seg-{networks}_scale-{scale}_dseg.label.gii"
    if not lh_gii.is_file() or not rh_gii.is_file():
        raise FileNotFoundError(f"Missing Schaefer {scale} label files under: {tpl_fsavg}")

    lh_labels, lh_code2name = _read_gifti_label(lh_gii)
    rh_labels, rh_code2name = _read_gifti_label(rh_gii)

    # Build parcel names list: LH parcels first (0-49), then RH parcels (50-99)
    # LH codes: 1-50 -> indices 0-49
    # RH codes: 1-50 -> indices 50-99
    lh_codes = sorted([c for c in lh_code2name.keys() if c not in exclude_codes])
    rh_codes = sorted([c for c in rh_code2name.keys() if c not in exclude_codes])

    parcel_names: List[str] = []
    lh_code_to_idx: Dict[int, int] = {}
    rh_code_to_idx: Dict[int, int] = {}

    # LH parcels: indices 0 to n_parcels_per_hemi-1
    for i, code in enumerate(lh_codes):
        name = lh_code2name[code]
        parcel_names.append(name)
        lh_code_to_idx[code] = i

    # RH parcels: indices n_parcels_per_hemi to 2*n_parcels_per_hemi-1
    for i, code in enumerate(rh_codes):
        name = rh_code2name[code]
        parcel_names.append(name)
        rh_code_to_idx[code] = n_parcels_per_hemi + i

    # Build fsaverage_sym LH parcel index (only LH parcels, 0-49)
    fsavg_sym_lh_parcel_index = np.full((NVERT,), -1, dtype=np.int32)
    for code, idx in lh_code_to_idx.items():
        fsavg_sym_lh_parcel_index[lh_labels == code] = idx

    # Centroids on fsaverage_sym LH sphere (for LH parcels only)
    coords_lh = _load_fsaverage_sym_sphere_coords("lh")
    coords_rh = _load_fsaverage_sym_sphere_coords("rh")

    # Centroids for all 100 parcels (LH from LH coords, RH from RH coords)
    centroids = np.zeros((len(parcel_names), 3), dtype=np.float32)
    for code, idx in lh_code_to_idx.items():
        m = lh_labels == code
        if np.any(m):
            centroids[idx] = np.mean(coords_lh[m], axis=0).astype(np.float32)
    for code, idx in rh_code_to_idx.items():
        m = rh_labels == code
        if np.any(m):
            centroids[idx] = np.mean(coords_rh[m], axis=0).astype(np.float32)

    cortex_mask_lh = _load_fsaverage_sym_cortex_mask("lh")

    # Map fsaverage_sym parcels -> fsLR32k parcels by nearest-neighbour on sphere coords
    fslr_coords_lh = _load_gifti_sphere_coords(fsLR_L)
    fslr_coords_rh = _load_gifti_sphere_coords(fsLR_R)

    idx_lh = _build_nn_index(coords_lh, fslr_coords_lh)
    idx_rh = _build_nn_index(coords_rh, fslr_coords_rh)

    fslr_lh_codes = lh_labels[idx_lh]
    fslr_rh_codes = rh_labels[idx_rh]

    # fsLR32k parcel indices: LH uses 0-49, RH uses 50-99
    fslr32k_lh_parcel_index = np.full((fslr_lh_codes.shape[0],), -1, dtype=np.int32)
    fslr32k_rh_parcel_index = np.full((fslr_rh_codes.shape[0],), -1, dtype=np.int32)
    for code, idx in lh_code_to_idx.items():
        fslr32k_lh_parcel_index[fslr_lh_codes == code] = idx
    for code, idx in rh_code_to_idx.items():
        fslr32k_rh_parcel_index[fslr_rh_codes == code] = idx

    return AtlasMapping(
        name=f"schaefer{scale}_{networks}",
        parcel_names=parcel_names,
        fsaverage_sym_lh_parcel_index=fsavg_sym_lh_parcel_index,
        fslr32k_lh_parcel_index=fslr32k_lh_parcel_index,
        fslr32k_rh_parcel_index=fslr32k_rh_parcel_index,
        parcel_centroids_fsavg_sym_lh=centroids,
        cortex_mask_fsavg_sym_lh=cortex_mask_lh,
    )


def map_parcels_to_yeo17_networks(
    atlas: AtlasMapping,
    templateflow_fsaverage_dir: str = "templateflow/tpl-fsaverage",
    density: str = "164k",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a per-parcel Yeo17 network assignment on the fsaverage_sym canonical LH surface.

    This is used for DeepEZ-style node-feature reduction:
      - keep the graph nodes as Brainnetome parcels (and optional subcortex/midline)
      - reduce per-node FC row into Yeo17 network-aggregated features

    Returns:
      parcel_to_net: (N_parcels,) int32 in [0..17], 0=unknown/medial wall.
      purity: (N_parcels,) float32, fraction of parcel vertices in the winning network (ignoring label 0).
    """
    lh_labels, _rh_labels, _code2name = load_yeo2011_17net_labels(
        templateflow_fsaverage_dir=templateflow_fsaverage_dir,
        density=density,
    )

    parcel_index = atlas.fsaverage_sym_lh_parcel_index.astype(np.int32)
    cortex = atlas.cortex_mask_fsavg_sym_lh.astype(bool)

    n_parcels = int(len(atlas.parcel_names))
    out = np.zeros((n_parcels,), dtype=np.int32)
    purity = np.zeros((n_parcels,), dtype=np.float32)

    for p in range(n_parcels):
        m = (parcel_index == int(p)) & cortex
        if not np.any(m):
            continue
        labs = lh_labels[m].astype(np.int32)
        labs = labs[labs != 0]
        if labs.size == 0:
            continue
        counts = np.bincount(labs, minlength=18).astype(np.int64)
        net = int(np.argmax(counts[1:])) + 1
        out[p] = net
        purity[p] = float(counts[net]) / float(labs.size)

    return out, purity
