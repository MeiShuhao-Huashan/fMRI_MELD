#!/usr/bin/env python3
"""
Three-Level Evaluation Module for MELD_fMRI FCD Detection.

This module provides comprehensive evaluation at three levels:
1. Vertex-level: DSC, IoU, PPV, TPR, PR/ROC curves
2. Cluster-level: TP/FP with multiple criteria (boxDSC, overlap, pinpointing), FROC curves
3. Subject-level: Detection rate, Specificity, Pinpointing rate

Reference: docs/evaluation/fcd_evaluation_metrics_three_levels.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse.csgraph
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import MELD_DATA_PATH, MELD_PARAMS_PATH, NVERT

logger = logging.getLogger("meld_fmri.three_level_eval")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VertexMetrics:
    """Vertex-level metrics for a single subject."""
    subject_id: str
    hemi: str
    split: str  # Added: data split (val/test/train)
    lesion_area_cm2: float
    n_pred_vertices: int
    n_lesion_vertices: int
    tp: int
    fp: int
    fn: int
    tn: int
    threshold: float = 0.5  # Added: prediction threshold used

    @property
    def dice(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2.0 * self.tp / denom) if denom > 0 else 0.0

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return (self.tp / denom) if denom > 0 else 0.0

    @property
    def ppv(self) -> float:
        """Positive Predictive Value (Precision)."""
        denom = self.tp + self.fp
        return (self.tp / denom) if denom > 0 else 0.0

    @property
    def tpr(self) -> float:
        """True Positive Rate (Recall/Sensitivity)."""
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else 0.0

    @property
    def tnr(self) -> float:
        """True Negative Rate (Specificity)."""
        denom = self.tn + self.fp
        return (self.tn / denom) if denom > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "hemi": self.hemi,
            "split": self.split,
            "lesion_area_cm2": self.lesion_area_cm2,
            "n_pred_vertices": self.n_pred_vertices,
            "n_lesion_vertices": self.n_lesion_vertices,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "DSC": self.dice,
            "IoU": self.iou,
            "PPV": self.ppv,
            "TPR": self.tpr,
            "TNR": self.tnr,
            "threshold": self.threshold,
        }


@dataclass
class ClusterMetrics:
    """Cluster-level metrics for a single predicted cluster."""
    subject_id: str
    hemi: str
    split: str  # Added: data split (val/test/train)
    cluster_id: int
    mean_score: float
    max_score: float
    area_cm2: float
    n_vertices: int
    tp_verts: int
    fp_verts: int
    fn_verts: int
    ppv_rand_expected: float = 0.0  # Expected PPV under random patch (same hemi)
    # TP criteria results
    box_dice: float = 0.0
    has_overlap: bool = False
    is_pinpointing: bool = False
    distance_to_lesion_mm: float = float("inf")
    # Vertex indices (optional, for visualization)
    vertex_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    @property
    def dice(self) -> float:
        denom = 2 * self.tp_verts + self.fp_verts + self.fn_verts
        return (2.0 * self.tp_verts / denom) if denom > 0 else 0.0

    @property
    def iou(self) -> float:
        denom = self.tp_verts + self.fp_verts + self.fn_verts
        return (self.tp_verts / denom) if denom > 0 else 0.0

    @property
    def ppv_in_mask(self) -> float:
        """Containment / PPV-in-mask = |pred∩GT| / |pred|."""
        denom = self.tp_verts + self.fp_verts
        return (self.tp_verts / denom) if denom > 0 else 0.0

    @property
    def ppv_lift(self) -> float:
        """Lift over random baseline on the same hemi: PPV / E[PPV_rand]."""
        if self.ppv_rand_expected <= 0:
            return float("nan")
        return self.ppv_in_mask / self.ppv_rand_expected

    @property
    def is_tp_boxdsc(self) -> bool:
        """TP by boxDSC > 0.22 criterion (Kersting paper)."""
        return self.box_dice > 0.22

    @property
    def is_tp_overlap(self) -> bool:
        """TP by >=1 voxel overlap criterion."""
        return self.has_overlap

    @property
    def is_tp_distance(self) -> bool:
        """TP by distance <= 20mm criterion (MELD Graph paper)."""
        return self.distance_to_lesion_mm <= 20.0

    @property
    def is_tp_ppv50(self) -> bool:
        """TP by containment / PPV-in-mask >= 0.5 criterion."""
        return self.ppv_in_mask >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "hemi": self.hemi,
            "split": self.split,
            "cluster_id": self.cluster_id,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "area_cm2": self.area_cm2,
            "n_vertices": self.n_vertices,
            "tp_verts": self.tp_verts,
            "fp_verts": self.fp_verts,
            "fn_verts": self.fn_verts,
            "dice": self.dice,
            "iou": self.iou,
            "ppv_in_mask": self.ppv_in_mask,
            "ppv_rand_expected": self.ppv_rand_expected,
            "ppv_lift": self.ppv_lift,
            "box_dice": self.box_dice,
            "has_overlap": self.has_overlap,
            "is_pinpointing": self.is_pinpointing,
            "distance_to_lesion_mm": self.distance_to_lesion_mm,
            "is_tp_boxdsc": self.is_tp_boxdsc,
            "is_tp_overlap": self.is_tp_overlap,
            "is_tp_distance": self.is_tp_distance,
            "is_tp_ppv50": self.is_tp_ppv50,
        }


@dataclass
class SubjectMetrics:
    """Subject-level metrics for detection evaluation."""
    subject_id: str
    split: str  # Added: data split (val/test/train)
    is_lesion: bool
    lesion_area_cm2: float
    n_clusters: int
    n_tp_clusters_boxdsc: int
    n_tp_clusters_overlap: int
    n_tp_clusters_distance: int
    n_tp_clusters_ppv50: int
    n_fp_clusters: int
    lesion_dice_union: float
    max_box_dice: float
    max_ppv_in_mask: float
    # Detection flags
    detected_boxdsc: bool = False
    detected_overlap: bool = False
    detected_distance: bool = False
    detected_ppv50: bool = False
    pinpointed: bool = False
    # Top-1 hit rate (most confident cluster)
    top1_hit_boxdsc: bool = False
    top1_hit_overlap: bool = False
    top1_hit_ppv50: bool = False
    # Additional info
    best_cluster_id: int = -1
    lesion_hemi: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "split": self.split,
            "is_lesion": self.is_lesion,
            "lesion_area_cm2": self.lesion_area_cm2,
            "lesion_hemi": self.lesion_hemi,
            "n_clusters": self.n_clusters,
            "n_tp_clusters_boxdsc": self.n_tp_clusters_boxdsc,
            "n_tp_clusters_overlap": self.n_tp_clusters_overlap,
            "n_tp_clusters_distance": self.n_tp_clusters_distance,
            "n_tp_clusters_ppv50": self.n_tp_clusters_ppv50,
            "n_fp_clusters": self.n_fp_clusters,
            "lesion_dice_union": self.lesion_dice_union,
            "max_box_dice": self.max_box_dice,
            "max_ppv_in_mask": self.max_ppv_in_mask,
            "detected_boxdsc": self.detected_boxdsc,
            "detected_overlap": self.detected_overlap,
            "detected_distance": self.detected_distance,
            "detected_ppv50": self.detected_ppv50,
            "pinpointed": self.pinpointed,
            "top1_hit_boxdsc": self.top1_hit_boxdsc,
            "top1_hit_overlap": self.top1_hit_overlap,
            "top1_hit_ppv50": self.top1_hit_ppv50,
            "best_cluster_id": self.best_cluster_id,
        }


# =============================================================================
# Core Computation Functions
# =============================================================================

def compute_box_dice(cluster_mask: np.ndarray, label_mask: np.ndarray,
                     coords: np.ndarray) -> float:
    """
    Compute bounding box Dice between predicted cluster and ground truth lesion.

    Args:
        cluster_mask: Boolean mask for cluster vertices (cortex-only)
        label_mask: Boolean mask for lesion vertices (cortex-only)
        coords: Vertex coordinates (N_cortex, 3)

    Returns:
        boxDSC: Bounding box Dice coefficient
    """
    if not np.any(cluster_mask) or not np.any(label_mask):
        return 0.0

    cluster_coords = coords[cluster_mask]
    label_coords = coords[label_mask]

    # Axis-aligned bounding boxes
    box_pred_min = cluster_coords.min(axis=0)
    box_pred_max = cluster_coords.max(axis=0)
    box_gt_min = label_coords.min(axis=0)
    box_gt_max = label_coords.max(axis=0)

    # Intersection volume
    inter_min = np.maximum(box_pred_min, box_gt_min)
    inter_max = np.minimum(box_pred_max, box_gt_max)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_size)

    # Individual volumes
    pred_vol = np.prod(box_pred_max - box_pred_min)
    gt_vol = np.prod(box_gt_max - box_gt_min)

    denom = pred_vol + gt_vol
    if denom <= 0:
        return 0.0
    return float(2 * inter_vol / denom)


def compute_cluster_com(cluster_mask: np.ndarray, coords: np.ndarray,
                        weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute center of mass of a cluster.

    Args:
        cluster_mask: Boolean mask for cluster vertices
        coords: Vertex coordinates (N, 3)
        weights: Optional weights for each vertex (e.g., prediction scores)

    Returns:
        3D coordinates of the center of mass
    """
    if not np.any(cluster_mask):
        return np.array([0.0, 0.0, 0.0])

    cluster_coords = coords[cluster_mask]
    if weights is not None:
        cluster_weights = weights[cluster_mask]
        # Normalize weights
        w_sum = cluster_weights.sum()
        if w_sum > 0:
            return np.average(cluster_coords, axis=0, weights=cluster_weights)
    return cluster_coords.mean(axis=0)


def check_pinpointing(cluster_mask: np.ndarray, label_mask: np.ndarray,
                      coords: np.ndarray, weights: Optional[np.ndarray] = None) -> bool:
    """
    Check if cluster's center of mass falls within the lesion (Kersting et al. definition).

    For surface data: compute COM, find nearest vertex, check if it's in lesion.

    Args:
        cluster_mask: Boolean mask for cluster vertices
        label_mask: Boolean mask for lesion vertices
        coords: Vertex coordinates (N, 3) - should be real anatomical coords (e.g., white surface)
        weights: Optional prediction scores for weighted COM

    Returns:
        True if COM is inside lesion (nearest vertex to COM is a lesion vertex)
    """
    if not np.any(cluster_mask) or not np.any(label_mask):
        return False

    # Compute (weighted) center of mass
    com = compute_cluster_com(cluster_mask, coords, weights)

    # Find the nearest vertex to the COM
    distances = np.linalg.norm(coords - com, axis=1)
    nearest_vertex = np.argmin(distances)

    # Check if the nearest vertex is in the lesion mask
    return bool(label_mask[nearest_vertex])


def compute_distance_to_lesion(cluster_mask: np.ndarray, label_mask: np.ndarray,
                               coords: np.ndarray) -> float:
    """
    Compute minimum Euclidean distance from cluster to lesion (in mm).

    For surface data, this is a reasonable approximation of geodesic distance
    for nearby clusters.
    """
    if not np.any(cluster_mask) or not np.any(label_mask):
        return float("inf")

    if np.any(cluster_mask & label_mask):
        # Overlapping - distance is 0
        return 0.0

    cluster_coords = coords[cluster_mask]
    label_coords = coords[label_mask]

    # Compute pairwise distances (this can be slow for large clusters)
    # Use a more efficient approach: find min distance per cluster vertex
    min_dist = float("inf")
    for cc in cluster_coords:
        dists = np.linalg.norm(label_coords - cc, axis=1)
        d = dists.min()
        if d < min_dist:
            min_dist = d

    return float(min_dist)


def cluster_connected_components(mask: np.ndarray, adj_mat: scipy.sparse.spmatrix,
                                 min_vertices: int = 100) -> np.ndarray:
    """
    Find connected components (clusters) in a binary mask using adjacency matrix.

    Args:
        mask: Boolean mask (cortex-only vertices)
        adj_mat: Sparse adjacency matrix (cortex-only)
        min_vertices: Minimum cluster size to keep

    Returns:
        Cluster labels (0 = background, 1+ = cluster IDs)
    """
    if mask.sum() == 0:
        return np.zeros(len(mask), dtype=np.int32)

    # Get subgraph of masked vertices
    n_comp, labels = scipy.sparse.csgraph.connected_components(adj_mat[mask][:, mask])

    # Assign cluster IDs
    clusters = np.zeros(len(mask), dtype=np.int32)
    cluster_id = 0
    for comp_idx in range(n_comp):
        include = labels == comp_idx
        if include.sum() < min_vertices:
            continue
        cluster_id += 1
        comp_mask = mask.copy()
        comp_mask[mask] = include
        clusters[comp_mask] = cluster_id

    return clusters


# =============================================================================
# Main Evaluator Class
# =============================================================================

class ThreeLevelEvaluator:
    """
    Three-level evaluation for FCD detection models.

    Evaluates at:
    1. Vertex-level: pixel-wise segmentation quality
    2. Cluster-level: per-cluster TP/FP with multiple criteria
    3. Subject-level: case-level detection and localization
    """

    def __init__(
        self,
        predictions_hdf5: Union[str, Path],
        cohort: MeldCohort,
        subject_ids: Optional[List[str]] = None,
        labels_hdf5: Optional[Union[str, Path]] = None,
        lesion_file_root: Optional[Union[str, Path]] = None,
        lesion_feature: str = "lesion",
        coords: Optional[np.ndarray] = None,
        min_cluster_vertices: int = 100,
        clustered_key: str = "prediction_clustered",
        split: str = "val",  # Added: data split
        model_name: str = "",  # Added: model name for summary
    ):
        """
        Initialize the evaluator.

        Args:
            predictions_hdf5: Path to predictions HDF5 file
            cohort: MeldCohort instance
            subject_ids: List of subject IDs to evaluate (default: all in predictions)
            labels_hdf5: Path to labels HDF5 (if different from cohort)
            lesion_file_root: Path to derived lesion labels directory
            lesion_feature: Feature name for lesion labels (default: "lesion")
            coords: Vertex coordinates for boxDSC/distance computation
            min_cluster_vertices: Minimum cluster size
            clustered_key: Dataset key for clustered predictions within each hemi group
            split: Data split name (val/test/train) for output
            model_name: Model name for summary output
        """
        self.predictions_path = Path(predictions_hdf5)
        self.cohort = cohort
        self.cortex_mask = cohort.cortex_mask.astype(bool)
        self.surf_area = cohort.surf_area[self.cortex_mask]  # mm^2
        self.hemi_cortex_area_cm2 = float(np.sum(self.surf_area)) / 100.0
        self.adj_mat = cohort.adj_mat[self.cortex_mask][:, self.cortex_mask]

        self.labels_hdf5 = Path(labels_hdf5) if labels_hdf5 else None
        self.lesion_file_root = Path(lesion_file_root) if lesion_file_root else None
        self.lesion_feature = lesion_feature
        self.min_cluster_vertices = min_cluster_vertices
        self.clustered_key = clustered_key
        self.split = split  # Added
        self.model_name = model_name  # Added

        # Load vertex coordinates for boxDSC computation
        if coords is not None:
            self.coords = coords[self.cortex_mask]
        else:
            self.coords = self._load_default_coords()

        # Discover subject IDs
        self.subject_ids = subject_ids or self._discover_subject_ids()

        # Results storage
        self.vertex_results: List[VertexMetrics] = []
        self.cluster_results: List[ClusterMetrics] = []
        self.subject_results: List[SubjectMetrics] = []

        # Raw data for PR/ROC curves (accumulated during evaluation)
        self._all_predictions: List[np.ndarray] = []
        self._all_labels: List[np.ndarray] = []

    def _load_default_coords(self) -> np.ndarray:
        """Load fsaverage_sym surface coordinates (white matter surface for accurate distances)."""
        import nibabel as nib
        # Use white surface (real anatomical coordinates) instead of sphere
        # This is critical for accurate distance calculations (boxDSC, pinpointing)
        surf_path = os.path.join(MELD_PARAMS_PATH, "fsaverage_sym", "surf", "lh.white")
        if os.path.exists(surf_path):
            coords, _ = nib.freesurfer.read_geometry(surf_path)
            # Use only cortex vertices
            return coords[self.cortex_mask]
        else:
            # Fallback to inflated if white not available
            surf_path_inflated = os.path.join(MELD_PARAMS_PATH, "fsaverage_sym", "surf", "lh.inflated")
            if os.path.exists(surf_path_inflated):
                coords, _ = nib.freesurfer.read_geometry(surf_path_inflated)
                return coords[self.cortex_mask]
            logger.warning("Could not load surface coordinates, using zeros")
            return np.zeros((self.cortex_mask.sum(), 3))

    def _discover_subject_ids(self) -> List[str]:
        """Discover subject IDs from predictions file."""
        subject_ids = []
        with h5py.File(self.predictions_path, "r") as f:
            for key in f.keys():
                try:
                    if "lh" in f[key] and "rh" in f[key]:
                        subject_ids.append(str(key))
                except (TypeError, KeyError):
                    continue
        return sorted(subject_ids)

    def _load_predictions(self, f: h5py.File, subject_id: str, hemi: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load prediction and clustered prediction from HDF5."""
        pred = np.asarray(f[subject_id][hemi]["prediction"][:], dtype=np.float32).reshape(-1)
        if self.clustered_key in f[subject_id][hemi]:
            clustered = np.asarray(f[subject_id][hemi][self.clustered_key][:], dtype=np.int32).reshape(-1)
        else:
            clustered = np.zeros_like(pred, dtype=np.int32)
        return pred, clustered

    def _load_label(self, subject_id: str, hemi: str) -> np.ndarray:
        """Load lesion label for a subject/hemi."""
        label = np.zeros(NVERT, dtype=np.float32)

        # Try derived labels first
        if self.lesion_file_root is not None:
            mgh_path = self.lesion_file_root / subject_id / f"{hemi}.on_lh.{self.lesion_feature}.mgh"
            if mgh_path.exists():
                import nibabel as nib
                img = nib.load(str(mgh_path))
                label = np.asarray(img.dataobj).astype(np.float32).reshape(-1)
                return (label[self.cortex_mask] > 0.5).astype(np.float32)

        # Try HDF5
        if self.labels_hdf5 is not None and self.labels_hdf5.exists():
            with h5py.File(self.labels_hdf5, "r") as f:
                try:
                    label = np.asarray(f[subject_id][hemi][f".on_lh.{self.lesion_feature}.mgh"][:]).reshape(-1)
                except (KeyError, TypeError):
                    pass

        return (label[self.cortex_mask] > 0.5).astype(np.float32) if len(label) == NVERT else np.zeros(self.cortex_mask.sum())

    def evaluate_all(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run full three-level evaluation on all subjects.

        Returns:
            Dictionary containing summary statistics
        """
        logger.info(f"Starting three-level evaluation for {len(self.subject_ids)} subjects")

        with h5py.File(self.predictions_path, "r") as f_pred:
            for i, subject_id in enumerate(self.subject_ids):
                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"  Processing {i + 1}/{len(self.subject_ids)}: {subject_id}")

                self._evaluate_subject(f_pred, subject_id)

        # Generate summary
        summary = self.generate_summary()
        logger.info("Evaluation complete")
        return summary

    def _evaluate_subject(self, f_pred: h5py.File, subject_id: str) -> None:
        """Evaluate a single subject at all three levels."""
        # Aggregate across hemispheres
        all_clusters: List[ClusterMetrics] = []
        vertex_metrics_list: List[VertexMetrics] = []
        total_lesion_area = 0.0
        lesion_hemi = ""

        for hemi in ["lh", "rh"]:
            try:
                pred, clustered = self._load_predictions(f_pred, subject_id, hemi)
                label = self._load_label(subject_id, hemi)
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to load data for {subject_id}/{hemi}: {e}")
                continue

            # Store raw data for PR/ROC curves (only for lesion hemispheres)
            if label.sum() > 0:
                self._all_predictions.append(pred)
                self._all_labels.append(label)

            # Vertex-level metrics
            pred_binary = (pred > 0.5).astype(bool)
            label_binary = label > 0.5
            tp = int(np.sum(pred_binary & label_binary))
            fp = int(np.sum(pred_binary & ~label_binary))
            fn = int(np.sum(~pred_binary & label_binary))
            tn = int(np.sum(~pred_binary & ~label_binary))
            lesion_area = float(np.sum(self.surf_area[label_binary])) / 100.0  # cm^2

            vm = VertexMetrics(
                subject_id=subject_id,
                hemi=hemi,
                split=self.split,  # Added
                lesion_area_cm2=lesion_area,
                n_pred_vertices=int(pred_binary.sum()),
                n_lesion_vertices=int(label_binary.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn,
            )
            vertex_metrics_list.append(vm)
            self.vertex_results.append(vm)
            total_lesion_area += lesion_area

            if label_binary.sum() > 0:
                lesion_hemi = hemi

            # Cluster-level metrics
            cluster_ids = np.unique(clustered[clustered > 0])
            for cid in cluster_ids:
                cluster_mask = clustered == cid
                cm = self._compute_cluster_metrics(
                    subject_id, hemi, int(cid), pred, cluster_mask, label_binary
                )
                all_clusters.append(cm)
                self.cluster_results.append(cm)

        # Subject-level metrics
        is_lesion = total_lesion_area > 0
        n_clusters = len(all_clusters)
        n_tp_boxdsc = sum(1 for c in all_clusters if c.is_tp_boxdsc)
        n_tp_overlap = sum(1 for c in all_clusters if c.is_tp_overlap)
        n_tp_distance = sum(1 for c in all_clusters if c.is_tp_distance)
        n_tp_ppv50 = sum(1 for c in all_clusters if c.is_tp_ppv50)
        n_fp = n_clusters - n_tp_overlap  # FP = not overlapping with lesion

        # Dice for union of all clusters
        lesion_dice_union = self._compute_union_dice(f_pred, subject_id)
        max_box_dice = max([c.box_dice for c in all_clusters], default=0.0)
        max_ppv_in_mask = max([c.ppv_in_mask for c in all_clusters], default=0.0)
        pinpointed = any(c.is_pinpointing for c in all_clusters)

        # Top-1 hit rate: check if the most confident cluster (highest mean_score) is TP
        top1_hit_boxdsc = False
        top1_hit_overlap = False
        top1_hit_ppv50 = False
        best_cluster_id = -1
        if all_clusters:
            # Sort by mean_score (confidence) descending
            sorted_clusters = sorted(all_clusters, key=lambda c: c.mean_score, reverse=True)
            top1_cluster = sorted_clusters[0]
            best_cluster_id = top1_cluster.cluster_id
            top1_hit_boxdsc = top1_cluster.is_tp_boxdsc
            top1_hit_overlap = top1_cluster.is_tp_overlap
            top1_hit_ppv50 = top1_cluster.is_tp_ppv50

        sm = SubjectMetrics(
            subject_id=subject_id,
            split=self.split,  # Added
            is_lesion=is_lesion,
            lesion_area_cm2=total_lesion_area,
            n_clusters=n_clusters,
            n_tp_clusters_boxdsc=n_tp_boxdsc,
            n_tp_clusters_overlap=n_tp_overlap,
            n_tp_clusters_distance=n_tp_distance,
            n_tp_clusters_ppv50=n_tp_ppv50,
            n_fp_clusters=n_fp,
            lesion_dice_union=lesion_dice_union,
            max_box_dice=max_box_dice,
            max_ppv_in_mask=max_ppv_in_mask,
            detected_boxdsc=n_tp_boxdsc > 0,
            detected_overlap=n_tp_overlap > 0,
            detected_distance=n_tp_distance > 0,
            detected_ppv50=n_tp_ppv50 > 0,
            pinpointed=pinpointed,
            top1_hit_boxdsc=top1_hit_boxdsc,
            top1_hit_overlap=top1_hit_overlap,
            top1_hit_ppv50=top1_hit_ppv50,
            best_cluster_id=best_cluster_id,
            lesion_hemi=lesion_hemi,
        )
        self.subject_results.append(sm)

    def _compute_cluster_metrics(
        self, subject_id: str, hemi: str, cluster_id: int,
        pred: np.ndarray, cluster_mask: np.ndarray, label: np.ndarray
    ) -> ClusterMetrics:
        """Compute metrics for a single cluster."""
        cluster_scores = pred[cluster_mask]
        mean_score = float(np.mean(cluster_scores)) if len(cluster_scores) > 0 else 0.0
        max_score = float(np.max(cluster_scores)) if len(cluster_scores) > 0 else 0.0
        area_cm2 = float(np.sum(self.surf_area[cluster_mask])) / 100.0
        n_vertices = int(cluster_mask.sum())

        tp_verts = int(np.sum(cluster_mask & label))
        fp_verts = int(np.sum(cluster_mask & ~label))
        fn_verts = int(np.sum(~cluster_mask & label))

        lesion_area_cm2_hemi = float(np.sum(self.surf_area[label])) / 100.0 if np.any(label) else 0.0
        ppv_rand_expected = (
            lesion_area_cm2_hemi / self.hemi_cortex_area_cm2 if self.hemi_cortex_area_cm2 > 0 else 0.0
        )

        has_overlap = tp_verts > 0
        box_dice = compute_box_dice(cluster_mask, label, self.coords) if has_overlap or label.sum() > 0 else 0.0
        # Use prediction scores as weights for COM calculation (following eval.txt approach)
        is_pinpointing = check_pinpointing(cluster_mask, label, self.coords, weights=pred)
        distance = compute_distance_to_lesion(cluster_mask, label, self.coords)

        return ClusterMetrics(
            subject_id=subject_id,
            hemi=hemi,
            split=self.split,  # Added
            cluster_id=cluster_id,
            mean_score=mean_score,
            max_score=max_score,
            area_cm2=area_cm2,
            n_vertices=n_vertices,
            tp_verts=tp_verts,
            fp_verts=fp_verts,
            fn_verts=fn_verts,
            ppv_rand_expected=ppv_rand_expected,
            box_dice=box_dice,
            has_overlap=has_overlap,
            is_pinpointing=is_pinpointing,
            distance_to_lesion_mm=distance,
            vertex_indices=np.where(cluster_mask)[0],
        )

    def _compute_union_dice(self, f_pred: h5py.File, subject_id: str) -> float:
        """Compute Dice for union of all predicted clusters vs label."""
        union_tp, union_fp, union_fn = 0, 0, 0

        for hemi in ["lh", "rh"]:
            try:
                _, clustered = self._load_predictions(f_pred, subject_id, hemi)
                label = self._load_label(subject_id, hemi)
            except (KeyError, TypeError):
                continue

            pred_mask = clustered > 0
            label_mask = label > 0.5
            union_tp += int(np.sum(pred_mask & label_mask))
            union_fp += int(np.sum(pred_mask & ~label_mask))
            union_fn += int(np.sum(~pred_mask & label_mask))

        denom = 2 * union_tp + union_fp + union_fn
        return (2.0 * union_tp / denom) if denom > 0 else 0.0

    def compute_pr_curve(self, thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute Precision-Recall curve and AUPRC.

        Args:
            thresholds: Array of thresholds to evaluate (default: 50 values from 0.01 to 0.99)

        Returns:
            Dictionary with precision, recall, thresholds, and AUPRC
        """
        if not self._all_predictions or not self._all_labels:
            logger.warning("No prediction data available for PR curve. Run evaluate_all() first.")
            return {"precision": np.array([]), "recall": np.array([]), "thresholds": np.array([]), "auprc": 0.0}

        # Concatenate all predictions and labels
        all_pred = np.concatenate(self._all_predictions)
        all_label = np.concatenate(self._all_labels)

        # Use sklearn for PR curve
        precision, recall, pr_thresholds = precision_recall_curve(all_label > 0.5, all_pred)
        auprc = auc(recall, precision)

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
            "auprc": float(auprc),
        }

    def compute_roc_curve(self, thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute ROC curve and AUROC.

        Args:
            thresholds: Array of thresholds to evaluate (default: auto from sklearn)

        Returns:
            Dictionary with fpr, tpr, thresholds, and auroc
        """
        if not self._all_predictions or not self._all_labels:
            logger.warning("No prediction data available for ROC curve. Run evaluate_all() first.")
            return {"fpr": np.array([]), "tpr": np.array([]), "thresholds": np.array([]), "auroc": 0.0}

        # Concatenate all predictions and labels
        all_pred = np.concatenate(self._all_predictions)
        all_label = np.concatenate(self._all_labels)

        # Use sklearn for ROC curve
        fpr, tpr, roc_thresholds = roc_curve(all_label > 0.5, all_pred)
        auroc = auc(fpr, tpr)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
            "auroc": float(auroc),
        }

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from all results."""
        # Filter lesion cases for detection metrics
        lesion_subjects = [s for s in self.subject_results if s.is_lesion]
        hc_subjects = [s for s in self.subject_results if not s.is_lesion]

        # Vertex-level summary (lesion cases only)
        lesion_vertex = [v for v in self.vertex_results
                         if v.lesion_area_cm2 > 0]
        dsc_values = [v.dice for v in lesion_vertex]
        iou_values = [v.iou for v in lesion_vertex]

        # Cluster-level summary
        all_clusters = self.cluster_results
        tp_clusters_boxdsc = [c for c in all_clusters if c.is_tp_boxdsc]
        tp_clusters_overlap = [c for c in all_clusters if c.is_tp_overlap]
        tp_clusters_ppv50 = [c for c in all_clusters if c.is_tp_ppv50]
        fp_clusters = [c for c in all_clusters if not c.is_tp_overlap]

        # Subject-level summary
        n_lesion = len(lesion_subjects)
        n_hc = len(hc_subjects)
        detection_rate_boxdsc = np.mean([s.detected_boxdsc for s in lesion_subjects]) if lesion_subjects else 0.0
        detection_rate_overlap = np.mean([s.detected_overlap for s in lesion_subjects]) if lesion_subjects else 0.0
        detection_rate_ppv50 = np.mean([s.detected_ppv50 for s in lesion_subjects]) if lesion_subjects else 0.0
        pinpointing_rate = np.mean([s.pinpointed for s in lesion_subjects]) if lesion_subjects else 0.0
        specificity = np.mean([s.n_clusters == 0 for s in hc_subjects]) if hc_subjects else 1.0

        # recall@threshold
        recall_015 = np.mean([s.lesion_dice_union >= 0.15 for s in lesion_subjects]) if lesion_subjects else 0.0
        recall_020 = np.mean([s.lesion_dice_union >= 0.20 for s in lesion_subjects]) if lesion_subjects else 0.0

        # Top-1 hit rate (most confident cluster hits lesion)
        top1_hit_rate_boxdsc = np.mean([s.top1_hit_boxdsc for s in lesion_subjects]) if lesion_subjects else 0.0
        top1_hit_rate_overlap = np.mean([s.top1_hit_overlap for s in lesion_subjects]) if lesion_subjects else 0.0
        top1_hit_rate_ppv50 = np.mean([s.top1_hit_ppv50 for s in lesion_subjects]) if lesion_subjects else 0.0

        # Compute PR curve for AUPRC
        pr_data = self.compute_pr_curve()
        auprc = pr_data.get("auprc", 0.0)

        # Compute precision and F1 for boxDSC criterion
        n_tp_boxdsc = len(tp_clusters_boxdsc)
        n_fp_boxdsc = sum(1 for c in all_clusters if not c.is_tp_boxdsc)
        precision_boxdsc = n_tp_boxdsc / (n_tp_boxdsc + n_fp_boxdsc) if (n_tp_boxdsc + n_fp_boxdsc) > 0 else 0.0
        f1_boxdsc = 2 * precision_boxdsc * detection_rate_boxdsc / (precision_boxdsc + detection_rate_boxdsc) if (precision_boxdsc + detection_rate_boxdsc) > 0 else 0.0

        # Compute precision and F1 for distance ≤ 20mm criterion (MELD Graph paper)
        tp_clusters_distance = [c for c in all_clusters if c.is_tp_distance]
        n_tp_distance = len(tp_clusters_distance)
        n_fp_distance = sum(1 for c in all_clusters if not c.is_tp_distance)
        precision_distance = n_tp_distance / (n_tp_distance + n_fp_distance) if (n_tp_distance + n_fp_distance) > 0 else 0.0
        detection_rate_distance = np.mean([s.detected_distance for s in lesion_subjects]) if lesion_subjects else 0.0
        f1_distance = 2 * precision_distance * detection_rate_distance / (precision_distance + detection_rate_distance) if (precision_distance + detection_rate_distance) > 0 else 0.0

        # Compute precision and F1 for containment / PPV-in-mask >= 0.5 criterion
        n_tp_ppv50 = len(tp_clusters_ppv50)
        n_fp_ppv50 = sum(1 for c in all_clusters if not c.is_tp_ppv50)
        precision_ppv50 = n_tp_ppv50 / (n_tp_ppv50 + n_fp_ppv50) if (n_tp_ppv50 + n_fp_ppv50) > 0 else 0.0
        f1_ppv50 = 2 * precision_ppv50 * detection_rate_ppv50 / (precision_ppv50 + detection_rate_ppv50) if (precision_ppv50 + detection_rate_ppv50) > 0 else 0.0

        # FP per patient statistics
        fp_per_patient = [s.n_fp_clusters for s in self.subject_results]
        fp_per_patient_mean = float(np.mean(fp_per_patient)) if fp_per_patient else 0.0
        fp_per_patient_median = float(np.median(fp_per_patient)) if fp_per_patient else 0.0

        fp_ppv50_per_patient = [s.n_clusters - s.n_tp_clusters_ppv50 for s in self.subject_results]
        fp_ppv50_per_patient_mean = float(np.mean(fp_ppv50_per_patient)) if fp_ppv50_per_patient else 0.0
        fp_ppv50_per_patient_median = float(np.median(fp_ppv50_per_patient)) if fp_ppv50_per_patient else 0.0

        summary = {
            "model": self.model_name,
            "split": self.split,
            "timestamp": datetime.now().isoformat(),
            "n_subjects": len(self.subject_results),
            "n_lesion_cases": n_lesion,
            "n_hc_cases": n_hc,
            "vertex_level": {
                "DSC_mean": float(np.mean(dsc_values)) if dsc_values else 0.0,
                "DSC_std": float(np.std(dsc_values)) if dsc_values else 0.0,
                "DSC_median": float(np.median(dsc_values)) if dsc_values else 0.0,
                "IoU_mean": float(np.mean(iou_values)) if iou_values else 0.0,
                "IoU_std": float(np.std(iou_values)) if iou_values else 0.0,
                "AUPRC": auprc,
            },
            "cluster_level": {
                "n_total_clusters": len(all_clusters),
                "n_tp_boxdsc": n_tp_boxdsc,
                "n_tp_distance": n_tp_distance,
                "n_tp_overlap": len(tp_clusters_overlap),
                "n_tp_ppv50": n_tp_ppv50,
                "n_fp": len(fp_clusters),
                "sensitivity_boxdsc": detection_rate_boxdsc,
                "sensitivity_distance": detection_rate_distance,
                "sensitivity_overlap": detection_rate_overlap,
                "sensitivity_ppv50": detection_rate_ppv50,
                "precision_boxdsc": precision_boxdsc,
                "precision_distance": precision_distance,
                "precision_overlap": len(tp_clusters_overlap) / len(all_clusters) if all_clusters else 0.0,
                "precision_ppv50": precision_ppv50,
                "F1_boxdsc": f1_boxdsc,
                "F1_distance": f1_distance,
                "F1_ppv50": f1_ppv50,
                "FP_per_patient_mean": fp_per_patient_mean,
                "FP_per_patient_median": fp_per_patient_median,
                "FP_per_patient_ppv50_mean": fp_ppv50_per_patient_mean,
                "FP_per_patient_ppv50_median": fp_ppv50_per_patient_median,
            },
            "subject_level": {
                "detection_rate_boxdsc": detection_rate_boxdsc,
                "detection_rate_distance": detection_rate_distance,
                "detection_rate_overlap": detection_rate_overlap,
                "detection_rate_ppv50": detection_rate_ppv50,
                "pinpointing_rate": pinpointing_rate,
                "top1_hit_rate_boxdsc": top1_hit_rate_boxdsc,
                "top1_hit_rate_overlap": top1_hit_rate_overlap,
                "top1_hit_rate_ppv50": top1_hit_rate_ppv50,
                "specificity": specificity,
                "recall_at_0.15": recall_015,
                "recall_at_0.20": recall_020,
            },
        }
        return summary

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save all results to CSV and JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Vertex-level
        vertex_df = pd.DataFrame([v.to_dict() for v in self.vertex_results])
        vertex_df.to_csv(output_dir / "vertex_level_results.csv", index=False)

        # Cluster-level
        cluster_df = pd.DataFrame([c.to_dict() for c in self.cluster_results])
        cluster_df.to_csv(output_dir / "cluster_level_results.csv", index=False)

        # Subject-level
        subject_df = pd.DataFrame([s.to_dict() for s in self.subject_results])
        subject_df.to_csv(output_dir / "subject_level_results.csv", index=False)

        # Summary
        summary = self.generate_summary()
        with open(output_dir / "summary_statistics.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save curve data for plotting
        curves_data = {}

        # FROC curve
        froc = self.compute_froc_curve()
        curves_data["froc"] = {
            "thresholds": froc["thresholds"].tolist(),
            "sensitivity": froc["sensitivity"].tolist(),
            "fp_per_patient": froc["fp_per_patient"].tolist(),
        }

        # PR curve (subsample to max 200 points for file size)
        pr = self.compute_pr_curve()
        if len(pr["precision"]) > 0:
            max_points = 200
            n_points = len(pr["precision"])
            if n_points > max_points:
                step = n_points // max_points
                indices = list(range(0, n_points, step))
                # Always include last point
                if indices[-1] != n_points - 1:
                    indices.append(n_points - 1)
                precision = [float(pr["precision"][i]) for i in indices]
                recall = [float(pr["recall"][i]) for i in indices]
                thresholds = [float(pr["thresholds"][i]) for i in indices if i < len(pr["thresholds"])]
            else:
                precision = [float(x) for x in pr["precision"]]
                recall = [float(x) for x in pr["recall"]]
                thresholds = [float(x) for x in pr["thresholds"]] if len(pr["thresholds"]) > 0 else []

            curves_data["pr"] = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
                "auprc": pr["auprc"],
            }

        # ROC curve (subsample to max 200 points for file size)
        roc = self.compute_roc_curve()
        if len(roc["fpr"]) > 0:
            max_points = 200
            n_points = len(roc["fpr"])
            if n_points > max_points:
                step = n_points // max_points
                indices = list(range(0, n_points, step))
                if indices[-1] != n_points - 1:
                    indices.append(n_points - 1)
                fpr = [float(roc["fpr"][i]) for i in indices]
                tpr = [float(roc["tpr"][i]) for i in indices]
                thresholds = [float(roc["thresholds"][i]) for i in indices if i < len(roc["thresholds"])]
            else:
                fpr = [float(x) for x in roc["fpr"]]
                tpr = [float(x) for x in roc["tpr"]]
                thresholds = [float(x) for x in roc["thresholds"]] if len(roc["thresholds"]) > 0 else []

            curves_data["roc"] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auroc": roc["auroc"],
            }

        with open(output_dir / "curves_data.json", "w") as f:
            json.dump(curves_data, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    def compute_froc_curve(self, thresholds: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute FROC curve: Sensitivity vs FP/patient at different thresholds.

        Uses cluster mean_score as the varying threshold.
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 51)

        sensitivities = []
        fp_per_patient = []

        lesion_subjects = [s for s in self.subject_results if s.is_lesion]
        n_lesion = len(lesion_subjects)
        n_total = len(self.subject_results)

        for thresh in thresholds:
            # Filter clusters by threshold
            n_detected = 0
            n_fp = 0

            for subject in self.subject_results:
                # Get clusters for this subject
                subj_clusters = [c for c in self.cluster_results if c.subject_id == subject.subject_id]
                filtered_clusters = [c for c in subj_clusters if c.mean_score >= thresh]

                # Count TP (at least one cluster overlaps lesion) and FP
                has_tp = any(c.is_tp_overlap for c in filtered_clusters)
                n_fp_subj = sum(1 for c in filtered_clusters if not c.is_tp_overlap)

                if subject.is_lesion and has_tp:
                    n_detected += 1
                n_fp += n_fp_subj

            sensitivity = n_detected / n_lesion if n_lesion > 0 else 0.0
            fpp = n_fp / n_total if n_total > 0 else 0.0

            sensitivities.append(sensitivity)
            fp_per_patient.append(fpp)

        return {
            "thresholds": thresholds,
            "sensitivity": np.array(sensitivities),
            "fp_per_patient": np.array(fp_per_patient),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def run_three_level_evaluation(
    model_path: Union[str, Path],
    split: str = "val",
    model_name: str = "best_model",
    output_dir: Optional[Union[str, Path]] = None,
    hdf5_file_root: str = "{site_code}_{group}_featurematrix_combat.hdf5",
    lesion_file_root: Optional[str] = None,
    lesion_feature: str = "lesion_main",
    clustered_key: str = "prediction_clustered",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run three-level evaluation on a trained model.

    Args:
        model_path: Path to fold directory (e.g., meld_data/models/<exp>/fold_00)
        split: Data split ("val", "test", "train")
        model_name: Model checkpoint name (default: "best_model")
        output_dir: Output directory (default: results_<model_name>/three_level_eval/)
        hdf5_file_root: HDF5 file root pattern
        lesion_file_root: Optional path to derived lesion labels
        lesion_feature: Lesion label feature name (default: "lesion_main")
        clustered_key: Dataset key for clustered predictions within each hemi group
        verbose: Print progress

    Returns:
        Summary statistics dictionary
    """
    model_path = Path(model_path)
    results_dir = model_path / f"results_{model_name}"

    # Determine predictions file
    suffix = {"val": "_val", "test": "", "train": "_train", "trainval": "_trainval"}
    pred_path = results_dir / f"predictions{suffix.get(split, '')}.hdf5"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    # Load cohort
    cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=None)

    # Determine output directory
    if output_dir is None:
        output_dir = results_dir / "three_level_eval" / split
    output_dir = Path(output_dir)

    # Load subject IDs from data_parameters.json
    data_params_path = model_path / "data_parameters.json"
    subject_ids = None
    if data_params_path.exists():
        with open(data_params_path) as f:
            data_params = json.load(f)
        subject_ids = data_params.get(f"{split}_ids", None)

    # Create evaluator
    lesion_root = Path(lesion_file_root) if lesion_file_root else None
    evaluator = ThreeLevelEvaluator(
        predictions_hdf5=pred_path,
        cohort=cohort,
        subject_ids=subject_ids,
        lesion_file_root=lesion_root,
        lesion_feature=lesion_feature,
        clustered_key=clustered_key,
        split=split,
        model_name=model_name,
    )

    # Run evaluation
    summary = evaluator.evaluate_all(verbose=verbose)

    # Save results
    evaluator.save_results(output_dir)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run three-level FCD evaluation")
    parser.add_argument("--model_path", required=True, help="Path to fold directory")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--model_name", default="best_model")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lesion_file_root", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    summary = run_three_level_evaluation(
        model_path=args.model_path,
        split=args.split,
        model_name=args.model_name,
        output_dir=args.output_dir,
        lesion_file_root=args.lesion_file_root,
        verbose=args.verbose,
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
