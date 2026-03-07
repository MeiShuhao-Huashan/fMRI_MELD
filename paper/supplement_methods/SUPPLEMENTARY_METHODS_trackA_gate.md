# Supplementary Methods — Deployable Track A Gate (Route A: threshold gate + tiered-low injection)

This section provides the **full, implementation-level definition** of the final Track A (T1+rs-fMRI) deployment gate used for Table 2 / Figures 2–4.

## S1. Overview

Track A is a **post-hoc, deployable fusion strategy** that uses **T1-only predictions by default** and only leverages rs-fMRI when deployment-available signals suggest that T1 is likely to fail. It has two modes of using rs-fMRI:

1) **Takeover (switch)**: replace the final candidate set with rs-fMRI candidates when T1 is low-confidence (or produces no clusters).  
2) **Injection (union)**: when T1 is retained, optionally **union** a limited number of rs-fMRI candidates with the T1 candidate set if rs-fMRI provides a strong complementary signal.

All gate decisions are computed **without oracle information** (no ground-truth laterality; no lesion mask usage).

## S2. Inputs and “no leakage” guards

Track A is evaluated in **five-fold cross-validation**, reusing the **exact same subject-level splits** as the T1-only baseline.

Deployment-time inputs per subject:

- **T1-only model outputs** (out-of-fold):
  - `prediction` (vertex probabilities; left and right hemispheres)
  - `prediction_clustered` (connected-component labels on the cortical mesh)
  - Source: per-fold `predictions_val.hdf5` recorded in `meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/fold*/val/deploy_config.json`.

- **rs-fMRI laterality classifier outputs** (out-of-fold; fMRI-only):
  - `prob_right` (P(lesion is right))
  - `pred_hemi` (L/R; argmax)
  - Source: fold-specific `val_predictions.csv` recorded in `meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/aggregate_val.json` under `meta.fmrihemi_v2.laterality_csv_by_fold`.

- **rs-fMRI V2 lesion probabilities** (out-of-fold):
  - V2 outputs ipsilateral/contralateral parcel probabilities; these are mapped to absolute left/right using `pred_hemi` (see S4).
  - Source: `meld_data/models/v2_gat3_diceproxy/` (fold-specific out-of-fold predictions).

## S3. T1 confidence signal (`t1_conf`)

Let `p_T1^h(v)` be the T1 vertex probability at cortical vertex `v` in hemisphere `h ∈ {L,R}` and `c_T1^h(v)` the corresponding integer cluster label (`0` = background; `>0` = cluster id).

For each hemisphere:

1) For each cluster id `k>0`, compute the cluster mean score:
   - `m_T1^h(k) = mean_{v: c_T1^h(v)=k} p_T1^h(v)`
2) Define the hemisphere confidence:
   - `t1_conf_h = max_k m_T1^h(k)` (0 if no clusters)

The deployment confidence and predicted hemisphere are:

- `t1_conf = max(t1_conf_L, t1_conf_R)`
- `t1_pred_hemi = L` if `t1_conf_L ≥ t1_conf_R`, else `R`
- `t1_has_cluster = any(c_T1^L>0) OR any(c_T1^R>0)`

This is the `top_cluster_mean` feature in code.

## S4. fMRIhemi(V2) signals (`fmri_top_score`, `fmri_minus_t1`, `lat_margin`)

### S4.1 fMRI laterality margin (`lat_margin`)

From the rs-fMRI laterality classifier:

- `prob_right ∈ [0,1]`
- `lat_margin = |prob_right − 0.5|`

`lat_margin` is used as a deployment-available measure of laterality confidence.

### S4.2 V2 probability mapping to absolute left/right

The V2 rs-fMRI model outputs parcel probabilities in an **ipsilateral/contralateral** ordering. Using `pred_hemi`:

- If `pred_hemi = L`: `p_V2^L = p_ipsi`, `p_V2^R = p_contra`
- If `pred_hemi = R`: `p_V2^R = p_ipsi`, `p_V2^L = p_contra`

Only the **105 cortical parcels** per hemisphere are used for candidate generation.

### S4.3 fMRIhemi candidate generation (deploy post-processing)

We convert parcel probabilities into vertex candidates using an **area-target + surface clustering** procedure:

1) **Area-target parcel selection** (`area_target_cm2 = 60`):  
   Concatenate left and right parcel scores (210 total), sort descending, then include parcels from highest to lowest until the summed parcel surface area reaches `area_target_cm2`.

2) **Surface clustering** on the cortical mesh using connected components.

3) **Minimum cluster size**: discard connected components with `<100` vertices.

4) **Deploy budgets (surface-area)**: keep up to `K=3` clusters, enforcing:
   - `AREA_MAX_CLUSTER_CM2 = 30`
   - `AREA_MAX_TOTAL_CM2 = 80`
   If a candidate cluster exceeds the remaining area budget, it is **trimmed by probability** (retain the highest-probability vertices until the area constraint is satisfied).

Clusters are ranked by **mean probability** (`cluster_sort = mean`) unless otherwise specified.

### S4.4 fMRI top score and relative score (`fmri_minus_t1`)

After deploy post-processing, define:

- `fmri_top_score` = mean probability of the **top-ranked** fMRIhemi cluster (0 if no clusters).
- `fmri_minus_t1 = fmri_top_score − t1_conf`
- `fmri_available = (at least one fMRIhemi cluster exists after filtering/budgets)`

## S5. Final Route A gate (switch + injection)

All fixed thresholds below were applied identically across folds.

### S5.1 Takeover (switch) rule

If either of the following holds:

- `t1_has_cluster = False`, **or**
- `t1_conf < 0.5773`

then Track A **switches to fMRIhemi**, provided `fmri_available = True`. If fMRIhemi is unavailable (e.g., missing fMRI inputs or no clusters after filtering), Track A falls back to T1.

### S5.2 Injection (union) rule

If Track A retains T1 under S5.1 and `fmri_available = True`, we optionally produce a **union** output (`source = "T1+fMRI"`) using a tiered rule:

- **High-diff injection**: `fmri_minus_t1 ≥ 0.15`
- **Conflict injection**: `(t1_pred_hemi ≠ fmri_pred_hemi) AND (lat_margin ≥ 0.25)`
- **Low-diff injection (guarded)**:  
  `fmri_minus_t1 ≥ 0.11` AND `t1_pred_hemi = fmri_pred_hemi` AND `t1_conf ≤ 0.85`

If any of the above holds, we create the final candidate set by union fusion (S5.3). Otherwise, the final output is the T1-only candidate set.

### S5.3 Union fusion (how “T1+fMRI” is constructed)

Let `K_total = 3` and `K_fMRI = 2`.

1) Rank fMRIhemi clusters by mean probability and keep the top `K_fMRI`.
2) Rank T1 clusters by mean probability and keep the top `K_total − K_fMRI_kept`.
3) Take the **union** of the retained T1 and fMRIhemi cluster masks on the cortical surface.
4) Re-label connected components on the union mask to produce the final cluster labels (this avoids overlap being “cut” into fragments).

## S6. Reproducible command and outputs

This exact configuration corresponds to:

- `meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/`

and is reproduced by:

```bash
conda activate MELD_fMRI
bash scripts/setup_third_party.sh
python scripts/end_to_end/run_trackA_v2_fmrihemi_takeover_three_level_eval.py \
  --enable_injection \
  --inject_diff_threshold 0.15 \
  --inject_diff_low_threshold 0.11 \
  --inject_lowdiff_t1conf_max 0.85 \
  --inject_lowdiff_require_same_hemi \
  --inject_conflict_margin_threshold 0.25 \
  --inject_k_fmri 2 \
  --name trackA_v2_fmrihemi_inject2_tieredlow \
  --t1_conf_threshold 0.5773 \
  --area_target_cm2 60 \
  --area_max_cluster_cm2 30 \
  --area_max_total_cm2 80 \
  --k_clusters 3 \
  --min_vertices 100
```

For public reuse on your own filesystem layout, override fold-aware inputs (T1 predictions, split JSONs, laterality CSVs) via:
`--t1_pred_hdf5_template/--t1_pred_hdf5_by_fold_json`, `--split_json_template/--split_json_by_fold_json`,
and `--laterality_csv_template/--laterality_csv_by_fold_json`.
