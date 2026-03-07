# Table 2 traceability (multi-scale performance; primary endpoints = Det(boxDSC>0.22) and Det(PPV-in-mask≥0.5))

This folder contains a publication-ready Table 2 comparing:
- **T1-only baseline** (in-house cohort; deploy-style post-processing; *no area/cluster budget constraints*)
- **fMRI-only (fMRIhemi B2a)** (single-modality fMRI deployment with **30/80** area/cluster budgets)
- **Track A (T1+fMRI; deployable post-hoc fusion)** using the **Route A** gate (**threshold gate + tiered-low injection**) and the same fMRIhemi B2a rescue source.

All numbers are derived from **out-of-fold** validation predictions in 5-fold CV (n=52; Engel I cohort; resection cavity as imaging proxy reference standard).

## 1) Source evaluation outputs (inputs)

The table is generated from these three-level evaluation folders (each contains either `all_folds_*.csv` and/or `fold0..4/val/*.csv`):

- T1-only baseline (unbudgeted; three-level evaluation):  
  `meld_data/output/three_level_eval/t1_baseline_val52/unbudgeted_ppv/`

- Track A (no-leak; threshold gate + tiered-low injection; 30/80 budgets; three-level evaluation):  
  `meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80/`

- fMRI-only (fMRIhemi B2a; **real deploy**, no T1 hemi guidance; 30/80 budgets):  
  `meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80/`

## 2) “No leakage” audit for Track A (why this version is deployable)

**(a) Out-of-fold fMRI rescue source (no cross-fold reuse)**

Track A uses the same out-of-fold fMRIhemi B2a pipeline that produced:
`meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80/`

For auditability, `aggregate_val.json` in the Track A folder records:
- `meta.fmrihemi_v2.v2_dir`: the fMRI V2 model root used for rescue
- `meta.fmrihemi_v2.laterality_csv_by_fold`: fold-specific laterality prediction CSVs
- `meta.no_leakage=true`: a run-time guard flag asserted by the evaluation script

**(b) Deployable gating (not using GT Dice to decide switching)**

In `aggregate_val.json`, Track A uses deploy-available signals only:
- `meta.gate.t1_conf_feature = "top_cluster_mean"`
- `meta.gate.t1_conf_threshold = 0.5773`
- injection is enabled (`meta.injection.enabled=true`) with parameters captured in the same JSON.

**(c) Deployable hemisphere selection (not using oracle hemisphere)**

Track A uses **fMRI-only hemisphere prediction** (from the laterality classifier used by fMRIhemi) and does not use GT laterality.

## 3) Table 2 generation (reproducible)

Generate the table with bootstrap CIs and paired tests:

```bash
python paper/table2/generate_table2.py \
  --t1_eval_dir meld_data/output/three_level_eval/t1_baseline_val52/unbudgeted_ppv \
  --tracka_eval_dir meld_data/output/three_level_eval/trackA_v2_fmrihemi_inject2_tieredlow_thr0.5773_area60_a30_t80 \
  --fmri_eval_dir meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80 \
  --out_dir paper/table2 \
  --n_boot 10000 \
  --n_perm 20000 \
  --seed 42
```

Outputs:
- `paper/table2/table2_multiscale_performance.tsv` (submission-friendly)
- `paper/table2/table2_multiscale_performance.md`
- `paper/table2/table2_source_data.csv` (per-subject paired source data)
- `paper/table2/table2_audit.json` (parameters, point estimates, p-values)

## 4) Metric mapping to your requested structure

- **Vertex-level**: DSC on the **lesion hemisphere** (the `vertex_level_results.csv` rows with `lesion_area_cm2>0`)
- **Cluster-level (boxDSC>0.22)**:
  - Number/subject: predicted clusters per subject (mean)
  - FP/pt: false-positive clusters per subject (mean)
  - Precision / Sensitivity / F1 score (pooled across clusters and subjects)
- **Cluster-level (PPV-in-mask≥0.5)**:
  - TP: clusters with `PPV-in-mask ≥ 0.5` (cluster-level containment criterion)
  - FP/pt: clusters not meeting the TP criterion, per subject (mean)
  - Precision / Sensitivity / F1 score (pooled across clusters and subjects)
- **Subject-level**:
  - Detection rate (primary): Det(boxDSC>0.22)
  - Detection rate (co-primary): Det(PPV-in-mask≥0.5)
  - Detection rate (secondary): Det(distance≤20mm)
  - Pinpointing rate: ≥1 cluster COM within lesion mask
  - Recall@0.15 (secondary): proportion of subjects with `lesion_dice_union >= 0.15`

Notes:
- Because all subjects are “positive cases” (Engel I with RC mask), **cluster-level sensitivity equals subject-level detection rate** under the boxDSC criterion.
- Cluster-level PPV-in-mask summary rows are moved to Supplementary Table `paper/table2/tableS_ppv50_cluster_level.tsv` (subject-level Det(PPV50) remains in Table 2).
