# End-to-end pipeline (code path; requires your own data)

This repository contains **two** reproducibility pathways:

1) **Paper-results reproduction (recommended for reviewers)**  
   Uses de-identified intermediate artifacts and does **not** require raw imaging or model weights.
   See `README.md`.

2) **End-to-end code pathway (this document)**  
   Publishes the **rs-fMRI model code** and the **TrackA post-hoc fusion/gating** code so that
   researchers can run the pipeline on their **own** data.

This pathway requires:
- Your own raw/preprocessed imaging data (not provided here).
- External model weights (see `weights/`).
- Upstream MELD/meld_graph as a dependency (installed via `scripts/setup_third_party.sh`).

## 0) Environment

Create the conda env:
```bash
conda env create -f environment/MELD_fMRI_env_export.yml
conda activate MELD_fMRI
```

## 1) Install upstream `meld_graph` (pinned) + patch

```bash
bash scripts/setup_third_party.sh
```

This clones upstream `meld_graph` at the locked commit in `third_party/meld_graph.lock` and applies
`third_party/patches/meld_graph.patch` (path overrides via `MELD_DATA_PATH`).

## 2) Set data root

All end-to-end scripts assume:
```bash
export MELD_DATA_PATH="$(pwd)/meld_data"
```

If you want MELD parameters (fsaverage_sym surfaces, etc.) you can download them via upstream:
```bash
python -c "from meld_graph.download_data import get_meld_params; get_meld_params()"
```

## 3) Download weights (external)

See `weights/README.md` and run:
```bash
bash weights/download_weights.sh
sha256sum -c weights/checksums.sha256
```

## 4) rs-fMRI (DeepEZ-style GCN) — training / inference

Core implementation lives under `meld_fmri/fmri_gcn/`.

Reference training scripts (require a precomputed cache root with `cache/` and `adjacency.npy`):
- `scripts/end_to_end/train_deepez_gcn.py`
- `scripts/end_to_end/train_deepez_laterality.py`

These scripts are fold-aware and accept `--split_json` to enforce leakage-free splits.

Reference inference scripts (to generate `.npz` / `.csv` inputs for TrackA):
- `scripts/end_to_end/predict_deepez_gcn.py`
- `scripts/end_to_end/predict_deepez_laterality.py`

## 5) TrackA gate (T1-first with rs-fMRI takeover/injection)

Core paper implementation:
- `scripts/end_to_end/run_trackA_v2_fmrihemi_takeover_three_level_eval.py`

This script writes:
- `predictions_val.hdf5` (fused)
- `gate_decisions.csv` (traceability)
- three-level evaluation outputs via `meld_fmri.three_level_evaluation.ThreeLevelEvaluator`

### Required inputs (fold-aware)

TrackA expects:
- **T1 out-of-fold predictions** (`predictions_val.hdf5`) for each fold.
- **Fold split JSONs** (`data_parameters.json` with `train_ids`/`val_ids`) for each fold.
- **fMRI laterality CSVs** (fMRI-only; no oracle columns) for each fold. Required columns:
  `subject_id`, `prob_right`, `pred_hemi` (L/R).

In the authors' internal layout these are auto-discovered by defaults, but for public reuse you can
provide them explicitly via:
- `--t1_pred_hdf5_template` or `--t1_pred_hdf5_by_fold_json`
- `--split_json_template` or `--split_json_by_fold_json`
- `--laterality_csv_template` or `--laterality_csv_by_fold_json`

The exact “Route A” rules (threshold gate + tiered injection) are defined in:
- `paper/supplement_methods/SUPPLEMENTARY_METHODS_trackA_gate.md`

## Notes

- End-to-end execution depends on neuroimaging preprocessing choices (FreeSurfer/XCP-D, etc.).
  This repo provides the **model + fusion** code; it does not bundle raw imaging data.
- The paper tables/figures can be reproduced without running end-to-end training/inference.
