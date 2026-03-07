# MELD_fMRI (Epilepsia) — reproducible archive + released code

This folder (`release/meld-fmri-epilepsia-repro/`) is a **standalone public-release archive** for our manuscript
results under `paper/revision/` *and* the corresponding released code (rs-fMRI model + TrackA fusion).

It supports two complementary pathways:

1) **Reviewer/reader pathway (recommended)** — reproduce the **paper tables/figures** from the **de-identified
   intermediate artifacts shipped here**. No raw MRI/fMRI data, no model weights, and no upstream MELD source code
   are required.
2) **Researcher pathway (end-to-end; requires your own data)** — run our **rs-fMRI model** and **TrackA
   post-hoc fusion/gating** on user-owned datasets, with upstream `meld_graph` treated as an external dependency.

> If you only need the paper numbers, run `bash reproduce_all.sh` and skip the end-to-end sections.

---

## Contents / non-contents (policy)

### Included

- `paper/` — final tables/figures used for the revision submission, plus supplementary material (including the rs-fMRI
  ablation matrix and TrackA gate definition).
- `scripts/` — scripts to regenerate those tables/figures **from the shipped intermediate artifacts**.
- `meld_data/` — **minimal, de-identified intermediate CSV/JSON artifacts** needed for reproduction
  (no raw imaging; no large prediction HDF5 files).
- `meld_fmri/` — released code:
  - `meld_fmri/fmri_gcn/` — rs-fMRI DeepEZ-style GCN/GAT implementation (+ laterality head)
  - `meld_fmri/three_level_evaluation.py` — three-level evaluation utilities used by TrackA
- `scripts/end_to_end/` — reference end-to-end entrypoints for training/inference/TrackA (requires your own data).
- `environment/` — conda environment export + linux-64 explicit lockfile for the **same environment name used in this
  project**: `MELD_fMRI`.
- `third_party/` + `scripts/setup_third_party.sh` — pinned upstream `meld_graph` commit + an idempotent patch workflow.
- `weights/` — helper scripts and checksums for externally hosted weights (not committed here).

### Not included (by design)

- Any raw imaging data (MRI/fMRI), NIfTI/DICOM, FreeSurfer subject directories, or per-subject PDFs.
- Any per-subject clinical dates or absolute filesystem paths.
- The upstream MELD/meld_graph source tree (it is installed as a dependency via `scripts/setup_third_party.sh`).
- The full manuscript `.docx` (journal policies vary). We provide scripts to check that your local manuscript files
  match this repo’s reproducible outputs.

---

## De-identification

All `subject_id` values in released artifacts are **deterministically hashed** (HMAC-SHA256) so that:
- IDs are consistent across all released CSV/JSON/TSV outputs.
- The secret hash key is not shipped in this public archive.

This means:
- You can reproduce the paper numbers exactly from the included artifacts.
- You cannot reverse the hashes to obtain original subject identifiers.

---

## Pathway 1 — reproduce paper tables/figures (no raw imaging required)

### 0) System requirements

- OS: Linux or macOS recommended (tested on Linux). Windows users should use WSL2.
- Conda: `conda` or `mamba`
- Hardware: CPU-only is sufficient (this pathway does **not** train models).
- Runtime: typically minutes (Table 3 uses bootstrap resampling; expect it to take longer than Table 1/2).

### 1) Create the conda environment (authors’ project environment)

From the **repo root of this archive** (`release/meld-fmri-epilepsia-repro/`):

Option A (portable export):
```bash
conda env create -f environment/MELD_fMRI_env_export.yml
conda activate MELD_fMRI
```

Option B (exact linux-64 lockfile; recommended for byte-for-byte reproducibility on Linux):
```bash
conda create -n MELD_fMRI --file environment/MELD_fMRI_env_explicit_linux-64.txt
conda activate MELD_fMRI
```

Optional sanity check:
```bash
python -c "import numpy, pandas; print('ok')"
pytest -q
```

### 2) One-command reproduction

```bash
bash reproduce_all.sh
```

What this script does (in order):
- Scans the release folder for private patterns (fails fast if any are found).
- Regenerates Table 1, Table 2, Table 3 markdown/TSV outputs in-place.
- Regenerates Figure 2 and Figure 3 panels (not required to be pixel-identical to the manuscript composites).
- Regenerates the rs-fMRI ablation matrix (Supplementary Table) TSV.
- Re-runs the private-pattern scan.

### 3) Where to find the reproduced tables (paper-facing outputs)

After `bash reproduce_all.sh`, the key paper-facing outputs are:

- **Table 1** (baseline cohort description):
  - `paper/revision/table1/table1_outcome58_baseline.md`
  - `paper/revision/table1/table1_outcome58_baseline.tsv`

- **Table 2** (main text prognosis table; final protocol):
  - `paper/revision/table2/final/prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md`
  - (per-endpoint TSVs, used for supplement and auditing) `paper/revision/table2/final/*.tsv`

- **Table 3** (scope38; multiplicity-adjusted; Epilepsia formatting):
  - (final formatted table; version-controlled) `paper/revision/table3/final/table3_scope38_multiplicity_adjusted_epilepsia.md`
  - (source table regenerated by `reproduce_all.sh`: full-precision p-values + bootstrap notes) `paper/revision/table3/final/table3_source_scope38_final.md`
  - (scope definition audit) `paper/revision/table3/final/table3_scope38_audit.json`

- **rs-fMRI ablation matrix** (Supplementary Table; scope38):
  - `paper/supplement_fmri_ablation/tableS_fmri_ablation_matrix_scope38.tsv`
  - (supplement text paragraph synced to the table) `paper/supplement_fmri_ablation/supple.txt`

Note: Tables are reproduced from the released intermediate evaluation artifacts under:
- `meld_data/output/three_level_eval/**`

### 4) Optional: check that your local manuscript files match the reproducible outputs

This public archive does **not** ship the full manuscript `.docx`. If you have the manuscript locally, you can verify
consistency in two ways:

#### 4.1 Lightweight anchor check (fast)

```bash
python scripts/release/check_manuscript_consistency.py --docx /path/to/manuscript.docx
```

This checks that key numeric anchors (e.g., `21/38`, `30/38`, ablation rescues) appear in your manuscript text.

#### 4.2 Full appeal-final check against Supplementary_Tables.xlsx (strict)

If you have **(a)** the appeal-final manuscript `.docx` **and** **(b)** `Supplementary_Tables.xlsx`, run:

```bash
python scripts/release/check_appeal_final_consistency.py \
  --docx /path/to/appeal_final_manuscript.docx \
  --supp_xlsx /path/to/Supplementary_Tables.xlsx
```

This script checks that:
- All **three** manuscript tables match the corresponding reproducible outputs in this archive.
- Supplementary tables S2/S3/S4 in `Supplementary_Tables.xlsx` match:
  - `paper/revision/supplement/table3_ppv_continuous_selftrained.tsv`
  - `paper/revision/supplement/native_meld/table2_native_meld_inject_union_tuned_thr0.10_diff0.15.tsv`
  - `paper/supplement_fmri_ablation/tableS_fmri_ablation_matrix_scope38.tsv`

---

## Pathway 2 — end-to-end rs-fMRI model + TrackA fusion (requires your own data)

This pathway publishes the **code** needed to run:
- the rs-fMRI lesion model (DeepEZ-style GCN/GAT),
- the rs-fMRI laterality model used by TrackA,
- TrackA gating + fusion + three-level evaluation.

It requires:
- Your own dataset (raw or preprocessed) and any local ground-truth labels you want for training/evaluation.
- Upstream `meld_graph` installed as a dependency.
- Model weights (either trained by you, or downloaded externally; see `weights/`).

### 2.0 Install upstream `meld_graph` (pinned) + patch

From the archive root:

```bash
conda activate MELD_fMRI
bash scripts/setup_third_party.sh
```

This:
- clones upstream `meld_graph` at the locked commit in `third_party/meld_graph.lock`,
- applies `third_party/patches/meld_graph.patch` idempotently,
- installs `meld_graph` in editable mode.

### 2.1 Set your `MELD_DATA_PATH`

Many `meld_graph` and evaluation utilities locate data via `MELD_DATA_PATH`.

For a standalone run *inside this archive* (paper reproduction only), you usually don’t need to set it.
For end-to-end runs on your own dataset, set:

```bash
export MELD_DATA_PATH="/path/to/your/meld_data"
```

To download MELD parameters (fsaverage_sym surfaces, etc.) via upstream:
```bash
python -c "from meld_graph.download_data import get_meld_params; get_meld_params()"
```

### 2.2 rs-fMRI model I/O contract (what `cache_root` must contain)

All rs-fMRI training/inference scripts in `scripts/end_to_end/` operate on a **precomputed graph cache root**:

```
<cache_root>/
  adjacency.npy         # float32, shape (N,N), normalized adjacency (dense)
  meta.json             # recommended; used to validate shapes and document node ordering
  cache/
    <subject_id>.npz    # one file per subject
    ...
```

Per-subject `*.npz` files may contain:

- `X` (**required**) — float32 array, shape `(N, F)`:
  - node feature matrix used by the lesion model
  - in the simplest setup, `F = N` (each node uses its FC row as features)
- `y` (required for training; optional for inference) — uint8/int64 array, shape `(N,)`:
  - node-level lesion labels (for training only)
- `L_loc` (optional) — float32 array, shape `(N, L)`:
  - per-node local features used by some fusion variants (FiLM / dual-expert) and by the paper laterality model
- `hemi` (required for laterality training) — string (`"L"` or `"R"`)

Important: **TrackA (paper implementation) assumes the Brainnetome cortical layout** (105 parcels per hemisphere) and
the “S26/V2” ordering described in `paper/supplement_methods/SUPPLEMENTARY_METHODS_trackA_gate.md`. If you use a
different atlas or node schema, you will need to adapt `scripts/end_to_end/run_trackA_v2_fmrihemi_takeover_three_level_eval.py`.

### 2.3 rs-fMRI lesion model — training (optional) and inference

#### Train (single fold)

You need a split file in MELD style (`data_parameters.json`) with:
- `train_ids`: list of subject IDs
- `val_ids`: list of subject IDs

Example:
```bash
python scripts/end_to_end/train_deepez_gcn.py \
  --cache_root /path/to/cache_root_ipsi_contra \
  --fold 0 \
  --split_json /path/to/fold_00/data_parameters.json \
  --arch gat \
  --fusion none \
  --device cuda:0
```

This writes a model directory containing (at minimum):
- `config.json`
- `best_model.pt`
- logs and per-epoch metrics CSVs

#### Inference (write per-subject `.npz` predictions)

```bash
python scripts/end_to_end/predict_deepez_gcn.py \
  --cache_root /path/to/cache_root_ipsi_contra \
  --model_dir /path/to/trained_model_dir \
  --split_json /path/to/fold_00/data_parameters.json \
  --split val \
  --out_dir /path/to/v2_dir/fold_00/val_predictions \
  --device cuda:0
```

Expected output files:
- `/path/to/v2_dir/fold_00/val_predictions/<subject_id>.npz` containing `p` (node probabilities).

To run inference on **all** cached subjects (no splits):
```bash
python scripts/end_to_end/predict_deepez_gcn.py \
  --cache_root /path/to/cache_root_ipsi_contra \
  --model_dir /path/to/trained_model_dir \
  --all_cache \
  --out_dir /path/to/v2_dir/fold_00/val_predictions
```

### 2.4 rs-fMRI laterality model (for TrackA) — training and inference

TrackA requires an rs-fMRI-only laterality CSV per fold with columns:
`subject_id, prob_right, pred_hemi`.

#### Train laterality model (single fold)

```bash
python scripts/end_to_end/train_deepez_laterality.py \
  --cache_root /path/to/cache_root_lh_rh \
  --fold 0 \
  --split_json /path/to/fold_00/data_parameters.json \
  --arch gat \
  --trunk_mode dual \
  --device cuda:0
```

This writes `best_model.pt` under the output directory.

#### Predict laterality CSV (single fold)

```bash
python scripts/end_to_end/predict_deepez_laterality.py \
  --cache_root /path/to/cache_root_lh_rh \
  --checkpoint /path/to/laterality_model_dir/best_model.pt \
  --arch gat \
  --trunk_mode dual \
  --split_json /path/to/fold_00/data_parameters.json \
  --split val \
  --out_csv /path/to/laterality/fold_00/val_predictions.csv \
  --device cuda:0
```

### 2.5 Model weights (optional)

Paper weights are distributed as **external downloads** (not committed in this repo). See:
- `weights/README.md`
- `weights/download_weights.sh`

Typical workflow:
```bash
bash weights/download_weights.sh
sha256sum -c weights/checksums.sha256
```

If you do not download weights, you can still run end-to-end by training models yourself (Sections 2.3–2.4).

---

## TrackA end-to-end (requires MELD_graph outputs)

### 3.0 What you need prepared

To run TrackA **as implemented in the paper**, you need (per fold):

1) **T1 out-of-fold predictions** from MELD_graph:
   - a `predictions_val.hdf5` file per fold (validation subjects only)
2) **Fold split JSONs**:
   - a `data_parameters.json` per fold containing `train_ids`/`val_ids`
3) **rs-fMRI laterality CSVs**:
   - `subject_id, prob_right, pred_hemi` per fold (Section 2.4)
4) **rs-fMRI lesion probabilities**:
   - per-subject `.npz` with `p` under `v2_dir/fold_XX/val_predictions/` (Section 2.3)
5) **Lesion labels (for evaluation)**:
   - `--lesion_root` pointing to MELD-style derived lesion labels (fsaverage_sym/xhemi-on-lh space)

### 3.1 Recommended directory layout (example)

You can use any layout, but the TrackA script is easiest to run if you structure your paths like:

```
/path/to/trackA_inputs/
  t1/
    fold_00/predictions_val.hdf5
    fold_00/data_parameters.json
    ...
  fmri_v2/
    fold_00/val_predictions/<subject_id>.npz
    fold_01/val_predictions/<subject_id>.npz
    ...
  laterality/
    fold_00/val_predictions.csv
    fold_01/val_predictions.csv
    ...
```

### 3.2 Run TrackA (paper Route A: threshold gate + tiered injection)

The implementation-level gate definition is in:
- `paper/supplement_methods/SUPPLEMENTARY_METHODS_trackA_gate.md`

Reference command (edit paths to your filesystem):
```bash
python scripts/end_to_end/run_trackA_v2_fmrihemi_takeover_three_level_eval.py \
  --name trackA_v2_fmrihemi_inject2_tieredlow \
  --v2_dir /path/to/trackA_inputs/fmri_v2 \
  --lesion_root /path/to/your/meld_data/derived_labels/lesion_main_island/template_fsaverage_sym_xhemi \
  --t1_pred_hdf5_template '/path/to/trackA_inputs/t1/fold_{fold:02d}/predictions_val.hdf5' \
  --split_json_template '/path/to/trackA_inputs/t1/fold_{fold:02d}/data_parameters.json' \
  --laterality_csv_template '/path/to/trackA_inputs/laterality/fold_{fold:02d}/val_predictions.csv' \
  --t1_conf_threshold 0.5773 \
  --area_target_cm2 60 \
  --area_max_cluster_cm2 30 \
  --area_max_total_cm2 80 \
  --k_clusters 3 \
  --min_vertices 100 \
  --enable_injection \
  --inject_diff_threshold 0.15 \
  --inject_diff_low_threshold 0.11 \
  --inject_lowdiff_t1conf_max 0.85 \
  --inject_lowdiff_require_same_hemi \
  --inject_conflict_margin_threshold 0.25 \
  --inject_k_fmri 2 \
  --fold all
```

### 3.3 TrackA outputs

By default, the script writes results to:
- `meld_data/output/three_level_eval/<name>_thr..._area..._a..._t.../`

Per fold:
- `fold0/val/predictions_val.hdf5` (fused output)
- `fold0/val/gate_decisions.csv` (traceability: which source was used and why)
- `fold0/val/deploy_config.json` (all parameters)
- three-level evaluation outputs under the same directory

Aggregated across folds (written when `--fold all`):
- `all_folds_subject_level_results.csv`
- `all_folds_cluster_level_results.csv`
- `all_folds_vertex_level_results.csv`
- `all_folds_gate_decisions.csv`
- `aggregate_val.json`

---

## Troubleshooting

- **`ModuleNotFoundError: meld_graph ...`**: run `bash scripts/setup_third_party.sh` inside the conda env.
- **Missing MELD params (fsaverage_sym)**: run `python -c "from meld_graph.download_data import get_meld_params; get_meld_params()"`.
- **CUDA not available**: scripts will fall back to CPU when possible, but training will be slow; use `--device cpu` explicitly.
- **Path templates**: `--*_template` strings must include `{fold}` or `{fold:02d}` and should be quoted if they contain braces.

---

## License and citation

- License: MIT (`LICENSE`)
- Citation metadata: `CITATION.cff`

---

## 中文说明（详细）

本文件夹（`release/meld-fmri-epilepsia-repro/`）是一个可独立发布的**公开复现包**，包含：

- 论文最终表格/补充材料（`paper/`）
- 复现脚本（`reproduce_all.sh` + `scripts/`）
- 我们自己实现的 rs-fMRI 模型代码与 TrackA 后融合/门控代码（`meld_fmri/` + `scripts/end_to_end/`）

### 复现论文表格（不需要原始影像）

```bash
conda env create -f environment/MELD_fMRI_env_export.yml
conda activate MELD_fMRI
bash reproduce_all.sh
```

### 端到端跑 rs-fMRI 模型与 TrackA（需要你自己的数据）

核心思想是：先准备好 rs-fMRI 的 `cache_root`（图结构 + 每个受试者的 `.npz` 特征），再用
`train_deepez_gcn.py / predict_deepez_gcn.py` 得到每个受试者的 node 概率输出；同时训练/推理 laterality
分类器得到 `prob_right/pred_hemi`；最后在已经有 MELD_graph 的 T1 输出（`predictions_val.hdf5`）的前提下，
运行 `run_trackA_v2_fmrihemi_takeover_three_level_eval.py` 完成 TrackA 融合并输出三层评估 CSV。

建议你按上面的英文步骤逐条执行；关键输入/输出的目录结构与命令都已写明。
