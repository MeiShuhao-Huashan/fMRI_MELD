# MANIFEST (public-release archive)

## Included

- `paper/revision/`
  - Final tables/figures for the revision submission (excluding the manuscript `.docx`).
- `paper/supplement_fmri_ablation/`
  - Scope38 ablation matrix (Supplementary Table) + traceability.
- `meld_fmri/`
  - Project code release: rs-fMRI model implementation + three-level evaluation utilities.
- `scripts/`
  - Scripts to regenerate paper tables/figures from the included intermediate artifacts.
- `scripts/end_to_end/`
  - End-to-end training/inference/evaluation entrypoints (requires user-owned data + external weights).
- `paper/table2/`
  - Lightweight metric utilities reused by table/ablation scripts.
- `supplement_table_hs_patient_three_level_metrics.filled.csv`
  - De-identified meta table (hashed subject IDs; no dates/paths).
- `meld_data/metadata/rc_volume_cm3_outcome58.csv`
  - De-identified RC volume table (no absolute label paths).
- `meld_data/output/three_level_eval/**`
  - Minimal three-level evaluation CSVs required for reproduction (no HDF5, no imaging).
- `third_party/`
  - Lock + patch files for upstream `meld_graph` (the upstream source tree is not shipped here).
- `environment/`
  - Conda env export + explicit lockfile for `MELD_fMRI`.
- `weights/`
  - Download helpers and checksums for externally hosted model weights.

## Excluded (by design)

- Any raw imaging data (MRI/fMRI), any NIfTI/DICOM, any FreeSurfer subject directories.
- Upstream MELD code (`meld_graph`) and other third-party tool source trees.
- Any absolute HPC paths and any per-subject report PDFs.
- The manuscript `.docx` (use the local consistency checker script instead).
