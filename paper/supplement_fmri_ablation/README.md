## Supplementary — fMRI Model Ablation (V2)

This folder contains reproducible summaries of the **V2 fMRI model ablation matrix** under
multiple *deploy-real* hemisphere-selection settings (no oracle hemisphere).

### Environment
Always run after activating the project conda environment:
```bash
source env.sh
```

### Inputs / Specs
- `docs/pretrain/v2_model_archive.md`
- `docs/pretrain/v2_full_ablation_matrix*.md`

### Scripts (generated tables + traceability)
- **Deploy-real using T1 hemisphere** (`hemi_mode=t1_pred`)
  - `python paper/supplement_fmri_ablation/make_fmri_ablation_table.py`
- **Deploy-real fMRI-only (no deploy hemisphere selection)**
  - `python paper/supplement_fmri_ablation/make_fmri_ablation_table_real_deploy_fmri_only.py`
- **Deploy-real fMRIhemi** (fMRI-only laterality classifier → hemisphere)
  - `python paper/supplement_fmri_ablation/make_fmri_ablation_table_real_deploy_fmrihemi.py`
  - `python paper/supplement_fmri_ablation/make_fmri_ablation_table_real_deploy_fmrihemi_t1fail_rescue.py`
    - Defines “rescue” as additional detections among the manuscript Table 2 T1-baseline failures
      (anchored to `paper/table2/table2_audit.json`), so it is not affected by TrackA gating changes.

### Outputs (generated)
- `paper/supplement_fmri_ablation/tableS_fmri_ablation*.tsv`
- `paper/supplement_fmri_ablation/figureS_fmri_ablation*.pdf`
- `paper/supplement_fmri_ablation/ablation_traceability*.json`
