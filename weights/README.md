# Model weights (external download; GitHub Releases recommended)

This public archive reproduces the paper tables/figures using **precomputed three-level evaluation CSVs**,
so model weights are **not required** for paper-result reproduction.

If you want to run inference end-to-end on your own data, download the paper weights externally and verify
their integrity with sha256.

## Files (expected)

- `fmri_models.tar.gz` (5-fold rs-fMRI lesion model weights; `v2_gat3_diceproxy/`)
- `laterality_models.tar.gz` (5-fold rs-fMRI laterality classifier weights; `fmri_laterality_absLR_dualExpert_balacc/`)
- Optional: `t1_models.tar.gz` (5-fold T1-only MELD_graph baseline weights; only needed if you want to run T1 inference yourself)

## Download

Recommended: host weights as **GitHub Release assets** for this repo, then set:

- `GITHUB_REPO` (default: `MeiShuhao-Huashan/fMRI_MELD_epilepsia`)
- `GITHUB_RELEASE_TAG` (required; e.g., `epilepsia-weights-v1`)

Then run:
```bash
bash weights/download_weights.sh
```

Alternative: Zenodo download is still supported (see `weights/download_weights.sh`).

### Maintainers: upload assets (one-time)

If you are preparing the Release assets yourself, build the archives (see repo instructions) and then:
```bash
export GITHUB_TOKEN=...              # PAT with Releases write access
export GITHUB_RELEASE_TAG=epilepsia-weights-v1
bash weights/upload_release_assets.sh
```

## Verify

```bash
sha256sum -c weights/checksums.sha256
```

## Extract

The download script stores archives under `weights/downloaded/`. To extract into the conventional layout:

```bash
mkdir -p meld_data/models
tar -xzf weights/downloaded/fmri_models.tar.gz -C meld_data/models
tar -xzf weights/downloaded/laterality_models.tar.gz -C meld_data/models
```

This yields:
- `meld_data/models/v2_gat3_diceproxy/fold_00/.../fold_04/best_model.pt`
- `meld_data/models/fmri_laterality_absLR_dualExpert_balacc/fold_00/.../fold_04/best_model.pt`

If you also downloaded `t1_models.tar.gz`:
```bash
tar -xzf weights/downloaded/t1_models.tar.gz -C meld_data/models
```

This yields:
- `meld_data/models/meld_t1_monetunet/fold_00/.../fold_04/best_model.pt` (+ `network_parameters.json`)
