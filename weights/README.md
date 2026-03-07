# Model weights (external download; GitHub Releases recommended)

This public archive reproduces the paper tables/figures using **precomputed three-level evaluation CSVs**,
so model weights are **not required** for paper-result reproduction.

If you want to run inference end-to-end on your own data, download the paper weights externally and verify
their integrity with sha256.

## Files (expected)

- `fmri_models.tar.gz` (5-fold rs-fMRI lesion model weights)
- `laterality_models.tar.gz` (5-fold rs-fMRI laterality classifier weights)
- Optional: `t1_models.tar.gz` (5-fold T1-only model weights; only needed if you want to run T1 inference yourself)

## Download

Recommended: host weights as **GitHub Release assets** for this repo, then set:

- `GITHUB_REPO` (default: `MeiShuhao-Huashan/MELD_fMRI`)
- `GITHUB_RELEASE_TAG` (required; e.g., `epilepsia-weights-v1`)

Then run:
```bash
bash weights/download_weights.sh
```

Alternative: Zenodo download is still supported (see `weights/download_weights.sh`).

## Verify

```bash
sha256sum -c weights/checksums.sha256
```
