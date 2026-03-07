# Prognosis association (adjusted; main-text; Holm-corrected p)

- Cohort: Intermediate+Difficult; n=58 (ILAE12=38, ILAE3456=20)
- Endpoints: 3 binary endpoints only (A/B/C); Intersection & continuous PPV reported in Supplement.
- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).

## A. Det(boxDSC>0.22 & Dice>0.01)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 21/38 (55.3%) | 8/20 (40.0%) | 1.77 | [0.58, 5.6] | 0.1668 |
| rs-fMRI | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 12.5 | [2.49, 133] | 0.0042 |
| TrackA (fusion) | 0/58 (0.0%) | 30/38 (78.9%) | 2/20 (10.0%) | 29.6 | [6.5, 246] | 4.7e-06 |

## B. Det(PPV-in-mask≥0.5)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 16/38 (42.1%) | 1/20 (5.0%) | 9.07 | [1.9, 89.6] | 0.0138 |
| rs-fMRI | 0/58 (0.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 8.4 | [0.893, 1.12e+03] | 0.1668 |
| TrackA (fusion) | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 8.71 | [1.82, 85.4] | 0.0150 |

## C. Pinpointing (any-cluster COM resected)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 22/38 (57.9%) | 4/20 (20.0%) | 4.78 | [1.46, 18.5] | 0.0196 |
| rs-fMRI | 0/58 (0.0%) | 10/38 (26.3%) | 1/20 (5.0%) | 4.34 | [0.859, 43.6] | 0.1668 |
| TrackA (fusion) | 0/58 (0.0%) | 23/38 (60.5%) | 2/20 (10.0%) | 10.3 | [2.68, 58.8] | 0.0016 |

Notes:
- p values in this main-text table are Holm–Bonferroni corrected across 9 tests (3 endpoints × 3 methods).
- Inference: penalized likelihood ratio tests; 95% CIs use profile penalized likelihood (Firth logistic).
