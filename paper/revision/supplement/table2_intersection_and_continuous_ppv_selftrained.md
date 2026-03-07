# Supplementary: Intersection & continuous PPV-in-mask (Intermediate + Difficult)

- Cohort: Intermediate+Difficult; n=58 (ILAE12=38, ILAE3456=20)
- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).
- Inference: p-values are penalized likelihood ratio tests; 95% CIs are profile penalized likelihood (Firth).
- Note: Main-text Table 2 reports Holm-corrected p-values for 3 binary endpoints × 3 methods; this supplement adds Intersection and the continuous PPV analysis.

## A. Det(boxDSC>0.22 & Dice>0.01)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 21/38 (55.3%) | 8/20 (40.0%) | 1.77 | [0.58, 5.6] | 0.1499 |
| rs-fMRI | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 12.5 | [2.49, 133] | 0.0006 |
| Intersection | 29/58 (50.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 18.5 | [1.74, 2.61e+03] | 0.0087 |
| TrackA (fusion) | 0/58 (0.0%) | 30/38 (78.9%) | 2/20 (10.0%) | 29.6 | [6.5, 246] | 5.2e-07 |

## B. Det(PPV-in-mask≥0.5)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 16/38 (42.1%) | 1/20 (5.0%) | 9.07 | [1.9, 89.6] | 0.0023 |
| rs-fMRI | 0/58 (0.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 8.4 | [0.893, 1.12e+03] | 0.0587 |
| Intersection | 29/58 (50.0%) | 9/38 (23.7%) | 0/20 (0.0%) | 17.9 | [1.77, 2.49e+03] | 0.0075 |
| TrackA (fusion) | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 8.71 | [1.82, 85.4] | 0.003 |

## C. Pinpointing (any-cluster COM resected)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 22/38 (57.9%) | 4/20 (20.0%) | 4.78 | [1.46, 18.5] | 0.0049 |
| rs-fMRI | 0/58 (0.0%) | 10/38 (26.3%) | 1/20 (5.0%) | 4.34 | [0.859, 43.6] | 0.0556 |
| Intersection | 29/58 (50.0%) | 9/38 (23.7%) | 0/20 (0.0%) | 17.9 | [1.77, 2.49e+03] | 0.0075 |
| TrackA (fusion) | 0/58 (0.0%) | 23/38 (60.5%) | 2/20 (10.0%) | 10.3 | [2.68, 58.8] | 0.0002 |

## D. Continuous PPV-in-mask (max across clusters)

| Model | PPV median [IQR] in ILAE12 | PPV median [IQR] in ILAE3456 | Mann–Whitney p | Adjusted OR per 0.1 PPV | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (self-trained) | 0.425 [0.000, 0.627] | 0.091 [0.000, 0.178] | 0.0296 | 1.32 | [1.07, 1.71] | 0.0006 |
| rs-fMRI | 0.017 [0.000, 0.390] | 0.000 [0.000, 0.000] | 0.0018 | 1.6 | [1.1, 3.27] | 0.0007 |
| Intersection | 0.000 [0.000, 0.052] | 0.000 [0.000, 0.000] | 0.0088 | 1.45 | [1.06, 3.76] | 0.0009 |
| TrackA (fusion) | 0.425 [0.145, 0.682] | 0.000 [0.000, 0.000] | 1.3e-05 | 1.53 | [1.19, 2.17] | 2.3e-05 |

