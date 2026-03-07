# Prognosis association (adjusted): native MELD + rs-fMRI TrackA (inject/union tuned)

- Cohort: Intermediate+Difficult; n=58 (ILAE12=38, ILAE3456=20)
- Constraints/policy: MELD(native) unconstrained; rs-fMRI uses laterality+area constraint (a30_t80); union=top2 fMRI + top1 T1 (relabel CC); TrackA gate: t1_conf_thr=0.10, diff_thr=0.15.
- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).

## A. Det(boxDSC>0.22)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (native) | 16/58 (27.6%) | 11/38 (28.9%) | 4/20 (20.0%) | 1.56 | [0.448, 6.17] | 0.2533 |
| rs-fMRI | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 12.5 | [2.49, 133] | 0.0006 |
| Intersection | 47/58 (81.0%) | 0/38 (0.0%) | 0/20 (0.0%) | NA | NA | NA |
| TrackA (fusion, inject/union tuned) | 0/58 (0.0%) | 22/38 (57.9%) | 3/20 (15.0%) | 7.06 | [1.99, 32.5] | 0.0010 |

## B. Det(PPV-in-mask≥0.5)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (native) | 16/58 (27.6%) | 20/38 (52.6%) | 3/20 (15.0%) | 6.13 | [1.72, 28.6] | 0.0024 |
| rs-fMRI | 0/58 (0.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 8.4 | [0.893, 1.12e+03] | 0.0587 |
| Intersection | 47/58 (81.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 17 | [1.61, 2.4e+03] | 0.0111 |
| TrackA (fusion, inject/union tuned) | 0/58 (0.0%) | 21/38 (55.3%) | 2/20 (10.0%) | 7.67 | [2.03, 42.2] | 0.0011 |

## C. Pinpointing (any-cluster COM resected)

| Model | No output n/N (%) | Resection=Yes in ILAE12 n/N (%) | Resection=Yes in ILAE3456 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (native) | 16/58 (27.6%) | 20/38 (52.6%) | 4/20 (20.0%) | 4.18 | [1.27, 16.4] | 0.0099 |
| rs-fMRI | 0/58 (0.0%) | 10/38 (26.3%) | 1/20 (5.0%) | 4.34 | [0.859, 43.6] | 0.0556 |
| Intersection | 47/58 (81.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 17 | [1.61, 2.4e+03] | 0.0111 |
| TrackA (fusion, inject/union tuned) | 0/58 (0.0%) | 22/38 (57.9%) | 3/20 (15.0%) | 6.61 | [1.9, 29.5] | 0.0013 |

## D. Continuous PPV-in-mask (max across clusters)

| Model | PPV median [IQR] in ILAE12 | PPV median [IQR] in ILAE3456 | Mann–Whitney p | Adjusted OR per 0.1 PPV | 95% CI | p |
|---|---|---|---|---|---|---|
| MELD (native) | 0.719 [0.000, 0.999] | 0.000 [0.000, 0.154] | 0.0278 | 1.19 | [1.04, 1.39] | 0.0004 |
| rs-fMRI | 0.017 [0.000, 0.390] | 0.000 [0.000, 0.000] | 0.0018 | 1.6 | [1.1, 3.27] | 0.0007 |
| Intersection | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.0297 | 1.33 | [1.05, 2.18] | 0.0009 |
| TrackA (fusion, inject/union tuned) | 0.611 [0.047, 0.993] | 0.000 [0.000, 0.045] | 9.6e-05 | 1.3 | [1.1, 1.6] | 4.9e-05 |

Notes:
- Multi-cluster handling: binary endpoints are `any-cluster positive`; continuous PPV uses `max_ppv_in_mask` across clusters.
- `No output` treated as not resected; PPV set to 0.
- Inference: p-values use penalized likelihood ratio tests; 95% CIs use profile penalized likelihood (Firth logistic).
