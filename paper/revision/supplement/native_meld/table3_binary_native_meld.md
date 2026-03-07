# Supplementary: native MELD pipeline (binary endpoints)

- Cohort N (from input tables): 58

| Endpoint | Index test | No output n/N (%) | Resected=Yes in ILAE 1–2 n/N (%) | Resected=Yes in ILAE 3–6 n/N (%) | Adjusted aOR | 95% CI | p |
|---|---|---|---|---|---|---|---|
| Det(boxDSC>0.22) | MELD (native) | 16/58 (27.6%) | 11/38 (28.9%) | 4/20 (20.0%) | 1.56 | [0.448, 6.17] | 0.2533 |
| Det(boxDSC>0.22) | rs-fMRI | 0/58 (0.0%) | 17/38 (44.7%) | 1/20 (5.0%) | 12.5 | [2.49, 133] | 0.0006 |
| Det(boxDSC>0.22) | Intersection | 47/58 (81.0%) | 0/38 (0.0%) | 0/20 (0.0%) | NA | NA | NA |
| Det(boxDSC>0.22) | TrackA (fusion) | 0/58 (0.0%) | 20/38 (52.6%) | 2/20 (10.0%) | 9.79 | [2.32, 64.2] | 0.0006 |
| Det(PPV-in-mask≥0.5) | MELD (native) | 16/58 (27.6%) | 20/38 (52.6%) | 3/20 (15.0%) | 6.13 | [1.72, 28.6] | 0.0024 |
| Det(PPV-in-mask≥0.5) | rs-fMRI | 0/58 (0.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 8.4 | [0.893, 1.12e+03] | 0.0587 |
| Det(PPV-in-mask≥0.5) | Intersection | 47/58 (81.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 17.0 | [1.61, 2.4e+03] | 0.0111 |
| Det(PPV-in-mask≥0.5) | TrackA (fusion) | 0/58 (0.0%) | 19/38 (50.0%) | 1/20 (5.0%) | 11.5 | [2.42, 112] | 0.0006 |
| Pinpointing (COM resected) | MELD (native) | 16/58 (27.6%) | 20/38 (52.6%) | 4/20 (20.0%) | 4.18 | [1.27, 16.4] | 0.0099 |
| Pinpointing (COM resected) | rs-fMRI | 0/58 (0.0%) | 10/38 (26.3%) | 1/20 (5.0%) | 4.34 | [0.859, 43.6] | 0.0556 |
| Pinpointing (COM resected) | Intersection | 47/58 (81.0%) | 8/38 (21.1%) | 0/20 (0.0%) | 17.0 | [1.61, 2.4e+03] | 0.0111 |
| Pinpointing (COM resected) | TrackA (fusion) | 0/58 (0.0%) | 20/38 (52.6%) | 2/20 (10.0%) | 8.24 | [2.09, 48.2] | 0.001 |

Notes:
- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).
- Inference: p-values use penalized likelihood ratio tests; 95% CIs use profile penalized likelihood.
- Policy: MELD(native) and Intersection have no area constraints; TrackA output uses a30_t80.
