# Supplementary: continuous PPV-in-mask vs outcome (adjusted)

- Cohort N (from input tables): 58

| Index test | No output n/N (%) | PPV median [IQR] in ILAE 1–2 | PPV median [IQR] in ILAE 3–6 | Mann–Whitney p | Adjusted OR per 0.1 PPV | 95% CI | p |
|---|---|---|---|---|---|---|---|
| MELD (self-trained) | 0/58 (0.0%) | 0.425 [0.000, 0.627] | 0.091 [0.000, 0.178] | 0.0296 | 1.32 | [1.07, 1.71] | 0.0006 |
| rs-fMRI | 0/58 (0.0%) | 0.017 [0.000, 0.390] | 0.000 [0.000, 0.000] | 0.0018 | 1.6 | [1.1, 3.27] | 0.0007 |
| Intersection | 29/58 (50.0%) | 0.000 [0.000, 0.052] | 0.000 [0.000, 0.000] | 0.0088 | 1.45 | [1.06, 3.76] | 0.0009 |
| TrackA (fusion) | 0/58 (0.0%) | 0.425 [0.145, 0.682] | 0.000 [0.000, 0.000] | 1.3e-05 | 1.53 | [1.19, 2.17] | 2.3e-05 |

Notes:
- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).
- Inference: p-values use penalized likelihood ratio tests; 95% CIs use profile penalized likelihood.
- Policy: MELD(self-trained) and Intersection use no extra area constraints; TrackA uses paper TrackA outputs and includes the additional 20 non-seizure-free cases.
- PPV is max across clusters; No output treated as PPV=0.
