# Table 2. Multi-scale performance of T1-only baseline, fMRI-only, and deployable multimodal fusion for EZ detection (n=38)

Primary endpoint: **Det(boxDSC > 0.22 & Dice > 0.01)**.  
Co-primary endpoint: **Det(PPV-in-mask ≥ 0.5)**.

| Scale | Metric | T1-only | fMRI-only (fMRIhemi B2a) | Track A (T1+fMRI) | Δ (Track A (T1+fMRI) − T1-only) | p (Track A (T1+fMRI) vs T1-only) |
| --- | --- | --- | --- | --- | --- | --- |
| Vertex-level | Dice similarity coefficient (DSC), mean [95% CI] | 0.322 [0.226, 0.417] | 0.118 [0.070, 0.169] | 0.353 [0.272, 0.435] | 0.031 [-0.011, 0.075] | 0.3063 |
| Cluster-level (boxDSC>0.22 & Dice>0.01) | Predicted clusters per subject, mean [95% CI] | 1.342 [1.158, 1.579] | 1.711 [1.500, 1.947] | 1.605 [1.368, 1.842] | 0.263 [0.026, 0.526] | 0.2604 |
| Cluster-level (boxDSC>0.22 & Dice>0.01) | False-positive clusters per subject, mean [95% CI] | 0.632 [0.395, 0.895] | 1.079 [0.816, 1.368] | 0.684 [0.421, 0.974] | 0.053 [-0.237, 0.342] | 0.8709 |
| Cluster-level (boxDSC>0.22 & Dice>0.01) | Precision (TP clusters / all clusters), % [95% CI] | 41.2% [28.3, 56.2] | 26.2% [17.1, 35.9] | 49.2% [38.2, 61.1] | 8.0% [-3.7, 20.0] | 0.3063 |
| Cluster-level (boxDSC>0.22 & Dice>0.01) | F1 score, mean [95% CI] | 0.472 [0.330, 0.621] | 0.330 [0.216, 0.447] | 0.606 [0.485, 0.723] | 0.134 [0.012, 0.264] | 0.2044 |
| Subject-level (primary) | Detection rate (Det(boxDSC>0.22 & Dice>0.01)), n/N (%) [95% CI] | 21/38 (55.3%) [39.5, 71.1] | 17/38 (44.7%) [28.9, 60.5] | 30/38 (78.9%) [65.8, 89.5] | 0.237 [0.079, 0.395] | 0.0352 (McNemar; n01=10, n10=1) |
| Subject-level (co-primary) | Detection rate (Det(PPV-in-mask≥0.5)), n/N (%) [95% CI] | 16/38 (42.1%) [26.3, 57.9] | 8/38 (21.1%) [7.9, 34.2] | 17/38 (44.7%) [28.9, 60.5] | 0.026 [-0.053, 0.105] | 1.0000 (McNemar; n01=2, n10=1; boot-p=0.7894) |
| Subject-level | Detection rate (Det(distance≤20mm)), n/N (%) [95% CI] | 31/38 (81.6%) [68.4, 92.1] | 27/38 (71.1%) [55.3, 84.2] | 35/38 (92.1%) [81.6, 100.0] | 0.105 [-0.026, 0.237] | 0.3372 (McNemar; n01=6, n10=2; boot-p=0.2024) |
| Subject-level | Pinpointing rate (any cluster COM within lesion), n/N (%) [95% CI] | 22/38 (57.9%) [42.1, 73.7] | 10/38 (26.3%) [13.2, 42.1] | 23/38 (60.5%) [44.7, 76.3] | 0.026 [-0.105, 0.158] | 1.0000 (McNemar; n01=4, n10=3; boot-p=0.8666) |
| Subject-level | Recall@0.15 (lesion Dice union ≥ 0.15), n/N (%) [95% CI] | 22/38 (57.9%) [42.1, 73.7] | 13/38 (34.2%) [18.4, 50.0] | 26/38 (68.4%) [52.6, 81.6] | 0.105 [0.000, 0.237] | 0.3063 (McNemar; n01=5, n10=1; boot-p=0.1340) |

## Notes (for manuscript footnotes)

1) **Cohort (n=38)**: intermediate/difficult cases within the original 52-case seizure-free cohort, restricted to ILAE 1–2.
2) **Reference standard**: resection cavity mask.
3) **Vertex-level DSC**: computed on the lesion hemisphere (rows with `lesion_area_cm2>0` in `vertex_level_results.csv`).
4) **Cluster-level criterion**: a cluster is a TP if `boxDSC > 0.22` and `Dice > 0.01`.
5) **Precision**: TP clusters / all predicted clusters (pooled across subjects).
6) **Sensitivity vs detection rate**: in this seizure-free cohort (all positive), the cluster-level sensitivity (subjects with ≥1 TP cluster / all subjects) is mathematically equivalent to the subject-level detection rate under the same TP definition; therefore, we report only the detection rate to avoid redundancy.
7) **Pinpointing**: at least one predicted cluster has its weighted center-of-mass (COM) within the lesion mask.
8) **Det(PPV-in-mask≥0.5)**: subject-level endpoint; a subject is detected if ≥1 predicted cluster has PPV-in-mask ≥ 0.5.
9) **Det(distance≤20mm)**: subject-level endpoint; a subject is detected if ≥1 predicted cluster has distance-to-lesion ≤ 20 mm (MELD Graph criterion).
10) **Recall@0.15**: subject-level endpoint; `lesion_dice_union ≥ 0.15`.
11) **Uncertainty**: 95% CIs from subject-level bootstrapping (`n_boot=10000`).
12) **Paired statistics (Track A (T1+fMRI) vs T1-only)**:
   - DSC / cluster counts / FP burden: paired sign-flip randomization test.
   - Detection rate / pinpointing / Recall@0.15: exact McNemar’s test (reported with discordant counts n01/n10).
   - Precision / F1: bootstrap-based two-sided p-value from the empirical Δ distribution.
13) **Multiple comparisons**: p values are multiplicity-adjusted using a hierarchical scheme: Holm–Bonferroni for the **primary family** of 3 endpoints (Det(boxDSC>0.22 & Dice safeguard), Det(PPV-in-mask≥0.5), Pinpointing), and Benjamini–Hochberg FDR for the **secondary family** of the remaining metrics.
