# Revision Figure 3 (scope38) — panel statistics & hypothesis tests (draft)

**Cohort (scope38).** n=38, restricted to the seizure-free cohort with **ILAE 1–2** and MRI detectability **Intermediate/Difficult** (i.e., excluding “Easy”). All panels use the same 38 subjects unless otherwise stated.

**Index tests (methods).** T1-only (self-trained MELD), rs-fMRI-only, and deployable multimodal fusion (Track A).

**Binary subject-level endpoints (per subject).** Unless explicitly stated, binary endpoints use an **“any-cluster positive”** rule: a subject is positive if **≥1** predicted cluster meets the criterion.

- **Primary technical endpoint (Detection):** `Det(boxDSC>0.22 & Dice>0.01)`.  
  Implementation note: subject-level `detected_boxdsc` is additionally safeguarded by requiring at least one **box-TP** cluster (`is_tp_boxdsc==True`) with **mask Dice > 0.01** (to avoid inflated boxDSC detections driven by very large boxes with negligible mask overlap).
- **Clinical endpoint:** `Det(PPV-in-mask≥0.5)` (subject positive if any cluster has PPV-in-mask ≥ 0.5).
- **Pinpointing:** subject positive if any predicted cluster has its weighted center-of-mass (COM) located within the reference standard mask.

**Uncertainty.** Detection rates are shown with **Wilson 95% confidence intervals**.

**Paired vs unpaired testing.**
- When comparing **Track A vs T1-only within the same subjects**, we use **exact McNemar’s test** (two-sided) on the paired binary outcomes (discordant counts `n01` vs `n10`).
- When comparing **Intermediate vs Difficult** detection rates (independent groups), we use a **two-sided Fisher’s exact test**.

**Multiple-comparison control (for subgroup panels).** For panels that report *multiple subgroup tests within the same question* (e.g., Intermediate and Difficult subgroups), p-values are **Holm–Bonferroni adjusted within that panel** (family defined as the set of subgroup tests in the panel).

---

## Panel A — Deploy gate (decision plane)

Each point is a subject-level fusion decision produced by the Track A deploy gate. The x-axis is the **T1 confidence** (`t1_conf`), and the y-axis is the **fMRI advantage** over T1 (`fmri_minus_t1`). Vertical/horizontal threshold lines indicate the fixed gate thresholds (`t1_conf_thr` and `diff_thr`). Points are colored by the final deploy action (**keep T1**, **switch to fMRI takeover**, **inject/union**), and the panel reports the counts in each action category. This panel is **descriptive** (no inferential p-value).

## Panel B — Subject-level endpoints (rates)

Bars show the subject-level positive rate for three endpoints: **Detection** (primary), **PPV≥0.5** (clinical), and **Pinpointing**, for each method (T1-only / rs-fMRI-only / Track A).  
Significance marking (if shown) compares **Track A vs T1-only** for the **primary Detection endpoint** using an **exact McNemar test** (paired, two-sided).

## Panel C — Complementarity (2×2) under the primary endpoint

The 2×2 table cross-tabulates primary Detection (Det(boxDSC>0.22 & Dice>0.01)) for **T1-only vs Track A**:

- **rescued (TrackA-only):** T1-only not detected, Track A detected (`n01`)
- **hurt (T1-only):** T1-only detected, Track A not detected (`n10`)
- **both / neither** accordingly

The p-value (and star, if shown) corresponds to the **exact McNemar test** for Track A vs T1-only on the paired detection labels.

## Panel D — Intermediate vs Difficult (T1 drop vs fMRI stability)

Two stacked bar charts show the Detection rate by MRI detectability group (**Intermediate vs Difficult**) separately for:

1) **T1-only**, and  
2) **rs-fMRI-only**.

For each method, the Intermediate vs Difficult difference is tested with a **two-sided Fisher’s exact test**. The two method-specific p-values (T1-only and rs-fMRI-only) are **Holm–Bonferroni adjusted within Panel D** (family size = 2).

## Panel E — Complementarity by detectability (T1 vs Track A)

For each detectability group (**Intermediate**, **Difficult**), stacked bars show the proportion of subjects in four mutually exclusive categories under the primary Detection endpoint:

- both detected  
- T1-only detected (**hurt**)  
- Track A-only detected (**rescued**)  
- neither detected

Within each subgroup, significance compares **Track A vs T1-only** using **exact McNemar’s test**. The subgroup p-values are **Holm–Bonferroni adjusted within Panel E** (family size = 2: Intermediate + Difficult).

## Panel F — SEEG-stratified detection

Dot-and-whisker plots show Detection rates (with Wilson 95% CI) stratified by **SEEG (Yes/No)** for each method (T1-only / rs-fMRI-only / Track A).  
Within each SEEG subgroup, significance compares **Track A vs T1-only** using **exact McNemar’s test**, with **Holm–Bonferroni adjustment within Panel F** (family size = 2: SEEG Yes + SEEG No).

**Endpoint:** `Det(boxDSC>0.22 & Dice>0.01)` (primary technical detection; any-cluster positive).

**SEEG = Yes** (n=33):
- T1-only: 17/33 (51.5%; Wilson 95% CI 35.2–67.5)
- rs-fMRI-only: 15/33 (45.5%; 29.8–62.0)
- Track A: 25/33 (75.8%; 59.0–87.2)
- Track A vs T1-only (paired): McNemar p=0.0215; discordant counts n01=9 (rescued), n10=1 (hurt)  
  Holm-adjusted within Panel F: p_adj=0.0430

**SEEG = No** (n=5):
- T1-only: 4/5 (80.0%; 37.6–96.4)
- rs-fMRI-only: 2/5 (40.0%; 11.8–76.9)
- Track A: 5/5 (100.0%; 56.6–100.0)
- Track A vs T1-only (paired): McNemar p=1.0000; n01=1, n10=0  
  Holm-adjusted within Panel F: p_adj=1.0000  

> Note: the non-SEEG subgroup is very small (n=5), so inference is exploratory and the CI is wide.
