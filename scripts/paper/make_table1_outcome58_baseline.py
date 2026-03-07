#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


ROW_ORDER: List[str] = [
    "Patients, n",
    "Sex, n (%)",
    "Male",
    "Female",
    "Age at surgery (years)",
    "Epilepsy duration (years)",
    "Seizure frequency, n (%)",
    "Daily",
    "Weekly",
    "Monthly",
    "SEEG, n (%)",
    "Yes",
    "No",
    "Histopathology, n (%)",
    "FCD I",
    "FCD IIa",
    "FCD IIb",
    "Encephalomalacia",
    # "Polymicrogyria",  # omitted in appeal manuscript (all-zero)
    "Nonspecific",
    "Lesion side, n (%)",
    "Left",
    "Right",
    "Dominant resection cavity location, n (%)",
    "Frontal",
    "Temporal",
    "Parietal",
    "Occipital",
    "Operculo-insular",
    "Central",
    "Cingulate",
    "Multilobar",
    "MRI detectability, n (%)",
    # "Easy",  # omitted in appeal manuscript (restricted cohort)
    "Intermediate",
    "Difficult",
    "Follow-up duration (years)",
    "Postoperative imaging modality for mask, n (%)",
    "MRI",
    "CT",
    "rs-fMRI QC summary",
    "Mean FD (mm)",
    "Censor proportion",
    "Standardized retained time (min)",
    "Retained volumes (standardized to 240)",
    "Resection cavity volume (cm³)",
]


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt_n_pct(k: int, n: int) -> str:
    if n <= 0:
        return "NA"
    return f"{k} ({(100.0 * k / n):.1f}%)"


def _fmt_continuous(values: Iterable[object], *, ndp: int) -> str:
    xs = [v for v in (_safe_float(x) for x in values) if v is not None and math.isfinite(float(v))]
    if not xs:
        return "NA"
    arr = np.asarray(xs, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
    med = float(np.median(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isfinite(sd):
        return f"{mean:.{ndp}f} ± {sd:.{ndp}f}; {med:.{ndp}f} [{q1:.{ndp}f}, {q3:.{ndp}f}]; {vmin:.{ndp}f}–{vmax:.{ndp}f}"
    return f"{mean:.{ndp}f}; {med:.{ndp}f} [{q1:.{ndp}f}, {q3:.{ndp}f}]; {vmin:.{ndp}f}–{vmax:.{ndp}f}"


def _fmt_median_iqr(values: Iterable[object], *, ndp: int, round_input_ndp: Optional[int] = None) -> str:
    xs = [v for v in (_safe_float(x) for x in values) if v is not None and math.isfinite(float(v))]
    if not xs:
        return "NA"
    arr = np.asarray(xs, dtype=float)
    if round_input_ndp is not None:
        arr = np.round(arr, round_input_ndp)
    med = float(np.median(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    return f"{med:.{ndp}f} [{q1:.{ndp}f}, {q3:.{ndp}f}]"


def _mw_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    a = [float(x) for x in a if x is not None and math.isfinite(float(x))]
    b = [float(x) for x in b if x is not None and math.isfinite(float(x))]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _fisher_p(a_yes: int, a_no: int, b_yes: int, b_no: int) -> Optional[float]:
    try:
        return float(fisher_exact([[a_yes, a_no], [b_yes, b_no]], alternative="two-sided")[1])
    except Exception:
        return None


def _fmt_p(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Table 1 baseline (Outcome-linked n=58) stratified by ILAE SF/NSF.")
    ap.add_argument("--meta_csv", default="supplement_table_hs_patient_three_level_metrics.filled.csv")
    ap.add_argument("--rc_csv", default="meld_data/metadata/rc_volume_cm3_outcome58.csv")
    ap.add_argument(
        "--scope_csv",
        default="paper/revision2026_clinical_response_trackAplus20_SELF5F/modelwise_subject_level_intermediate_difficult.csv",
        help="Scope list to define the 58-subject outcome-linked cohort (any duplicated rows ok).",
    )
    ap.add_argument("--out_md", default="paper/revision/table1/table1_outcome58_baseline.md")
    ap.add_argument("--out_tsv", default="paper/revision/table1/table1_outcome58_baseline.tsv")
    ap.add_argument("--out_raw_csv", default="paper/revision/table1/table1_outcome58_baseline_raw.csv")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_csv)
    rc = pd.read_csv(args.rc_csv)
    scope_df = pd.read_csv(args.scope_csv)
    scope_subs = sorted({str(x).strip() for x in scope_df["subject_id"].tolist() if str(x).strip()})

    # Restrict meta to scope subjects and intermediate/difficult only
    meta = meta[meta["subject_id"].astype(str).str.strip().isin(scope_subs)].copy()
    meta["MRIDetectability_norm"] = meta["MRIDetectability"].astype(str).str.strip().str.lower()
    meta = meta[meta["MRIDetectability_norm"].isin(["intermediate", "difficult"])].copy()

    meta["ilae_int"] = pd.to_numeric(meta["Prognosis(ILAE)"], errors="coerce").astype("Int64")
    meta = meta[meta["ilae_int"].notna()].copy()
    meta["outcome_group"] = np.where(meta["ilae_int"].isin([1, 2]), "SF", "NSF")

    # Safety: ensure one row per subject (meta may contain duplicate rows).
    meta = meta.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").copy()

    df = meta.merge(rc[["subject_id", "rc_volume_cm3"]], on="subject_id", how="left")
    df.to_csv(args.out_raw_csv, index=False)

    overall_n = int(len(df))
    sf = df[df["outcome_group"] == "SF"].copy()
    nsf = df[df["outcome_group"] == "NSF"].copy()

    def summarize_bin(mask: pd.Series) -> Tuple[str, str, str, str]:
        mask = mask.astype(bool)
        o = _fmt_n_pct(int(mask.sum()), int(len(mask)))
        a_mask = mask.loc[sf.index]
        b_mask = mask.loc[nsf.index]
        a = _fmt_n_pct(int(a_mask.sum()), int(len(a_mask)))
        b = _fmt_n_pct(int(b_mask.sum()), int(len(b_mask)))
        p = _fmt_p(
            _fisher_p(
                int(a_mask.sum()),
                int(len(a_mask) - int(a_mask.sum())),
                int(b_mask.sum()),
                int(len(b_mask) - int(b_mask.sum())),
            )
        )
        return o, a, b, p

    def summarize_continuous(col: str, *, ndp: int) -> Tuple[str, str, str, str]:
        o = _fmt_continuous(df[col], ndp=ndp)
        a = _fmt_continuous(sf[col], ndp=ndp)
        b = _fmt_continuous(nsf[col], ndp=ndp)
        p = _fmt_p(_mw_p(sf[col].tolist(), nsf[col].tolist()))
        return o, a, b, p

    def summarize_median_iqr(col: str, *, ndp: int, round_input_ndp: Optional[int] = None) -> Tuple[str, str, str, str]:
        o = _fmt_median_iqr(df[col], ndp=ndp, round_input_ndp=round_input_ndp)
        a = _fmt_median_iqr(sf[col], ndp=ndp, round_input_ndp=round_input_ndp)
        b = _fmt_median_iqr(nsf[col], ndp=ndp, round_input_ndp=round_input_ndp)
        p = _fmt_p(_mw_p(sf[col].tolist(), nsf[col].tolist()))
        return o, a, b, p

    rows: Dict[str, Dict[str, str]] = {}

    def add_row(label: str, overall: str, il12: str, il36: str, p: str) -> None:
        rows[label] = {"Characteristic": label, "Overall": overall, "ILAE 1–2": il12, "ILAE 3–6": il36, "p": p}

    add_row("Patients, n", str(overall_n), str(len(sf)), str(len(nsf)), "")

    # ---- Sex ----
    add_row("Sex, n (%)", "", "", "", "")
    sex = df["Sex"].astype(str).str.strip().replace({"男": "Male", "女": "Female"})
    sex = sex.astype(str).str.strip().str.lower()
    add_row("Male", *summarize_bin(sex == "male"))
    add_row("Female", *summarize_bin(sex == "female"))

    # ---- Age + duration ----
    add_row("Age at surgery (years)", *summarize_continuous("AgeAtSurgery_years", ndp=1))
    add_row("Epilepsy duration (years)", *summarize_continuous("EpilepsyDuration_years", ndp=1))

    # ---- Seizure frequency ----
    add_row("Seizure frequency, n (%)", "", "", "", "")
    freq = df["SeizureFrequency"].astype(str).str.strip().str.lower()
    # Chinese -> buckets
    freq = (
        freq.str.replace("每天", "daily", regex=False)
        .str.replace("每周", "weekly", regex=False)
        .str.replace("每月", "monthly", regex=False)
    )
    add_row("Daily", *summarize_bin(freq == "daily"))
    add_row("Weekly", *summarize_bin(freq == "weekly"))
    add_row("Monthly", *summarize_bin(freq == "monthly"))

    # ---- SEEG ----
    add_row("SEEG, n (%)", "", "", "", "")
    seeg = df["SEEG"].astype(str).str.strip().str.lower()
    add_row("Yes", *summarize_bin(seeg == "yes"))
    add_row("No", *summarize_bin(seeg == "no"))

    # ---- Histopathology ----
    add_row("Histopathology, n (%)", "", "", "", "")
    hist = df["Histopathology"].astype(str).str.strip().str.lower()
    add_row("FCD I", *summarize_bin(hist == "fcd i"))
    add_row("FCD IIa", *summarize_bin(hist == "fcd iia"))
    add_row("FCD IIb", *summarize_bin(hist == "fcd iib"))
    add_row("Encephalomalacia", *summarize_bin(hist == "encephalomalacia"))
    add_row("Nonspecific", *summarize_bin(hist == "nonspecific"))

    # ---- Lesion side ----
    add_row("Lesion side, n (%)", "", "", "", "")
    side = df["LesionSide"].astype(str).str.strip().str.lower()
    add_row("Left", *summarize_bin(side == "left"))
    add_row("Right", *summarize_bin(side == "right"))

    # ---- Dominant RC location ----
    add_row("Dominant resection cavity location, n (%)", "", "", "", "")

    def _loc_norm(x: object) -> str:
        s = str(x).strip()
        sl = s.lower()
        if "insul" in sl or "operc" in sl:
            return "Operculo-insular"
        return s

    loc = df["DominantResectionCavityLocation_EZproxy"].map(_loc_norm).astype(str).str.strip()
    for name in ["Frontal", "Temporal", "Parietal", "Occipital", "Operculo-insular", "Central", "Cingulate", "Multilobar"]:
        add_row(name, *summarize_bin(loc == name))

    # ---- MRI detectability ----
    add_row("MRI detectability, n (%)", "", "", "", "")
    det = df["MRIDetectability"].astype(str).str.strip().str.lower()
    add_row("Intermediate", *summarize_bin(det == "intermediate"))
    add_row("Difficult", *summarize_bin(det == "difficult"))

    # ---- Follow-up ----
    # Match manuscript: compute IQR on FollowUp rounded to 2 dp, then format to 2 dp.
    add_row("Follow-up duration (years)", *summarize_median_iqr("FollowUp_years", ndp=2, round_input_ndp=2))

    # ---- Postop imaging modality ----
    add_row("Postoperative imaging modality for mask, n (%)", "", "", "", "")
    postop = df.get("PostopImagingForMask", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip().str.lower()
    postop_norm = np.where(postop.str.contains("ct"), "CT", np.where(postop.str.contains("mri"), "MRI", postop.str.upper()))
    postop_norm = pd.Series(postop_norm, index=df.index)
    add_row("MRI", *summarize_bin(postop_norm == "MRI"))
    add_row("CT", *summarize_bin(postop_norm == "CT"))

    # ---- rs-fMRI QC ----
    add_row("rs-fMRI QC summary", "", "", "", "")
    if "rsfMRI_meanFD" in df.columns:
        add_row("Mean FD (mm)", *summarize_continuous("rsfMRI_meanFD", ndp=3))
    else:
        add_row("Mean FD (mm)", "NA", "NA", "NA", "NA")
    if "rsfMRI_censorRatio" in df.columns:
        add_row("Censor proportion", *summarize_median_iqr("rsfMRI_censorRatio", ndp=3))
    else:
        add_row("Censor proportion", "NA", "NA", "NA", "NA")
    if "rsfMRI_standardizedRetainedTime_min_(5.26min*retained_std240/240)" in df.columns:
        add_row(
            "Standardized retained time (min)",
            *summarize_median_iqr("rsfMRI_standardizedRetainedTime_min_(5.26min*retained_std240/240)", ndp=3),
        )
    else:
        add_row("Standardized retained time (min)", "NA", "NA", "NA", "NA")
    if "rsfMRI_retainedVolumes" in df.columns:
        add_row("Retained volumes (standardized to 240)", *summarize_median_iqr("rsfMRI_retainedVolumes", ndp=1))
    else:
        add_row("Retained volumes (standardized to 240)", "NA", "NA", "NA", "NA")

    # ---- Resection cavity volume ----
    add_row("Resection cavity volume (cm³)", *summarize_continuous("rc_volume_cm3", ndp=1))

    out_df = pd.DataFrame([rows[k] for k in ROW_ORDER if k in rows])
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)

    lines = []
    lines.append("# Table 1. Outcome-linked cohort baseline characteristics (Intermediate + Difficult)")
    lines.append("")
    lines.append("| Characteristic | Overall | ILAE 1–2 | ILAE 3–6 | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in out_df.iterrows():
        lines.append(f"| {r['Characteristic']} | {r['Overall']} | {r['ILAE 1–2']} | {r['ILAE 3–6']} | {r['p']} |")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"n_outcome58={overall_n}")
    print(f"out_md={Path(args.out_md).resolve()}")
    print(f"out_tsv={Path(args.out_tsv).resolve()}")
    print(f"out_raw_csv={Path(args.out_raw_csv).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
