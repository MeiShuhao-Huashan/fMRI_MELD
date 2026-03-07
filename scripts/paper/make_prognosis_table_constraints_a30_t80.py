#!/usr/bin/env python3
"""
Make a prognosis association table (ILAE12 vs ILAE3456) under consistent deploy constraints (a30_t80).

Models included (4):
  - Self-trained MELD (post-processed with a30_t80 constraints)
  - rs-fMRI (a30_t80)
  - Intersection (MELD ∩ fMRI, both constrained)
  - TrackA (takeover fusion using constrained MELD + constrained fMRI)

For multi-cluster handling:
  - Binary endpoints use ThreeLevelEvaluator's subject-level flags:
      detected_boxdsc = (any cluster with boxDSC>0.22)
      detected_ppv50  = (any cluster with PPV-in-mask>=0.5)
      pinpointed      = (any cluster pinpointing)
    i.e. "any-cluster positive" after enforcing K<=3 and area budgets.
  - Continuous endpoint uses max_ppv_in_mask (max across clusters), and is modeled per 0.1 PPV.

Optional safeguard:
  - For the boxDSC endpoint, you may require **mask Dice > threshold** in addition to `boxDSC>0.22`
    to avoid false positives where a large predicted cluster yields a high bounding-box overlap.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.utils.firth_inference import firth_or_ci_p  # noqa: E402


GOOD_SET = {1, 2}


def _parse_ilae(x: object) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = int(float(s))
        if 1 <= v <= 6:
            return v
    except Exception:
        return None
    return None


def _sex_male(x: object) -> Optional[int]:
    s = str(x).strip()
    if s in {"Male", "male", "男", "M", "m"}:
        return 1
    if s in {"Female", "female", "女", "F", "f"}:
        return 0
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _penalized_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    mu = _sigmoid(eta)
    eps = 1e-12
    ll = float(np.sum(y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
    w = mu * (1.0 - mu)
    I = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(I + np.eye(I.shape[0]) * 1e-12)
    if sign <= 0:
        return -np.inf
    return ll + 0.5 * float(logdet)


def firth_logistic(X: np.ndarray, y: np.ndarray, *, max_iter: int = 200, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int]:
    n, p = X.shape
    beta = np.zeros(p, dtype=float)
    prev_obj = _penalized_loglik(beta, X, y)

    I_inv = np.eye(p, dtype=float)
    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = _sigmoid(eta)
        w = mu * (1.0 - mu)
        w = np.clip(w, 1e-12, None)

        I = X.T @ (w[:, None] * X)
        I += np.eye(p) * 1e-9
        I_inv = np.linalg.inv(I)

        WX = (np.sqrt(w)[:, None]) * X
        h = np.sum((WX @ I_inv) * WX, axis=1)
        adj = (0.5 - mu) * h
        U = X.T @ (y - mu + adj)
        delta = I_inv @ U

        step = 1.0
        beta_new = beta + step * delta
        obj_new = _penalized_loglik(beta_new, X, y)
        while obj_new < prev_obj and step > 1e-6:
            step *= 0.5
            beta_new = beta + step * delta
            obj_new = _penalized_loglik(beta_new, X, y)

        beta = beta_new
        if np.max(np.abs(step * delta)) < tol:
            se = np.sqrt(np.diag(I_inv))
            return beta, se, it
        prev_obj = obj_new

    se = np.sqrt(np.diag(I_inv))
    return beta, se, max_iter


def _fmt_n_pct(k: int, n: int) -> str:
    if n <= 0:
        return "NA"
    return f"{k}/{n} ({(100.0 * k / n):.1f}%)"


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _median_iqr(x: pd.Series, *, ndp: int = 3) -> str:
    v = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if v.empty:
        return "NA"
    med = float(v.median())
    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    return f"{med:.{ndp}f} [{q1:.{ndp}f}, {q3:.{ndp}f}]"


def _mw_p(a: pd.Series, b: pd.Series) -> float:
    a2 = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b2 = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a2) == 0 or len(b2) == 0:
        return float("nan")
    return float(mannwhitneyu(a2, b2, alternative="two-sided").pvalue)


def _load_meta_scope(meta_csv: Path, rc_csv: Path, subject_ids: List[str]) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv, encoding="utf-8")
    meta["subject_id"] = meta["subject_id"].astype(str).str.strip()
    meta["MRIDetectability_norm"] = meta["MRIDetectability"].astype(str).str.strip().str.lower()
    meta["ilae_int"] = meta["Prognosis(ILAE)"].map(_parse_ilae)
    meta = meta[meta["subject_id"].isin(set(subject_ids))].copy()
    meta = meta[meta["MRIDetectability_norm"].isin(["intermediate", "difficult"])].copy()
    meta = meta[meta["ilae_int"].notna()].copy()
    meta = meta.sort_values("subject_id").drop_duplicates("subject_id", keep="first").copy()

    meta["outcome_group"] = np.where(meta["ilae_int"].isin(list(GOOD_SET)), "ILAE12", "ILAE3456")
    meta["sf"] = (meta["outcome_group"] == "ILAE12").astype(int)
    meta["sex_male"] = meta["Sex"].map(_sex_male)

    rc = pd.read_csv(rc_csv)
    meta = meta.merge(rc[["subject_id", "rc_volume_cm3"]], on="subject_id", how="left")
    meta["log_rc"] = np.log(pd.to_numeric(meta["rc_volume_cm3"], errors="coerce").astype(float) + 1e-6)
    meta = meta[meta["rc_volume_cm3"].notna() & meta["sex_male"].notna()].copy()
    return meta


def _load_eval_subject_table(
    eval_dir: Path,
    *,
    model_name: str,
    boxdsc_min_cluster_dice: float = 0.0,
) -> pd.DataFrame:
    df = pd.read_csv(eval_dir / "subject_level_results.csv")
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    # For optional boxDSC+DICE safeguard we may need per-cluster dice.
    cl_path = eval_dir / "cluster_level_results.csv"
    cl = pd.read_csv(cl_path) if cl_path.exists() else pd.DataFrame()
    if not cl.empty:
        cl["subject_id"] = cl["subject_id"].astype(str).str.strip()

    det_box = df["detected_boxdsc"].astype(bool)
    if float(boxdsc_min_cluster_dice) > 0.0 and not cl.empty:
        if "is_tp_boxdsc" in cl.columns:
            cl_tp = cl["is_tp_boxdsc"].astype(str).str.strip().str.lower() == "true"
        else:
            cl_tp = pd.Series(False, index=cl.index, dtype=bool)
        if "dice" in cl.columns:
            cl_dice = pd.to_numeric(cl["dice"], errors="coerce").fillna(0.0).astype(float)
        else:
            cl_dice = pd.Series(0.0, index=cl.index, dtype=float)
        hit_ids = set(cl.loc[cl_tp & (cl_dice > float(boxdsc_min_cluster_dice)), "subject_id"].astype(str).tolist())
        det_box = df["subject_id"].isin(hit_ids)
    out = pd.DataFrame(
        {
            "subject_id": df["subject_id"],
            "model": model_name,
            "no_prediction": df["n_clusters"].astype(float).fillna(0.0) <= 0.0,
            "det_boxdsc_022": det_box.astype(bool),
            "det_ppv_05": df["detected_ppv50"].astype(bool),
            "pinpointed": df["pinpointed"].astype(bool),
            "ppv": pd.to_numeric(df["max_ppv_in_mask"], errors="coerce").fillna(0.0).astype(float),
        }
    )
    # no-pred policy: treat as not resected and ppv=0
    no_pred = out["no_prediction"].astype(bool)
    for col in ["det_boxdsc_022", "det_ppv_05", "pinpointed"]:
        out.loc[no_pred, col] = False
    out.loc[no_pred, "ppv"] = 0.0
    return out


def _fit_binary(df: pd.DataFrame, endpoint_col: str, model_order: List[str], labels: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in model_order:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        n_total = int(len(g))
        n_no = int(g["no_prediction"].astype(bool).sum())
        sf = g[g["outcome_group"] == "ILAE12"]
        nsf = g[g["outcome_group"] == "ILAE3456"]
        sf_yes = int(sf[endpoint_col].astype(bool).sum())
        nsf_yes = int(nsf[endpoint_col].astype(bool).sum())

        y = g["sf"].to_numpy(dtype=float)
        x_endpoint = g[endpoint_col].astype(bool).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), x_endpoint, g["log_rc"].to_numpy(dtype=float), g["sex_male"].to_numpy(dtype=float)])
        if float(np.std(x_endpoint)) == 0.0:
            aor = ci_lo = ci_hi = p = float("nan")
        else:
            infer = firth_or_ci_p(X, y, idx=1)
            aor = float(infer["or"])
            ci_lo = float(infer["ci_lo"])
            ci_hi = float(infer["ci_hi"])
            p = float(infer["p_plr"])

        rows.append(
            {
                "Model": labels.get(model, model),
                "No output n/N (%)": _fmt_n_pct(n_no, n_total),
                "Resection=Yes in ILAE12 n/N (%)": _fmt_n_pct(sf_yes, int(len(sf))),
                "Resection=Yes in ILAE3456 n/N (%)": _fmt_n_pct(nsf_yes, int(len(nsf))),
                "Adjusted aOR": f"{aor:.3g}" if math.isfinite(aor) else "NA",
                "95% CI": f"[{ci_lo:.3g}, {ci_hi:.3g}]" if math.isfinite(ci_lo) and math.isfinite(ci_hi) else "NA",
                "p": _fmt_p(p),
            }
        )
    return pd.DataFrame(rows)


def _fit_ppv_cont(df: pd.DataFrame, model_order: List[str], labels: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in model_order:
        g = df[df["model"] == model].copy()
        if g.empty:
            continue
        sf = g[g["outcome_group"] == "ILAE12"]
        nsf = g[g["outcome_group"] == "ILAE3456"]
        mw_p = _mw_p(sf["ppv"], nsf["ppv"])

        y = g["sf"].to_numpy(dtype=float)
        x_ppv01 = (pd.to_numeric(g["ppv"], errors="coerce").fillna(0.0).astype(float) / 0.1).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), x_ppv01, g["log_rc"].to_numpy(dtype=float), g["sex_male"].to_numpy(dtype=float)])
        if float(np.std(x_ppv01)) == 0.0:
            aor = ci_lo = ci_hi = p = float("nan")
        else:
            infer = firth_or_ci_p(X, y, idx=1)
            aor = float(infer["or"])
            ci_lo = float(infer["ci_lo"])
            ci_hi = float(infer["ci_hi"])
            p = float(infer["p_plr"])

        rows.append(
            {
                "Model": labels.get(model, model),
                "PPV median [IQR] in ILAE12": _median_iqr(sf["ppv"], ndp=3),
                "PPV median [IQR] in ILAE3456": _median_iqr(nsf["ppv"], ndp=3),
                "Mann–Whitney p": _fmt_p(mw_p),
                "Adjusted OR per 0.1 PPV": f"{aor:.3g}" if math.isfinite(aor) else "NA",
                "95% CI": f"[{ci_lo:.3g}, {ci_hi:.3g}]" if math.isfinite(ci_lo) and math.isfinite(ci_hi) else "NA",
                "p": _fmt_p(p),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default="supplement_table_hs_patient_three_level_metrics.filled.csv")
    ap.add_argument("--rc_csv", default="meld_data/metadata/rc_volume_cm3_outcome58.csv")
    ap.add_argument(
        "--meld_eval_dir",
        default="meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface_constraints_a30_t80",
    )
    ap.add_argument(
        "--fmri_eval_dir",
        default="meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg",
    )
    ap.add_argument(
        "--inter_eval_dir",
        default="meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_fmri_intersection_constraints_a30_t80",
    )
    ap.add_argument(
        "--tracka_eval_dir",
        default="meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_constraints_a30_t80",
    )
    ap.add_argument("--out_md", default="paper/revision/table2/prognosis_table_constraints_a30_t80.md")
    ap.add_argument("--out_dir", default="paper/revision/table2")
    ap.add_argument("--meld_label", default="MELD (self-trained, constrained a30_t80)")
    ap.add_argument("--fmri_label", default="rs-fMRI (constrained a30_t80)")
    ap.add_argument("--intersection_label", default="Intersection (constrained)")
    ap.add_argument("--tracka_label", default="TrackA (fusion, constrained)")
    ap.add_argument(
        "--out_tag",
        default="constraints_a30_t80",
        help="Tag used in output filenames, e.g. prognosis_table_boxdsc022_<tag>.tsv",
    )
    ap.add_argument(
        "--constraints_note",
        default="K<=3, AREA_MAX_CLUSTER_CM2=30, AREA_MAX_TOTAL_CM2=80, min_vertices=100; trimming by probability when exceeding budgets.",
        help="Free-text note describing post-processing constraints/policy for this run.",
    )
    ap.add_argument(
        "--boxdsc_min_cluster_dice",
        type=float,
        default=0.0,
        help="Optional extra requirement for the boxDSC endpoint: a subject is boxDSC-detected only if it has a TP(boxDSC>0.22) cluster with cluster Dice > this threshold (e.g. 0.01).",
    )
    ap.add_argument(
        "--title",
        default="Prognosis association (adjusted)",
        help="Markdown title line written to --out_md.",
    )
    args = ap.parse_args()

    model_order = ["meld_selftrained_constraints", "fmri", "intersection", "trackA"]
    labels = {
        "meld_selftrained_constraints": str(args.meld_label),
        "fmri": str(args.fmri_label),
        "intersection": str(args.intersection_label),
        "trackA": str(args.tracka_label),
    }

    meld = _load_eval_subject_table(
        Path(args.meld_eval_dir).resolve(),
        model_name="meld_selftrained_constraints",
        boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice),
    )
    fmri = _load_eval_subject_table(
        Path(args.fmri_eval_dir).resolve(),
        model_name="fmri",
        boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice),
    )
    inter = _load_eval_subject_table(
        Path(args.inter_eval_dir).resolve(),
        model_name="intersection",
        boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice),
    )
    tracka = _load_eval_subject_table(
        Path(args.tracka_eval_dir).resolve(),
        model_name="trackA",
        boxdsc_min_cluster_dice=float(args.boxdsc_min_cluster_dice),
    )
    metrics = pd.concat([meld, fmri, inter, tracka], ignore_index=True)

    subject_ids = sorted(metrics["subject_id"].unique().tolist())
    meta = _load_meta_scope(Path(args.meta_csv).resolve(), Path(args.rc_csv).resolve(), subject_ids)
    merged = metrics.merge(meta, on="subject_id", how="inner")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tbl_box = _fit_binary(merged, "det_boxdsc_022", model_order, labels)
    tbl_ppv = _fit_binary(merged, "det_ppv_05", model_order, labels)
    tbl_pin = _fit_binary(merged, "pinpointed", model_order, labels)
    tbl_cont = _fit_ppv_cont(merged, model_order, labels)

    tag = str(args.out_tag).strip()
    if not tag:
        tag = "constraints_a30_t80"

    tbl_box.to_csv(out_dir / f"prognosis_table_boxdsc022_{tag}.tsv", sep="\t", index=False)
    tbl_ppv.to_csv(out_dir / f"prognosis_table_ppv05_{tag}.tsv", sep="\t", index=False)
    tbl_pin.to_csv(out_dir / f"prognosis_table_pinpointing_{tag}.tsv", sep="\t", index=False)
    tbl_cont.to_csv(out_dir / f"prognosis_table_ppv_continuous_{tag}.tsv", sep="\t", index=False)

    md_lines: List[str] = []
    md_lines.append(f"# {args.title}")
    md_lines.append("")
    md_lines.append(f"- Cohort: Intermediate+Difficult; n={meta['subject_id'].nunique()} (ILAE12={int((meta['outcome_group']=='ILAE12').sum())}, ILAE3456={int((meta['outcome_group']=='ILAE3456').sum())})")
    md_lines.append(f"- Constraints/policy: {args.constraints_note}")
    md_lines.append("- Regression: Firth logistic; covariates = log(RC_volume_cm3) + Sex(Male=1).")
    md_lines.append("")
    if float(args.boxdsc_min_cluster_dice) > 0.0:
        md_lines.append(f"## A. Det(boxDSC>0.22 & Dice>{float(args.boxdsc_min_cluster_dice):g})")
    else:
        md_lines.append("## A. Det(boxDSC>0.22)")
    md_lines.append("")
    md_lines.extend(_md_table(tbl_box))
    md_lines.append("")
    md_lines.append("## B. Det(PPV-in-mask≥0.5)")
    md_lines.append("")
    md_lines.extend(_md_table(tbl_ppv))
    md_lines.append("")
    md_lines.append("## C. Pinpointing (any-cluster COM resected)")
    md_lines.append("")
    md_lines.extend(_md_table(tbl_pin))
    md_lines.append("")
    md_lines.append("## D. Continuous PPV-in-mask (max across clusters)")
    md_lines.append("")
    md_lines.extend(_md_table(tbl_cont))
    md_lines.append("")
    md_lines.append("Notes:")
    md_lines.append("- Multi-cluster handling: binary endpoints are `any-cluster positive`; continuous PPV uses `max_ppv_in_mask` across clusters.")
    md_lines.append("- `No output` treated as not resected; PPV set to 0.")
    md_lines.append("- Inference: p-values use penalized likelihood ratio tests; 95% CIs use profile penalized likelihood (Firth logistic).")

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
