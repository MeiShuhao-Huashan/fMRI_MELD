#!/usr/bin/env python3
"""
Regenerate the fMRI ablation *matrix* (Supplementary Table 2) on the revised scope38 cohort:

- Cohort: the 38-subject seizure-free subset restricted to MRI detectability in {Intermediate, Difficult}
  (subject list default: paper/revision/table3/final/table3_scope38_subject_ids.tsv).
- Ablations: V2 fMRI models under *real-deploy fMRIhemi* evaluation (same 19 ablations as the original table).
- Overall metrics: computed from MELD three-level evaluation CSVs (vertex/subject/cluster) restricted to scope38.
- Rescue metrics: computed *within the baseline T1 detection-failure set* (scope38):
    - First, define the primary "T1 failure" subset as baseline Det(boxDSC>0.22)=False (n=17).
    - Then, for each other endpoint, define failures *within that subset* to avoid mixing:
        - PPV failures within T1 Det(boxDSC) fails
        - Pinpointing failures within T1 Det(boxDSC) fails
- Ranking: average ranks for ties (lower is better); Mean rescue rank = mean of the three ranks.

Outputs:
  - paper/supplement_fmri_ablation/tableS_fmri_ablation_matrix_scope38.tsv
  - paper/supplement_fmri_ablation/Supplementary_table2_fmri_ablation_scope38.xlsx
  - paper/supplement_fmri_ablation/ablation_traceability_scope38.json

Run:
  source env.sh
  python paper/supplement_fmri_ablation/make_fmri_ablation_matrix_scope38.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Ensure repo root is importable when running as a script.
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Reuse the original Table2 metric utilities (no pandas/numpy dependency).
from paper.table2.generate_table2 import (  # type: ignore
    _compute_metrics,
    _load_cluster_rows,
    _load_subject_rows,
)


def _read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [{k: str(v) for k, v in row.items()} for row in r]


def _read_scope_subject_ids(path: Path) -> List[str]:
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames or "subject_id" not in set(r.fieldnames):
            raise ValueError(f"Expected a 'subject_id' column in: {path}")
        ids = [str(row["subject_id"]).strip() for row in r]
    ids = [x for x in ids if x]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate subject_id in: {path}")
    return ids


def _rank_average_ties_desc(values_by_id: Dict[str, int]) -> Dict[str, float]:
    """
    Rank with 1=best (largest value), using average ranks for ties.
    """
    items = sorted(values_by_id.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: Dict[str, float] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i
        v = items[i][1]
        while j < n and items[j][1] == v:
            j += 1
        # positions are 1-based
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[items[k][0]] = float(avg_rank)
        i = j
    return ranks


def _fmt_rescue(n: int, denom: int) -> str:
    if denom <= 0:
        return f"{n}/{denom}"
    return f"{n}/{denom} ({100.0 * float(n) / float(denom):.1f}%)"


def _fmt_num(x: float, *, decimals: int) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{decimals}f}"


def _fmt_rank(x: float) -> str:
    # Excel expects numeric; keep compact string (no trailing zeros).
    if not math.isfinite(float(x)):
        return "nan"
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _iter_xlsx_shared_strings(template_xlsx: Path) -> List[str]:
    with zipfile.ZipFile(template_xlsx, "r") as z:
        xml = z.read("xl/sharedStrings.xml")
    root = ET.fromstring(xml)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out: List[str] = []
    for si in root.findall("m:si", ns):
        parts = [t.text or "" for t in si.findall(".//m:t", ns)]
        out.append("".join(parts))
    return out


def _find_cell(row: ET.Element, ref: str, ns: Dict[str, str]) -> ET.Element:
    for c in row.findall("m:c", ns):
        if c.attrib.get("r") == ref:
            return c
    raise KeyError(f"Missing cell {ref} in row r={row.attrib.get('r')}")


def _set_cell_number(cell: ET.Element, value: str, ns: Dict[str, str]) -> None:
    # Remove shared-string typing if present.
    if "t" in cell.attrib:
        cell.attrib.pop("t", None)
    # Keep any existing formula (<f>) untouched.
    v = cell.find("m:v", ns)
    if v is None:
        v = ET.SubElement(cell, f"{{{ns['m']}}}v")
    v.text = str(value)
    # Remove inline strings if any.
    is_ = cell.find("m:is", ns)
    if is_ is not None:
        cell.remove(is_)


def _set_cell_inline_string(cell: ET.Element, text: str, ns: Dict[str, str]) -> None:
    cell.attrib["t"] = "inlineStr"
    # Remove numeric value if present.
    v = cell.find("m:v", ns)
    if v is not None:
        cell.remove(v)
    # Replace inline string
    is_ = cell.find("m:is", ns)
    if is_ is not None:
        cell.remove(is_)
    is_ = ET.SubElement(cell, f"{{{ns['m']}}}is")
    t = ET.SubElement(is_, f"{{{ns['m']}}}t")
    t.text = text


def _update_xlsx(
    *,
    template_xlsx: Path,
    out_xlsx: Path,
    rows_sorted: Sequence[Dict[str, object]],
    rescue_n_fail_det: int,
    rescue_n_fail_pin_within_det: int,
    rescue_n_fail_ppv_within_det: int,
) -> None:
    def _col_to_number(col: str) -> int:
        n = 0
        for ch in col:
            n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
        return n

    def _cell_col(ref: str) -> str:
        return "".join(ch for ch in ref if ch.isalpha())

    def _get_or_create_cell(row: ET.Element, ref: str, ns: Dict[str, str]) -> ET.Element:
        for c in row.findall("m:c", ns):
            if c.attrib.get("r") == ref:
                return c
        c_new = ET.Element(f"{{{ns['m']}}}c", {"r": ref})
        target = _col_to_number(_cell_col(ref))
        cells = list(row.findall("m:c", ns))
        for idx, c in enumerate(cells):
            r_attr = c.attrib.get("r", "")
            if not r_attr:
                continue
            if _col_to_number(_cell_col(r_attr)) > target:
                row.insert(idx, c_new)
                break
        else:
            row.append(c_new)
        return c_new

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_xlsx, out_xlsx)

    with zipfile.ZipFile(out_xlsx, "r") as z:
        sheet_xml = z.read("xl/worksheets/sheet1.xml")

    # Preserve default namespace (avoid ns0 prefixes).
    ET.register_namespace("", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(sheet_xml)
    sheet_data = root.find("m:sheetData", ns)
    if sheet_data is None:
        raise ValueError("sheet1.xml missing <sheetData>")

    # Update header rescue columns to reflect denominators.
    header_row = sheet_data.find("m:row[@r='1']", ns)
    if header_row is not None:
        header_row.attrib["spans"] = "1:16"
        j1 = _find_cell(header_row, "J1", ns)
        k1 = _find_cell(header_row, "K1", ns)
        _set_cell_inline_string(j1, f"Rescue Det(boxDSC>0.22) among T1 Det(boxDSC) fails (n={rescue_n_fail_det})", ns)
        _set_cell_inline_string(
            k1,
            f"Rescue Pinpointing among T1 pinpointing fails within T1 Det(boxDSC) fails (n={rescue_n_fail_pin_within_det})",
            ns,
        )

        o1 = _get_or_create_cell(header_row, "O1", ns)
        p1 = _get_or_create_cell(header_row, "P1", ns)
        _set_cell_inline_string(
            o1,
            f"Rescue Det(PPV>=0.5) among T1 Det(PPV) fails within T1 Det(boxDSC) fails (n={rescue_n_fail_ppv_within_det})",
            ns,
        )
        _set_cell_inline_string(p1, "Rescue Det(PPV>=0.5) rank (1=best)", ns)

    # Expand the worksheet used-range dimension to include appended PPV columns.
    last_row = len(rows_sorted) + 1  # header row
    dim = root.find("m:dimension", ns)
    if dim is not None:
        dim.attrib["ref"] = f"A1:P{last_row}"

    # Fill rows 2..N in the XLSX in the (already sorted) output order,
    # to mirror the original "sorted-by-mean-rank" presentation.
    data_rows = [row for row in sheet_data.findall("m:row", ns) if str(row.attrib.get("r", "")).isdigit() and int(row.attrib["r"]) >= 2]
    if len(data_rows) != len(rows_sorted):
        raise ValueError(
            f"Template expects {len(data_rows)} data rows, but computed {len(rows_sorted)} rows. "
            "Refuse to write a mismatched XLSX."
        )

    for row, d in zip(sorted(data_rows, key=lambda r: int(r.attrib["r"])), rows_sorted):
        r = str(row.attrib["r"])
        row.attrib["spans"] = "1:16"

        # Update A/B/C as inline strings (safe even if the template used shared strings).
        _set_cell_inline_string(_find_cell(row, f"A{r}", ns), str(d["Experiment"]), ns)
        _set_cell_inline_string(_find_cell(row, f"B{r}", ns), str(d["Group"]), ns)
        _set_cell_inline_string(_find_cell(row, f"C{r}", ns), str(d["Description"]), ns)

        _set_cell_number(_find_cell(row, f"D{r}", ns), str(d["Vertex DSC (mean)"]), ns)
        _set_cell_number(_find_cell(row, f"E{r}", ns), str(d["Det(boxDSC>0.22) (overall)"]), ns)
        _set_cell_number(_find_cell(row, f"F{r}", ns), str(d["Pinpointing (overall)"]), ns)
        _set_cell_number(_find_cell(row, f"G{r}", ns), str(d["FP clusters/pt (overall)"]), ns)
        _set_cell_number(_find_cell(row, f"H{r}", ns), str(d["Cluster precision (overall)"]), ns)
        _set_cell_number(_find_cell(row, f"I{r}", ns), str(d["Cluster F1 (overall)"]), ns)

        _set_cell_inline_string(
            _find_cell(row, f"J{r}", ns),
            str(d[f"Rescue Det(boxDSC>0.22) among T1 Det(boxDSC) fails (n={rescue_n_fail_det})"]),
            ns,
        )
        _set_cell_inline_string(
            _find_cell(row, f"K{r}", ns),
            str(
                d[
                    f"Rescue Pinpointing among T1 pinpointing fails within T1 Det(boxDSC) fails (n={rescue_n_fail_pin_within_det})"
                ]
            ),
            ns,
        )

        _set_cell_number(_find_cell(row, f"L{r}", ns), str(d["Rescue Det(boxDSC) rank (1=best)"]), ns)
        _set_cell_number(_find_cell(row, f"M{r}", ns), str(d["Rescue Pinpointing rank (1=best)"]), ns)

        o_cell = _get_or_create_cell(row, f"O{r}", ns)
        p_cell = _get_or_create_cell(row, f"P{r}", ns)
        _set_cell_inline_string(
            o_cell,
            str(
                d[
                    f"Rescue Det(PPV>=0.5) among T1 Det(PPV) fails within T1 Det(boxDSC) fails (n={rescue_n_fail_ppv_within_det})"
                ]
            ),
            ns,
        )
        _set_cell_number(p_cell, str(d["Rescue Det(PPV>=0.5) rank (1=best)"]), ns)

        # Mean rescue rank now averages three ranks: Det(boxDSC), Pinpointing, and Det(PPV>=0.5).
        n_cell = _find_cell(row, f"N{r}", ns)
        f = n_cell.find("m:f", ns)
        if f is None:
            f = ET.SubElement(n_cell, f"{{{ns['m']}}}f")
        f.text = f"(L{r}+M{r}+P{r})/3"
        _set_cell_number(n_cell, str(d["Mean rescue rank"]), ns)

    # Write back updated XML (replace in-place).
    out_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    tmp = out_xlsx.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(out_xlsx, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "xl/worksheets/sheet1.xml":
                zout.writestr(item, out_bytes)
            else:
                zout.writestr(item, zin.read(item.filename))
    tmp.replace(out_xlsx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scope_subjects_tsv",
        default="paper/revision/table3/final/table3_scope38_subject_ids.tsv",
        type=Path,
    )
    ap.add_argument(
        "--ablation_trace_json",
        default="paper/supplement_fmri_ablation/ablation_traceability_real_deploy_fmrihemi.json",
        type=Path,
    )
    ap.add_argument(
        "--ablation_base_tsv",
        default="paper/supplement_fmri_ablation/tableS_fmri_ablation_real_deploy_fmrihemi.tsv",
        type=Path,
    )
    ap.add_argument(
        "--table2_audit_json",
        default="paper/table2/table2_audit.json",
        type=Path,
        help="Used to locate the baseline T1 eval_dir for defining the T1-failure set.",
    )
    ap.add_argument(
        "--out_tsv",
        default="paper/supplement_fmri_ablation/tableS_fmri_ablation_matrix_scope38.tsv",
        type=Path,
    )
    ap.add_argument(
        "--template_xlsx",
        default="paper/supplement_fmri_ablation/Supplementary_table2_fmri_ablation.xlsx",
        type=Path,
    )
    ap.add_argument(
        "--out_xlsx",
        default="paper/supplement_fmri_ablation/Supplementary_table2_fmri_ablation_scope38.xlsx",
        type=Path,
    )
    ap.add_argument(
        "--out_trace_json",
        default="paper/supplement_fmri_ablation/ablation_traceability_scope38.json",
        type=Path,
    )
    args = ap.parse_args()

    scope_subjects = _read_scope_subject_ids((PROJECT_DIR / args.scope_subjects_tsv).resolve())
    if len(scope_subjects) != 38:
        raise ValueError(f"Expected 38 subjects in scope, got {len(scope_subjects)} from {args.scope_subjects_tsv}")

    trace_in = json.loads((PROJECT_DIR / args.ablation_trace_json).read_text(encoding="utf-8"))
    ablations = trace_in.get("ablations", [])
    if not ablations:
        raise ValueError(f"No ablations found in: {args.ablation_trace_json}")

    base_rows = _read_tsv_rows((PROJECT_DIR / args.ablation_base_tsv).resolve())
    base_meta: Dict[str, Dict[str, str]] = {str(r["Experiment"]).strip(): r for r in base_rows}

    table2_audit = json.loads((PROJECT_DIR / args.table2_audit_json).read_text(encoding="utf-8"))
    t1_eval_dir_rel = str(table2_audit["inputs"]["t1_eval_dir"])
    t1_eval_dir = (PROJECT_DIR / Path(t1_eval_dir_rel)).resolve()
    t1_sub = _load_subject_rows(t1_eval_dir)
    t1_det_fail_ids = [sid for sid in scope_subjects if not t1_sub[sid].detected_boxdsc]
    n_fail_det = len(t1_det_fail_ids)
    t1_det_fail_set = set(t1_det_fail_ids)

    # Avoid mixing rescue metrics by keeping the rescue cohort fixed to baseline T1 detection failures,
    # then defining PPV/pinpointing failures *within that subset*.
    t1_ppv_fail_within_det_ids = [sid for sid in t1_det_fail_ids if not t1_sub[sid].detected_ppv50]
    t1_pin_fail_within_det_ids = [sid for sid in t1_det_fail_ids if not t1_sub[sid].pinpointed]
    n_fail_ppv_within_det = len(t1_ppv_fail_within_det_ids)
    n_fail_pin_within_det = len(t1_pin_fail_within_det_ids)
    t1_ppv_fail_within_det_set = set(t1_ppv_fail_within_det_ids)
    t1_pin_fail_within_det_set = set(t1_pin_fail_within_det_ids)

    # Compute per-ablation metrics.
    rows: List[Dict[str, object]] = []
    rescue_det_counts: Dict[str, int] = {}
    rescue_ppv_counts: Dict[str, int] = {}
    rescue_pin_counts: Dict[str, int] = {}

    for a in ablations:
        exp_id = str(a.get("id", "")).strip()
        eval_dir = Path(str(a.get("eval_dir", "")).strip()).resolve()
        if not exp_id:
            continue
        if exp_id not in base_meta:
            raise KeyError(f"Missing {exp_id} in base ablation TSV: {args.ablation_base_tsv}")

        sub = _load_subject_rows(eval_dir)
        cl = _load_cluster_rows(eval_dir)
        m = _compute_metrics(scope_subjects, sub, cl)

        resc_det = sum(bool(sub[sid].detected_boxdsc) for sid in t1_det_fail_set)
        resc_ppv = sum(bool(sub[sid].detected_ppv50) for sid in t1_ppv_fail_within_det_set)
        resc_pin = sum(bool(sub[sid].pinpointed) for sid in t1_pin_fail_within_det_set)
        rescue_det_counts[exp_id] = int(resc_det)
        rescue_ppv_counts[exp_id] = int(resc_ppv)
        rescue_pin_counts[exp_id] = int(resc_pin)

        rows.append(
            {
                "Experiment": exp_id,
                "Group": str(base_meta[exp_id].get("Group", "")).strip(),
                "Description": str(base_meta[exp_id].get("Description", "")).strip(),
                "Vertex DSC (mean)": float(m["vertex_dsc_mean"]),
                "Det(boxDSC>0.22) (overall)": float(m["detection_rate_boxdsc"]),
                "Pinpointing (overall)": float(m["pinpointing_rate"]),
                "FP clusters/pt (overall)": float(m["fp_per_patient_mean"]),
                "Cluster precision (overall)": float(m["precision_boxdsc"]),
                "Cluster F1 (overall)": float(m["f1_boxdsc"]),
                f"Rescue Det(boxDSC>0.22) among T1 Det(boxDSC) fails (n={n_fail_det})": _fmt_rescue(int(resc_det), n_fail_det),
                f"Rescue Pinpointing among T1 pinpointing fails within T1 Det(boxDSC) fails (n={n_fail_pin_within_det})": _fmt_rescue(
                    int(resc_pin), n_fail_pin_within_det
                ),
                f"Rescue Det(PPV>=0.5) among T1 Det(PPV) fails within T1 Det(boxDSC) fails (n={n_fail_ppv_within_det})": _fmt_rescue(
                    int(resc_ppv), n_fail_ppv_within_det
                ),
                "_resc_det": int(resc_det),
                "_resc_ppv": int(resc_ppv),
                "_resc_pin": int(resc_pin),
            }
        )

    # Rank rescues.
    rank_det = _rank_average_ties_desc(rescue_det_counts)
    rank_ppv = _rank_average_ties_desc(rescue_ppv_counts)
    rank_pin = _rank_average_ties_desc(rescue_pin_counts)

    for r in rows:
        exp_id = str(r["Experiment"])
        rd = float(rank_det[exp_id])
        rppv = float(rank_ppv[exp_id])
        rp = float(rank_pin[exp_id])
        r["Rescue Det(boxDSC) rank (1=best)"] = rd
        r["Rescue Det(PPV>=0.5) rank (1=best)"] = rppv
        r["Rescue Pinpointing rank (1=best)"] = rp
        r["Mean rescue rank"] = float((rd + rppv + rp) / 3.0)

        # Convert floats to the exact numeric formatting expected by the XLSX template.
        r["Vertex DSC (mean)"] = _fmt_num(float(r["Vertex DSC (mean)"]), decimals=4)
        r["Det(boxDSC>0.22) (overall)"] = _fmt_num(float(r["Det(boxDSC>0.22) (overall)"]), decimals=3)
        r["Pinpointing (overall)"] = _fmt_num(float(r["Pinpointing (overall)"]), decimals=3)
        r["FP clusters/pt (overall)"] = _fmt_num(float(r["FP clusters/pt (overall)"]), decimals=3)
        r["Cluster precision (overall)"] = _fmt_num(float(r["Cluster precision (overall)"]), decimals=3)
        r["Cluster F1 (overall)"] = _fmt_num(float(r["Cluster F1 (overall)"]), decimals=3)
        r["Rescue Det(boxDSC) rank (1=best)"] = _fmt_rank(float(r["Rescue Det(boxDSC) rank (1=best)"]))
        r["Rescue Det(PPV>=0.5) rank (1=best)"] = _fmt_rank(float(r["Rescue Det(PPV>=0.5) rank (1=best)"]))
        r["Rescue Pinpointing rank (1=best)"] = _fmt_rank(float(r["Rescue Pinpointing rank (1=best)"]))
        r["Mean rescue rank"] = _fmt_rank(float(r["Mean rescue rank"]))

    # Sort rows by mean rescue rank (lower is better), then by rescue counts.
    rows = sorted(
        rows,
        key=lambda d: (
            float(d["Mean rescue rank"]),
            -int(d["_resc_det"]),
            -int(d["_resc_ppv"]),
            -int(d["_resc_pin"]),
            str(d["Experiment"]),
        ),
    )

    # Write TSV (drop helper cols).
    out_tsv = (PROJECT_DIR / args.out_tsv).resolve()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order: match the XLSX template.
    rescue_det_col = f"Rescue Det(boxDSC>0.22) among T1 Det(boxDSC) fails (n={n_fail_det})"
    rescue_pin_col = (
        "Rescue Pinpointing among T1 pinpointing fails within "
        f"T1 Det(boxDSC) fails (n={n_fail_pin_within_det})"
    )
    rescue_ppv_col = (
        "Rescue Det(PPV>=0.5) among T1 Det(PPV) fails within "
        f"T1 Det(boxDSC) fails (n={n_fail_ppv_within_det})"
    )
    cols = [
        "Experiment",
        "Group",
        "Description",
        "Vertex DSC (mean)",
        "Det(boxDSC>0.22) (overall)",
        "Pinpointing (overall)",
        "FP clusters/pt (overall)",
        "Cluster precision (overall)",
        "Cluster F1 (overall)",
        rescue_det_col,
        rescue_pin_col,
        "Rescue Det(boxDSC) rank (1=best)",
        "Rescue Pinpointing rank (1=best)",
        "Mean rescue rank",
        rescue_ppv_col,
        "Rescue Det(PPV>=0.5) rank (1=best)",
    ]

    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # Update XLSX (optional).
    #
    # In the private project we used a styled XLSX template to preserve exact formatting.
    # For public release, the template may be omitted to avoid shipping Office metadata.
    template_xlsx = (PROJECT_DIR / args.template_xlsx).resolve()
    out_xlsx = (PROJECT_DIR / args.out_xlsx).resolve()
    if template_xlsx.is_file():
        _update_xlsx(
            template_xlsx=template_xlsx,
            out_xlsx=out_xlsx,
            rows_sorted=rows,
            rescue_n_fail_det=n_fail_det,
            rescue_n_fail_pin_within_det=n_fail_pin_within_det,
            rescue_n_fail_ppv_within_det=n_fail_ppv_within_det,
        )
    else:
        # Fallback: write a simple XLSX without template styling.
        try:
            import pandas as pd  # type: ignore

            out_xlsx.parent.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame([{k: r.get(k, "") for k in cols} for r in rows])
            df_out.to_excel(out_xlsx, index=False)
            print(f"WARNING: template_xlsx missing; wrote a simple XLSX without styling: {out_xlsx}")
        except Exception as e:
            print(f"WARNING: template_xlsx missing and failed to write fallback XLSX ({e}); continuing without XLSX.")

    # Traceability.
    trace = {
        "timestamp": datetime.now().isoformat(),
        "scope_subjects_tsv": str(args.scope_subjects_tsv),
        "n_scope": len(scope_subjects),
        "ablation_trace_json": str(args.ablation_trace_json),
        "ablation_base_tsv": str(args.ablation_base_tsv),
        "table2_audit_json": str(args.table2_audit_json),
        "t1_eval_dir": t1_eval_dir_rel,
        "t1_det_fail_definition": "Det(boxDSC>0.22)=False on the baseline T1 eval_dir, restricted to scope38",
        "n_t1_det_fail_in_scope": n_fail_det,
        "t1_det_fail_ids": t1_det_fail_ids,
        "t1_ppv_fail_within_det_definition": (
            "Det(PPV>=0.5)=False on the baseline T1 eval_dir, restricted to "
            "scope38 ∩ {baseline Det(boxDSC>0.22)=False}"
        ),
        "n_t1_ppv_fail_within_det_in_scope": n_fail_ppv_within_det,
        "t1_ppv_fail_within_det_ids": t1_ppv_fail_within_det_ids,
        "t1_pin_fail_within_det_definition": (
            "Pinpointing=False on the baseline T1 eval_dir, restricted to "
            "scope38 ∩ {baseline Det(boxDSC>0.22)=False}"
        ),
        "n_t1_pin_fail_within_det_in_scope": n_fail_pin_within_det,
        "t1_pin_fail_within_det_ids": t1_pin_fail_within_det_ids,
        "outputs": {"table_tsv": str(args.out_tsv), "table_xlsx": str(args.out_xlsx)},
        "ranking": {
            "method": "average rank for ties; 1=best (largest rescue count)",
            "metrics": [rescue_det_col, rescue_ppv_col, rescue_pin_col],
            "mean_rank": "Mean rescue rank = mean of the three ranks",
        },
    }
    out_trace = (PROJECT_DIR / args.out_trace_json).resolve()
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    out_trace.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    print(f"out_tsv={out_tsv}")
    print(f"out_xlsx={(PROJECT_DIR / args.out_xlsx).resolve()}")
    print(f"out_trace={out_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
