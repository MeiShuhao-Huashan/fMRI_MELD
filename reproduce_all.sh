#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Workaround for some HPC images where MKL is present but a compatible OpenMP runtime
# is not on LD_LIBRARY_PATH, causing Python to crash on import (e.g., NumPy BLAS).
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libgomp.so.1" ]; then
  _libgomp="${CONDA_PREFIX}/lib/libgomp.so.1"
  case ":${LD_PRELOAD:-}:" in
    *":${_libgomp}:"*) : ;;
    *) export LD_PRELOAD="${_libgomp}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
  unset _libgomp
fi

mkdir -p outputs

python scripts/release/check_no_private_patterns.py --root "${ROOT}"

echo "[1/6] Table 1 (baseline; outcome-linked n=58)"
python scripts/paper/make_table1_outcome58_baseline.py \
  --meta_csv supplement_table_hs_patient_three_level_metrics.filled.csv \
  --rc_csv meld_data/metadata/rc_volume_cm3_outcome58.csv \
  --scope_csv meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface/subject_level_results.csv \
  --out_md paper/revision/table1/table1_outcome58_baseline.md \
  --out_tsv paper/revision/table1/table1_outcome58_baseline.tsv \
  --out_raw_csv outputs/table1_outcome58_baseline_raw.csv

echo "[2/6] Table 2 (prognosis; final protocol)"
python scripts/paper/make_prognosis_table_constraints_a30_t80.py \
  --meta_csv supplement_table_hs_patient_three_level_metrics.filled.csv \
  --rc_csv meld_data/metadata/rc_volume_cm3_outcome58.csv \
  --meld_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface \
  --fmri_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg \
  --inter_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_fmri_intersection \
  --tracka_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_paper_plus20 \
  --boxdsc_min_cluster_dice 0.01 \
  --meld_label "MELD (self-trained)" \
  --fmri_label "rs-fMRI" \
  --intersection_label "Intersection" \
  --tracka_label "TrackA (fusion)" \
  --out_dir paper/revision/table2/final \
  --out_md paper/revision/table2/final/prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md \
  --out_tag maintext_selftrained_unconstrained_meld_tracka_paper_plus20

python scripts/paper/make_table2_maintext_3binary_3methods_holm.py \
  --in_dir paper/revision/table2/final \
  --tag maintext_selftrained_unconstrained_meld_tracka_paper_plus20 \
  --out_md paper/revision/table2/final/prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md

echo "[3/6] Table 3 (scope38; final protocol)"
python scripts/paper/make_table3_scope38_multiscale_triplet.py \
  --meta_csv supplement_table_hs_patient_three_level_metrics.filled.csv \
  --paper_table2_csv paper/table2/table2_source_data.csv \
  --t1_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface \
  --fmri_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg \
  --tracka_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_paper_plus20 \
  --boxdsc_min_cluster_dice 0.01 \
  --p_adjust holm3_fdr_bh \
  --out_md paper/revision/table3/final/table3_source_scope38_final.md \
  --out_subjects_tsv paper/revision/table3/final/table3_scope38_subject_ids.tsv \
  --out_audit_json paper/revision/table3/final/table3_scope38_audit.json

echo "[4/6] Figure 2 (scope38 panels)"
python scripts/paper/make_revision_figure2_scope38_panels.py

echo "[5/6] Figure 3 (scope38 panels)"
python scripts/paper/make_revision_figure3_scope38_panels.py

echo "[6/6] Supplementary: fMRI ablation matrix (scope38)"
python paper/supplement_fmri_ablation/make_fmri_ablation_matrix_scope38.py

python scripts/release/check_no_private_patterns.py --root "${ROOT}"

echo "DONE"
