# Table 3（工程主表）最终评估口径与模型来源说明（final；n=38）

本文件夹用于固定 **Table 3（multi-scale performance）** 的最终口径，并集中存放 n=38（ILAE1–2 且 Intermediate+Difficult）子队列的结果文件。

## 1) 队列定义（n=38）

- 起始集合：原文 seizure-free 52 例（见 `paper/table2/table2_source_data.csv`）。
- 仅保留：`MRIDetectability ∈ {Intermediate, Difficult}` 且 `Prognosis(ILAE) ∈ {1,2}`。
- 最终：`n=38`（严格由脚本根据 `supplement_table_hs_patient_three_level_metrics.filled.csv` 自动筛选）。
- **Reference standard**：resection cavity mask（用于 boxDSC / PPV-in-mask 等三层评估）。

## 2) 方法口径（与你在 Table2/final 选定的主文口径一致）

- **T1-only**：`MELD (self-trained; unconstrained)`（自训练 MELD；不加面积约束）
- **fMRI-only**：`rs-fMRI (pure fMRI; laterality+area constrained)`（纯 fMRI；先做 fMRI 定侧，再施加 a30_t80 面积约束）
- **Fusion**：`TrackA (paper fusion; +20)`（原文 TrackA 融合输出；包含 +20 病例，但本表只取其中的 38 例子队列）

Δ 与 p 值均为 **TrackA vs MELD（自训练、不加约束）** 的配对比较。

> 本表（n=38）的技术主端点已采用 safeguard：`Det(boxDSC>0.22 & Dice>0.01)`（避免仅靠 boxDSC 的包围盒重叠导致“虚高命中”）。

## 3) 对应的 three-level evaluation 输入目录（可追溯）

- MELD（自训练、不加约束）：  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface`
- rs-fMRI（纯 fMRI，定侧+面积约束，fmri_parcelavg）：  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg`
- TrackA（paper 融合 +20）：  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_paper_plus20`

## 4) 模型权重来源（便于审稿追溯）

- 自训练 MELD 五折权重（T1-only）：  
  `meld_data/models/25-12-16_MELD_fMRI_fold0/fold_00`  
  `meld_data/models/25-12-15_MELD_fMRI_fold1/fold_01`  
  `meld_data/models/25-12-15_MELD_fMRI_fold2/fold_02`  
  `meld_data/models/25-12-16_MELD_fMRI_fold3/fold_03`  
  `meld_data/models/25-12-16_MELD_fMRI_fold4/fold_04`
- rs-fMRI 五折权重（纯 fMRI；fmrihemi laterality + area constraint）：  
  `meld_data/models/v2_gat3_diceproxy/fold_00` … `fold_04`
- TrackA（paper）融合规则：`paper/supplement_methods/SUPPLEMENTARY_METHODS_trackA_gate.md`

## 5) 本文件夹内输出文件

- 主表（按 `paper/revision/table3/table3_source_original.md` 的格式）：  
  `paper/revision/table3/final/table3_source_scope38_final.md`
- 38 例受试者名单：  
  `paper/revision/table3/final/table3_scope38_subject_ids.tsv`
- 复现审计信息（输入目录/参数/label 等）：  
  `paper/revision/table3/final/table3_scope38_audit.json`

## 6) 复现命令

```bash
python scripts/paper/make_table3_scope38_multiscale_triplet.py \
  --t1_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface \
  --fmri_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg \
  --tracka_eval_dir meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_paper_plus20 \
  --boxdsc_min_cluster_dice 0.01 \
  --p_adjust holm3_fdr_bh \
  --out_md paper/revision/table3/final/table3_source_scope38_final.md \
  --out_subjects_tsv paper/revision/table3/final/table3_scope38_subject_ids.tsv \
  --out_audit_json paper/revision/table3/final/table3_scope38_audit.json
```
