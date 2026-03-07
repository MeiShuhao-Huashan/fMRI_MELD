# README_reproduce_appeal（Editor-first 最小复现入口）

本文件对应 `paper/appeal/EXPERIMENT_CHECKLIST_editor_first.md` 中 E3 的最低交付：
**给编辑/审稿人一个可以直接执行的最小命令链**，用于复现主文与补充表格数值。

---

## 0) 目标与范围

- 目标：复现稿件中核心表格/图的**数值**（尤其 Table 1/2/3 与补充表）
- 不包含：原始影像级端到端训练/推理（该部分见 `docs/end_to_end.md`）
- 数据来源：本仓库已附带去标识的中间评估产物（`meld_data/output/three_level_eval/**`）

---

## 1) 环境准备

在仓库根目录 `release/meld-fmri-epilepsia-repro/` 执行：

```bash
conda env create -f environment/MELD_fMRI_env_export.yml
conda activate MELD_fMRI
```

（Linux 精确锁定可选）

```bash
conda create -n MELD_fMRI --file environment/MELD_fMRI_env_explicit_linux-64.txt
conda activate MELD_fMRI
```

---

## 2) 最小可执行复现链（一步命令）

```bash
bash reproduce_all.sh
```

该命令将顺序完成：
- 私有信息扫描（失败即停止）
- 重建 Table 1 / Table 2 / Table 3
- 重建 Figure 2 / Figure 3 关键 panel
- 重建补充 rs-fMRI ablation 矩阵
- 再次私有信息扫描

---

## 3) 结果位置（编辑重点）

- Table 1  
  `paper/revision/table1/table1_outcome58_baseline.md`  
  `paper/revision/table1/table1_outcome58_baseline.tsv`

- Table 2（主文最终口径）  
  `paper/revision/table2/final/prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md`

- Table 3（scope38）  
  `paper/revision/table3/final/table3_scope38_multiplicity_adjusted_epilepsia.md`  
  `paper/revision/table3/final/table3_source_scope38_final.md`

- Supplementary（fMRI ablation）  
  `paper/supplement_fmri_ablation/tableS_fmri_ablation_matrix_scope38.tsv`

---

## 4) 一致性核验（可选但推荐）

### 4.1 脚本/依赖自检

```bash
pytest -q
```

### 4.2 稿件与补充表一致性核验（你本地有 docx/xlsx 时）

```bash
python scripts/release/check_appeal_final_consistency.py \
  --docx /path/to/appeal_final_manuscript.docx \
  --supp_xlsx /path/to/Supplementary_Tables.xlsx
```

---

## 5) 隐私与发布安全

发布前建议执行：

```bash
python scripts/release/check_no_private_patterns.py --root .
```

并确保以下内容不上传：
- `release/_PRIVATE_DO_NOT_UPLOAD/`
- 原始影像（NIfTI/DICOM/FreeSurfer subjects）
- 任意绝对路径与可逆 ID 映射

---

## 6) 端到端代码路径（非本最小复现必需）

若要跑 fMRI 模型训练/推理与 TrackA 融合，请看：

- `docs/end_to_end.md`
- `scripts/end_to_end/train_deepez_gcn.py`
- `scripts/end_to_end/predict_deepez_gcn.py`
- `scripts/end_to_end/train_deepez_laterality.py`
- `scripts/end_to_end/predict_deepez_laterality.py`
- `scripts/end_to_end/run_trackA_v2_fmrihemi_takeover_three_level_eval.py`

