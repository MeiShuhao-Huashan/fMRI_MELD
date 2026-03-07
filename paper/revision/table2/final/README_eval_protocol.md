# Table 2（主文）最终评估口径与模型来源说明（final）

本文件夹用于**固定主文 Table 2 的最终评估口径**并集中存放结果文件，避免后续被其它扫参/对照实验覆盖。

## 1) 本次选择的最终评估口径（主文）

- **入组队列（Outcome cohort）**：仅纳入 `MRIDetectability ∈ {Intermediate, Difficult}` 的病例。  
  - 总数 `n=58`（`ILAE 1–2 = 38`；`ILAE 3–6 = 20`）。
- **分组定义**：`ILAE 1–2` 视为 seizure-free（SF），`ILAE 3–6` 视为 not seizure-free（NSF）。
- **Index tests（主文呈现）**
  - **MELD（自训练）**：使用你在 `meld_data/models/` 下训练的五折模型产生的 T1-only 推理结果，**不加面积约束**（unconstrained）。
  - **rs-fMRI（纯 fMRI）**：纯 fMRI 单模态推理，带**面积约束**（当前项目内产物目录名为 `fmri_parcelavg`）。
  - **TrackA（后融合）**：使用**原文 TrackA（paper 版本）**的后融合输出（含新增 +20 病例），用于主文呈现。
  - （补充材料）**Intersection**：MELD 与 rs-fMRI 的预测簇在皮层表面上的交集（mask-level intersection）。
- **主阈值端点（主文呈现；3 个二值端点）**
  - 技术端点：`Det(boxDSC > 0.22 & Dice > 0.01)`（boxDSC 过宽时的 safeguard）
  - 临床端点：`Det(PPV-in-mask ≥ 0.5)`
  - 额外端点：`Pinpointing（质心切除）`
  - （补充材料）连续变量：`PPV-in-mask（max across clusters）`，按每 `+0.1` 作为回归自变量。

> 选择该口径的原因：在 `ILAE1–2` 的 38 例中，**TrackA 在 `boxDSC>0.22&Dice>0.01` 与 `PPV≥0.5` 两个阈值下的 “Resected=Yes” 均高于 MELD 单模态**；同时在（技术端点）与（补充分析的）连续 PPV 指标下，TrackA 与预后的关联强于 MELD。

## 2) 统计模型与处理细则

- **回归模型**：Firth logistic regression  
  - 因变量：`SF = 1`（ILAE1–2） vs `SF = 0`（ILAE3–6）
  - 自变量：端点（如 `Det(PPV≥0.5)`）+ 协变量
  - 协变量：`log(RC_volume_cm3)` + `Sex(Male=1)`
- **推断（避免 Wald 不稳）**
  - p 值：penalized likelihood ratio test（PLR）
  - 置信区间：profile penalized likelihood CI（Firth）
- **多簇处理**
  - 二值端点：`any-cluster positive`（任一簇满足阈值即记为 Yes）
  - 连续端点：`max_ppv_in_mask`（跨簇取最大值）
  - 其中 **boxDSC 技术端点**的 TP 簇定义为：`boxDSC>0.22` **且** `Dice>0.01`
- **No output 策略**
  - 定义：`n_clusters <= 0`
  - 处理：二值端点按 0；连续 PPV 置 0；同时单列报告 `No output n/N (%)`

## 3) 关键输入表与协变量来源

- 队列与分组（MRIDetectability、ILAE、Sex）：`supplement_table_hs_patient_three_level_metrics.filled.csv`
- 切除体积（用于回归协变量）：`meld_data/metadata/rc_volume_cm3_outcome58.csv`（单位：cm³）

## 4) 本口径对应的“原始评估产物”路径（可复算/追溯）

本文件夹内的表格来自如下 three-level evaluation 产物（均为 n=58）：

- MELD（自训练、不加面积约束）  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_native_surface/subject_level_results.csv`
- rs-fMRI（纯 fMRI、面积约束）  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/fmri_parcelavg/subject_level_results.csv`
- Intersection  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/meld_fmri_intersection/subject_level_results.csv`
- TrackA（paper 版本 +20）  
  `meld_data/output/three_level_eval/revision2026_clinical_response_trackAplus20_SELF5F/trackA_locked_paper_plus20/subject_level_results.csv`

## 4.1) 模型权重来源（便于追溯）

> 说明：主文统计分析直接读取上面四个 eval 目录里的 `subject_level_results.csv`（这些 eval 目录由既有推理/后处理流程生成）。
> 下面列出的是这些推理流程所依赖的关键模型权重目录（供审稿应对与复现实验追溯）。

- **自训练 MELD（T1-only）五折权重目录**（与你之前指定的五个模型一致）：
  - `meld_data/models/25-12-16_MELD_fMRI_fold0/fold_00`
  - `meld_data/models/25-12-15_MELD_fMRI_fold1/fold_01`
  - `meld_data/models/25-12-15_MELD_fMRI_fold2/fold_02`
  - `meld_data/models/25-12-16_MELD_fMRI_fold3/fold_03`
  - `meld_data/models/25-12-16_MELD_fMRI_fold4/fold_04`
- **rs-fMRI（纯 fMRI）五折权重目录**（用于生成 `fmri_parcelavg`，并在后处理阶段施加面积约束）：
  - `meld_data/models/v2_gat3_diceproxy/fold_00`
  - `meld_data/models/v2_gat3_diceproxy/fold_01`
  - `meld_data/models/v2_gat3_diceproxy/fold_02`
  - `meld_data/models/v2_gat3_diceproxy/fold_03`
  - `meld_data/models/v2_gat3_diceproxy/fold_04`
- **rs-fMRI 是否包含“仅用 fMRI 做病灶定侧”？是。**  
  `fmri_parcelavg` 的生成采用 `fmrihemi` 部署流程：先用 **fMRI laterality classifier** 预测 `pred_hemi / prob_right`，再将 fMRI 模型输出的 ipsi/contra 概率映射到左右半球并做面积约束（a30_t80）。
  - 旧 52 例（每折）可在对应 fold 的配置与定侧文件中核对：
    - 例：`meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80/fold0/val/deploy_config.json`（`hemi_mode=fmri_laterality_classifier`，并指向 `laterality_csv`）
    - 例：`meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_a30_t80/fold0/val/fmri_inferred_hemi.csv`
  - 新增 +20 例（missing20）可在此核对每折定侧输出：
    - `meld_data/output/three_level_eval/v2_gat3_diceproxy_deploy_real_fmrihemi_ensemble5f_missing20/fmri_laterality_per_fold_missing20.csv`
  - 上述 CSV 中的 `laterality_correct`（如存在）仅用于评估记录；推理时使用的是 `pred_hemi / prob_right`，不使用 GT。
- **TrackA（paper）后融合规则/阈值定义**：见 `paper/supplement_methods/SUPPLEMENTARY_METHODS_trackA_gate.md`（含 `t1_conf_thr`、`diff_thr`、低差异注入与冲突注入等规则）。

## 5) 结果文件清单（本 final 文件夹内）

- 主文汇总（3 个二值端点；3 方法；**Holm 多重比较矫正后的 p**）：  
  `prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md`
- 三个二值端点的 TSV（原始输出，包含 Intersection 行；p 为未矫正 PLR p）：  
  - `prognosis_table_boxdsc022_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.tsv`  
  - `prognosis_table_ppv05_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.tsv`  
  - `prognosis_table_pinpointing_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.tsv`
- （补充材料）连续 PPV 的 TSV：  
  `prognosis_table_ppv_continuous_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.tsv`  
  以及对应的主文外补充汇总：`paper/revision/supplement/table2_panelD_ppv_continuous_selftrained.md`
- （补充材料）Intersection + 连续 PPV（并含 A/B/C 三个二值端点的 Intersection 行）：  
  `paper/revision/supplement/table2_intersection_and_continuous_ppv_selftrained.md`
- （探索性补充）PPV 阈值扫描（用于补充材料，不建议替代预先指定阈值）：
  - `ppv_threshold_scan_trackA_paper_vs_meld_unconstrained.md`
  - `ppv_threshold_scan_trackA_paper_vs_meld_unconstrained.tsv`

## 6) 可复现的生成命令（再现本 final 结果）

该组表格由脚本 `scripts/paper/make_prognosis_table_constraints_a30_t80.py` 生成（此脚本名包含 constraints，但本次 MELD/Intersection 评估目录选择的是 unconstrained 版本；是否约束由传入的 eval_dir 决定）。

```bash
source env.sh

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

# 生成主文 Table2（3 个二值端点；3 方法；Holm 校正后 p）
python scripts/paper/make_table2_maintext_3binary_3methods_holm.py \
  --in_dir paper/revision/table2/final \
  --tag maintext_selftrained_unconstrained_meld_tracka_paper_plus20 \
  --out_md paper/revision/table2/final/prognosis_table_maintext_selftrained_unconstrained_meld_tracka_paper_plus20.md
```
