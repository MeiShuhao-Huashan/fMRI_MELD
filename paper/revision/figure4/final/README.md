# Figure 4（2×2 版）— Panel A/C 替换（Easy 剔除后）

## 目标
参考 `paper/figure4/2026-02-10_213048_065.png` 的 2×2 布局，在 **剔除 Easy** 后替换：
- Panel A：No-hurt case
- Panel C：Rescue case

## 新病例
- Panel A（No hurt）：caseA（fold4, lesion=RH）
- Panel C（Rescue）：caseC（fold4, lesion=LH）

## 生成逻辑
- MRI 正交切面图：复用 `paper/figure4/fmrihemi_t1_fail_rescue_native_debug/generate_fmrihemi_t1_fail_rescue_native_debug.py`
  - 运行参数：`--include_t1 --include_tracka`
  - 说明：第一行使用 fMRIhemi (blue) + GT (red) + multi-overlap (yellow) 并叠加 T1/TrackA 轮廓；第二行是 T1-only；第三行是 anatomy-only。
- Surface：从 `paper/figure4/cases/<subject_id>/case_panel.png` 的顶部截取 “Combined (T1/fMRI/TrackA/GT)” 的 lateral/medial 两图（带标题），缩放放入对应 panel 顶部。

## 输出文件
- 2×2 总图（仅替换 A/C，保留原 B/D）：
  - `paper/revision/figure4/final/figure4_2x2_replace_AC.png`
- Panel A 单独导出：
  - `paper/revision/figure4/final/panelA_nohurt_caseA.png`
- Panel C 单独导出：
  - `paper/revision/figure4/final/panelC_rescue_caseC.png`

## 说明
- 本公开包仅保留最终 Figure4 结果图，不包含用于手工排版/截图的中间图像资产。
