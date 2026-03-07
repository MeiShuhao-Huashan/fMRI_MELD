# PPV 阈值扫描：TrackA vs MELD（Firth logistic, adjusted）

- 口径：自训练MELD不做约束；fMRI面积约束；TrackA=原文TrackA（paper locked +20）
- 队列：Intermediate+Difficult（n=58；ILAE12=38；ILAE3456=20）。
- 回归：Firth logistic；协变量=log(RC_volume_cm3)+Sex(Male=1)。

## 最优阈值（按 ΔLR=LR_trackA−LR_MELD 最大）

- PPV 阈值：0.100
- MELD：aOR=2.25 95%CI=[0.736,7.15] p=0.0793 (ILAE12=25/38, ILAE3456=9/20)
- TrackA：aOR=15.5 95%CI=[4.11,81.3] p=9.1e-06 (ILAE12=29/38, ILAE3456=3/20)
- ΔlogOR(TrackA−MELD)=1.93；ΔLR=16.6

备注：阈值扫描属于探索性分析（存在多重比较/选择偏倚风险），建议主文仍预先指定阈值（如0.5），扫描结果可放补充材料。
