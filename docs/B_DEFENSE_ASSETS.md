# B 模块答辩素材包

## 冻结身份

- Branch：`feature/detection`
- HEAD：`7ba5249fdca5eaa8eff48ff077e0e427c6632812`
- 输出：`results/defense-assets/`
- 生成命令：`python -m scripts.build_b_defense_assets`

本素材包只读取冻结 JSON、review manifest、模型配置和已有报告；未重训、未修改阈值/NMS/校准参数、未重新运行 108 页 test，也未使用 test 做选择。

## 核心结果

| Model | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Original frozen | 69 / 960 / 1360 | 6.71% | 4.83% | 5.61% |
| Enhanced v2 | 247 / 2647 / 1182 | 8.53% | 17.28% | 11.43% |

协议：Manga109、冻结 108-page test、IoU≥0.5、按 score 的自动一对一匹配。V2 的 Recall 与 F1 较高，但 FP 也从 960 增至 2647，不能描述为无代价或全面提升。

原 3-stage 模型的 Stage 0 evaluated/passed/rejected 为 13,514,148/4,202,378/9,311,770，即早拒绝 68.90%。Stage timing 仅为 scoring，不包含 feature extraction。

## HNM 边界

- Round 1：mined/accepted/rejected=120/109/11，扫描 6 页。
- Round 2：mined/accepted/rejected=120/106/14，扫描 12 页。
- 前两轮 accepted=215；V2 新 train accepted=287；累计 train accepted=502。
- V2 validation accepted=88，只用于选择/校准；test hard negatives=0。
- 验证负结果也被保留：512 baseline/HN FPR=41.11%/47.50%；1024 baseline/HN weight-3=33.33%/36.94%。

## 图例与案例

六个 accepted HN 类型：dialogue text、high-contrast hair、clothing texture、dense manga lines、eye-like local structure、building/object outline。源页、坐标、score、review decision 和解释均在 `results/defense-assets/03_hard_negatives/metadata.json`。

检测案例来自冻结 V2 逐页 artifact：Success=`UltraEleven:036`、`DollGun:025`；FP=`YoumaKourin:099`、`Joouari:001`；FN=`MoeruOnisan_vol01:068`、`TetsuSan:028`。选择规则及逐页 TP/FP/FN 在 `04_detection_examples/metadata.json`。

## 追溯入口

- 汇总机器可读数据：`results/defense-assets/metadata/DEFENSE_METRICS.json`
- 每张图的 PPT 用途、来源与注意事项：`results/defense-assets/PPT_ASSET_INDEX.md`
- 原检测结果：`results/b-final/test-full-108pages.json`、`results/optimization-v2/manga109-test-final-108.json`
- 正式旧消融：`docs/b_results/B_FINAL_SUMMARY.json`
- HNM 报告数字：`docs/B_RETRAIN_COMPARISON.md`（原始 JSON 不方便统一抽取的条目标注为 `source_type=document`）

## 不应混用

本素材包没有生成 B+C 主图。`integration/b-c` 的 old B + C final ensemble 与本分支 local-face-box 协议不同；B-side NME normalization 也不等于 C 正式 interocular NME，不能放在同一比较图中。
