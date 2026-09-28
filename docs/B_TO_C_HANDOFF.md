# B → C 检测器交接说明

更新日期：2026-09-28

B 的提交：[`feature/detection`](https://github.com/immortal-flower/dip-project1-anime-face/tree/feature/detection)

B 的 Pull Request：[#2 Complete B Cascade experiments and final evaluation](https://github.com/immortal-flower/dip-project1-anime-face/pull/2)

## 1. 交接结论

B 的 Cascade 检测训练、困难负样本实验、扫描参数选择和 108 页 test 评价已经完成。C 可以直接使用冻结检测器输出的原图坐标人脸框，再连接自己的 28 点回归模型。

最终检测器完成了课程要求的算法和实验链路，但真实 test 召回率只有 4.83%。因此 C 应同时报告：

1. 使用真实框（oracle bbox）的关键点 NME，用来单独评价关键点回归器；
2. 使用 B 检测框的端到端 NME 和成功覆盖率，用来评价完整流程。

不能只报告成功检测到的人脸上的 NME 而忽略大量漏检，也不能继续使用 108 页 test 调整参数。

## 2. C 需要读取的文件

### 2.1 冻结检测模型（本地目录与 GitHub 压缩包）

目录：`results/b-final/model/`

为解决 `results/` 被 `.gitignore` 排除、C 无法从 GitHub 获得模型的问题，完整目录已压缩为 `deliverables/b-final-model.zip`。C 克隆仓库后可按 `deliverables/README.md` 恢复到上述路径并校验 SHA256。压缩包不含 Manga109 原页或训练图片。

| 文件 | C 的用途 |
|---|---|
| `detector.json` | 三级 Cascade、弱树、权重、阈值和训练日志 |
| `config.json` | 最终滑窗、金字塔、NMS、批量大小和分数阈值 |
| `feature_definition.json` | `dip11-int-v1` 十一通道定义及指纹 |
| `B_MODEL_MANIFEST.json` | 上述交付文件的大小和 SHA256 |
| `splits.json` | B 训练使用的区域样本与固定划分记录 |
| `landmark.npz` | 现有联合包中的关键点基线；B 未验证其真实 NME，C 不应把它当成最终关键点模型 |

最终检测参数已经写入 `config.json`：

```text
window_size=24×24
step=6
scale_factor=1.3
nms_threshold=0.3
pre_nms_limit=500
scan_batch_size=4096
score_threshold=13.853865092499843
```

### 2.2 C 的关键点数据

目录：`data/processed/joint_a_c_v1/`

- `landmarks_validated.json`：253 条通过格式与可见点边界检查的记录，train/val/test=190/25/38；建议作为当前训练入口。
- `landmarks_imported.json`：C 原始 256 条标注的可移植副本。
- `annotation_issues.json`：3 张图、4 个越界可见点，C 确认前不要自动截断或改 visibility。
- `images/`：上述 256 张图像的本地副本。
- 点序：`hysts28-v1`；每条记录为 28×2 原图坐标并带 28 个 visibility 标志。

详细数据检查见 `docs/C_DATA_INTEGRATION.md` 和 `docs/C_ANNOTATION_FEEDBACK.md`。

### 2.3 B 的结果证据

- `docs/B_FINAL_REPORT.md`：完整实验报告和限制。
- `docs/B_RETRAIN_COMPARISON.md`：困难负样本增强前后的逐 Stage 对比。
- `docs/b_results/B_FINAL_SUMMARY.json`：可上传 GitHub 的机器可读汇总。
- `results/b-final/test-full-108pages.json`：108 页预测、逐页日志和检测指标。
- `results/b-final/examples/examples-sheet.png`：6 页典型检测结果。

## 3. 推荐的接入方式

### 3.1 先单独取得 B 的检测框

这条路径不会加载 `landmark.npz`，适合 C 调试自己的回归器：

```python
import json
from pathlib import Path

import cv2

from src.sliding_window import scan_image

model_dir = Path("results/b-final/model")
config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
detector = json.loads((model_dir / "detector.json").read_text(encoding="utf-8"))

bgr = cv2.imread("path/to/page.jpg", cv2.IMREAD_COLOR)
predictions, scan_logs = scan_image(bgr, detector, config)

# predictions 已按 score 降序并完成 NMS 和最终分数过滤。
# 每项格式：{"bbox": [x1, y1, x2, y2], "score": float}
```

坐标为原图中的 `xyxy`，右下边界不包含。`score` 是各 Stage AdaBoost 裕量之和，不是概率。无检测时返回空列表。

### 3.2 连接 C 的关键点模型

C 训练并验证关键点模型后，应在新目录中组装最终包，不要覆盖 `results/b-final/model/`，以便保留 B 的冻结证据。例如创建 `results/c-final/combined-model/`，复制 B 的检测文件，再放入 C 的最终 `landmark.npz`。

组合目录至少需要：

```text
combined-model/
├── config.json
├── detector.json
├── feature_definition.json
└── landmark.npz       # C 训练并验证后的最终模型
```

之后可以使用统一接口：

```python
import cv2
from src.detector import AnimeFaceDetector

detector = AnimeFaceDetector("results/c-final/combined-model")
bgr = cv2.imread("path/to/page.jpg", cv2.IMREAD_COLOR)
results = detector.detect(bgr)

# 每项格式：
# {"bbox": [x1, y1, x2, y2], "score": float, "landmarks": [[x, y], ...]}
# landmarks 为原图坐标的 28×2 数组；无脸时 results=[]。
```

如果 C 修改 `config.json`，应保留 B 的检测参数和特征指纹，只新增明确命名的关键点模型元数据，不要暗中改变 `step`、`scale_factor` 或 `score_threshold`。

## 4. C 的评价协议建议

### 4.1 关键点模型单独评价

- 只用 train 拟合，val 选择级数、采样偏移或正则化参数，test 只运行一次最终评价。
- 使用 `landmarks_validated.json` 的固定 split，不重新随机划分。
- 调用 `src.landmark_metrics.nme` 时显式传入正的归一化距离。
- 只对 visibility=1 的点计算误差；全不可见样本应排除并记录数量。
- 报告整体 NME、按图分布、失败样例，以及各点可见样本数量。

### 4.2 端到端评价

- 先按 IoU≥0.5 将 B 检测框和真值框一对一匹配，再对匹配人脸计算关键点 NME。
- 同时报告检测 TP/FP/FN、检测召回率、成功产生 28 点的人脸数、匹配人脸 NME。
- 漏检人脸没有关键点结果，必须作为端到端覆盖率损失单列，不能从 NME 分母中悄悄消失。
- 建议把 oracle bbox NME 与 detected bbox NME 并排报告，以区分回归误差和检测框误差。

## 5. B 最终结果与已知限制

| 集合 | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| validation，统一阈值后 | 33 / 702 / 546 | 0.0449 | 0.0570 | 0.0502 |
| test，108 页 | 69 / 960 / 1360 | 0.0671 | 0.0483 | 0.0561 |

- test 共扫描 13,514,148 个窗口，108/108 页完成，遗漏页为 0。
- 第 0 Stage 拒绝约 68.9% 窗口；后两级条件通过率偏高，是误报仍多的主要原因之一。
- 两轮困难负样本共复核 240 个候选、接受 215 个，但增强模型没有超过同容量非增强模型，因此最终冻结非增强 1024 候选模型。
- 典型误报来自对白文字、头发线条、衣物纹理、速度线和局部脸部结构。
- 检测框质量会显著限制端到端关键点效果。C 不应为了得到更好 NME 而只选容易页或在 test 上提高分数阈值。

## 6. C 接手后的完成清单

- [ ] 阅读 `docs/INTERFACES.md` 的“C 回归与最终接口”。
- [ ] 确认 `annotation_issues.json` 中 3 张边界例的处理决定，或继续使用当前 253 条 validated 集。
- [ ] 在固定 train/val/test 上训练并选择最终关键点模型。
- [ ] 输出真实的 oracle bbox validation/test NME 和失败样例。
- [ ] 导出无 pickle 的最终 `landmark.npz`。
- [ ] 在新目录组装 B 检测器与 C 关键点模型，不覆盖 B 冻结目录。
- [ ] 验证 `AnimeFaceDetector.detect` 的空结果、坐标范围、排序及保存后重载一致性。
- [ ] 报告 detected bbox 端到端 NME、覆盖率以及检测 TP/FP/FN。
- [ ] 更新 `docs/STATUS.md`、README 和团队最终报告，明确区分合成接口测试与真实数据效果。

## 7. 出现问题时优先核查

1. 图像必须是非空 `uint8`、OpenCV BGR、形状 H×W×3。
2. bbox 必须是原图坐标 `[x1,y1,x2,y2]`，且 `x2/y2` 不包含在框内。
3. C 的点坐标是 `(x,y)`，数组访问才使用 `[y,x]`。
4. 使用的 `feature_definition.json` SHA256 必须与 `config.json` 一致。
5. 不要把 `score` 当作概率，也不要在 test 上重新选择阈值。
6. 若统一接口能运行但 NME 异常，先分别用真实框和检测框测试 `predict_shape`，确认问题来自回归器还是检测框偏移。
