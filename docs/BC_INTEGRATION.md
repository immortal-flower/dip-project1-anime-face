# B检测与C关键点联合运行

## 当前状态

- B冻结模型已随仓库保存为 `deliverables/b-final-model.zip`，模型包SHA256为 `8000810243b1b7be69a27d6ca201a803ea87416becf6d2428f38e4885babefb1`。
- C最终模型为HOG与形状回归各50%融合的 `landmark-ensemble-human-reviewed`，人工test平均NME为0.0864。
- 2026-09-28已在C固定的38张人工复核test图上完成真实B+C联合评价；结果见 [BC_END_TO_END_REPORT.md](BC_END_TO_END_REPORT.md)。
- 联合评价使用AnimeFace裁剪脸图，每张图只有一个真值框。这与B的Manga109整页108页test不是同一数据分布，两组检测指标不能直接互换。

## 恢复B冻结模型

```powershell
Expand-Archive deliverables/b-final-model.zip results/b-final -Force
```

恢复后模型目录为：

```text
results/b-final/model/
├── detector.json
├── config.json
├── feature_definition.json
├── B_MODEL_MANIFEST.json
├── splits.json
└── landmark.npz
```

其中B包内的 `landmark.npz` 只是兼容占位，联合运行时必须用C最终关键点模型覆盖它的功能。

## C模型

```text
models/landmark-ensemble-human-reviewed/
├── landmark.npz
├── config.json
└── metrics.json
```

## 单图端到端演示

```powershell
python demo.py --image data/example.jpg `
  --model-dir results/b-final/model `
  --landmark-model-dir models/landmark-ensemble-human-reviewed `
  --output results/end-to-end/example.jpg
```

输出图片绘制B检测框和C的28点；同名JSON保存分数、坐标和逐Stage扫描日志。

## 联合评价

```powershell
python -m scripts.evaluate_end_to_end_landmarks `
  --manifest data/landmarks/corrected/manifest.json `
  --detector-model results/b-final/model `
  --landmark-model models/landmark-ensemble-human-reviewed `
  --output results/end-to-end/test-b-final-c-ensemble `
  --split test --iou-threshold 0.5
```

评价分两层：

1. 检测层报告TP、FP、FN、Precision、Recall和F1。
2. 只有IoU达到0.5的一对一匹配框才计算关键点NME和PCK；漏检始终计入FN。

本次38图结果为TP/FP/FN=3/7/35，Recall=7.89%；3个匹配框的平均NME为0.1281，完整端到端成功率（检出且NME≤0.10）为5.26%。这说明当前联合系统的主要瓶颈是检测漏检，而不是接口没有接通。
