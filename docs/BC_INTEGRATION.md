# B检测与C关键点联合运行

## 当前状态

- B代码、实验报告和机器可读汇总已经在 `feature/detection` 完整提交。
- C训练、人工测试、HOG融合和Fern/LBF已经在 `feature/landmarks-demo` 完整提交。
- `integration/b-c` 从B最终分支建立，并合入C三次提交；39项测试和合成端到端流程通过。
- GitHub按项目规则忽略 `models/` 和 `results/`。B文档所述冻结模型 `results/b-final/model/` 不在仓库，本机也未找到，因此当前只能验证接口，不能生成真实联合指标。

## B同学需要共享的冻结模型包

直接复制B机器上的整个 `results/b-final/model/`，至少应包含：

```text
detector.json
config.json
feature_definition.json
B_MODEL_MANIFEST.json
```

收到后放到本机同名目录。不要用GitHub里的 `B_FINAL_SUMMARY.json` 代替模型；汇总文件只有指标，没有弱树参数。

## C模型

C最终模型目录为：

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
  --output results/end-to-end/test `
  --split test --iou-threshold 0.5
```

指标分两层：

1. 检测层报告TP、FP、FN、Precision、Recall和F1。
2. 只有IoU达到0.5的匹配框才计算关键点NME；漏检始终计入FN，不能从总结果中删除。

B的108页最终test Recall为0.0483，当前检测器质量较低。即使C在人工框上的NME为0.0864，真实端到端结果仍可能因大量漏检和框偏移明显下降；报告必须分别展示人工框结果和检测框结果。
