# B 提交文件与本地参考目录

## 1. 应上传 GitHub

以下文件不包含 Manga109 原图或大型训练产物，可随 `feature/detection` 分支提交：

- 核心实现：`src/adaboost.py`、`src/cascade.py`、`src/sliding_window.py`
- 训练与实验入口：`scripts/train_baseline.py`、`mine_hard_negatives.py`、`review_hard_negatives.py`、`compare_cascade_models.py`
- 最终评价与交付入口：`scripts/evaluate_page_detector.py`、`select_detection_threshold.py`、`summarize_b_results.py`、`render_detection_examples.py`、`export_b_model.py`
- 测试：`tests/test_contracts.py`、`scripts/smoke_test.py`
- 文档：`README.md`、`docs/INTERFACES.md`、`docs/STATUS.md`、`docs/B_WORK_LOG.md`、`docs/B_RETRAIN_COMPARISON.md`、`docs/B_FINAL_REPORT.md`、本文件
- 小型机器可读证据：`docs/b_results/B_FINAL_SUMMARY.json`

不应把 `data/`、`models/`、`results/` 或 Manga109 原页加入公开 GitHub；这些路径已由 `.gitignore` 排除。

A 的本地交接文件 `START_HERE_B_DATA.md`、`SHA256SUMS_B_DATA.json`、`交接说明.md` 明确标注为团队数据包配套材料，不随公开 GitHub 提交；B 的核验结论已写入 `B_WORK_LOG.md`。

## 2. 本地最终参考目录

根目录：`results/b-final/`

```text
results/b-final/
├── model/                              # 冻结模型与 SHA256 清单
├── examples/                           # 6 张逐页图及 examples-sheet.png
├── test-full-108pages.json             # 108 页全部预测、日志和指标
├── validation-full-step6-scale1.3.json # 60 页阈值选择输入
├── threshold-full-validation.json      # 阈值扫描与选择结果
├── validation-step12-scale1.5.json     # 2×2 消融
├── validation-step6-scale1.5.json
├── validation-step12-scale1.3.json
└── validation-step6-scale1.3.json
```

困难负样本参考目录：

```text
results/hard-negatives/round-1-preview/
results/hard-negatives/round-2-diverse/
```

其中 `review/` 保存显式决定、接受/剔除清单和上下文复核页；`retrain-comparison*.json` 保存各权重的逐 Stage 对比。

原始训练模型目录仍保留在 `models/`。最终可运行副本是 `results/b-final/model/`，其 `config.json` 已写入 step=6、scale=1.3、NMS=0.3、pre-NMS=500、batch=4096 和 score threshold=13.8538650925；`B_MODEL_MANIFEST.json` 保存文件大小与 SHA256。

## 3. 最终结果速查

- 60 页 validation 阈值后：TP/FP/FN=33/702/546，P/R/F1=0.0449/0.0570/0.0502。
- 108 页 test：TP/FP/FN=69/960/1360，P/R/F1=0.0671/0.0483/0.0561。
- test 窗口数 13,514,148，耗时 1774.11 秒，平均 16.43 秒/页，遗漏页 0。
- 两轮困难负样本复核 240/240，接受 215；增强模型未超过同容量非增强模型。

## 4. 复现顺序

1. 运行 `python -m unittest discover -s tests -v` 和 `python -m scripts.smoke_test`。
2. 按 `B_WORK_LOG.md` 训练或直接使用本地冻结模型。
3. 用 `scripts.evaluate_page_detector` 在 val 运行无分数阈值预测。
4. 用 `scripts.select_detection_threshold` 只在 val 选择阈值。
5. 用冻结配置一次性评价 test；不要使用 test 再调参。
6. 用 `scripts.summarize_b_results` 生成 GitHub 安全汇总，用 `render_detection_examples` 生成本地效果图。
