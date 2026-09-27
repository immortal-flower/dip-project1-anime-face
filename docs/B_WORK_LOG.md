# B 工作记录：Cascade 与困难负样本挖掘

## 已完成

- `src/adaboost.py`：保存每轮加权错误、alpha 和停止原因。
- `src/cascade.py`：按验证集目标召回率校准阈值，在弱树前缀中控制每级误报率；不复用已淘汰样本；保存逐 Stage 的训练/验证误拒、误放、阈值轨迹和耗时。
- `src/sliding_window.py`：保存每个金字塔层、每个 Stage 的 evaluated、passed、rejected 和耗时。
- `scripts/mine_hard_negatives.py`：扫描训练整页，排除与真值脸框重叠的预测，导出 24×24 困难负样本、增强清单、画廊和逐页日志；支持轮次字段和跨轮去重。
- `scripts/train_baseline.py`：增加 Stage 数、弱树上限、目标召回率、目标误报率、候选数和随机种子参数；保留 A 定义的特征配置与 SHA256 指纹。

## A 交接包核验

- `SHA256SUMS_B_DATA.json`：6731/6731 文件大小与 SHA256 匹配。
- `manifest_candidate.json`：6407 条（train 4770、val 587、test 1050；正 2522、负 3885），全部图片可解码。
- `joint_manifest_candidate.json`：6660 条（train 4960、val 612、test 1088；正 2775、负 3885），全部图片可解码。
- `landmarks_validated.json`：253 条（train 190、val 25、test 38），均有 28 点信息且图片可解码。
- `pages.json`：654 条（train 486、val 60、test 108），但所引用 Manga109 原页在本交接包中 0/654 可用。

## 复现

在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v

python -m scripts.train_baseline `
  --manifest data/processed/joint_a_c_v1/joint_manifest_candidate.json `
  --output models/baseline-region-v1 `
  --stages 3 --max-weak-trees 20 `
  --target-recall 0.995 --target-fpr 0.5 `
  --candidates 512 --seed 42

python -m scripts.mine_hard_negatives `
  --manifest data/processed/joint_a_c_v1/joint_manifest_candidate.json `
  --pages data/processed/manga109_detection_v2_margin10/pages.json `
  --model-dir models/baseline-region-v1 `
  --output results/hard-negatives/round-1 `
  --round 1 --score-threshold 0 --max-face-iou 0.3 `
  --step 12 --scale-factor 1.5 --pre-nms-limit 500
```

挖掘输出的 `review_status=needs_review` 表示候选仍需人工确认。人工接受后再把增强清单用于下一轮训练；不能直接把所有检测误报当作已确认背景。

## 2026-09-27 至 2026-09-28 本机验证结果

- 28 项单元测试通过；合成训练、导出重载、检测、逐 Stage 扫描日志和 28 点输出通过。
- 用联合候选执行 512 个特征候选、每 Stage 最多 20 棵树的真实区域训练，完成 3 个 Stage。第 0 级验证召回率 0.9960、条件误报率 0.4944，达到 0.995/0.5 目标；第 1、2 级分别为 0.9960/0.8989、0.9960/0.9250，未达到误报目标，日志正确标记 `best_available_within_limit`。这说明后续仍需增加候选/弱树、改善特征或加入真实困难负样本，不能把首版区域训练描述为最终模型。
- 初次启动真实整页挖掘时，预检准确报告 A 原交接包缺少 486 张 train 原页，并且没有创建半成品结果。用户随后下载约 4.07 GB 压缩包；解压后 `pages.json` 的 654 张引用页全部存在。
- 为避免弱模型数千个候选导致 NMS 二次复杂度失控，扫描支持 `pre_nms_limit`，按分数保留前 N 个再做 NMS，并在每页 `scan_summary` 记录原始候选数、截断数和最终检测数。
- `round-1-preview` 对固定前 6 张 train 页运行：每页 23159 个窗口，共 138954 个；原始候选每页 3900–7571 个，NMS 前各保留 500 个，最终 2168 个检测；每页选 20 个、共导出 120 个 IoU 合格候选。运行 211.63 秒，增强清单 6780 条（train 5080、val 612、test 1088），全部可由 `load_manifest` 解码，120 条均标记 `needs_review`。
- 使用 `scripts.review_hard_negatives` 导出 10 张上下文复核页，由 Codex 做 AI 辅助逐项视觉检查。接受 109 项；剔除索引为 4、5、10、11、48、67、78、79、86、94、95，原因是包含眼睛、脸侧、下巴等脸部局部，或与真值脸重叠较高且上下文不确定。决定文件显式覆盖 120/120，没有默认接受规则，并记录 `ai_assisted_visual_review`；最终课程材料前仍建议同学快速签字确认。
- 复核产物位于 `results/hard-negatives/round-1-preview/review/`：`reviewed_manifest.json`、`accepted_manifest.json`、`rejected_manifest.json`、`review_summary.json` 和 `accepted_augmented_manifest.json`。最终可训练清单共 6769 条，其中 109 条是已接受困难负样本；全量图像解码、来源划分和 bbox 校验通过。
- 使用与原模型完全相同的 512 候选、20 棵树上限、0.995/0.5 目标和种子 42，对 6769 条增强清单重训 `baseline-region-v2-hnm-r1`。候选特征列表完全一致，训练负样本从 2893 增至 3002。
- 重训没有改善总体误报：三级累计验证 FPR 从 0.4111 升至 0.4750，验证累计正样本通过数保持 251/250/249；109 个困难负样本在原模型和重训模型中均全部通过三级。因此保留原模型，不把重训模型标为改进版。逐 Stage 条件/累计结果见 B_RETRAIN_COMPARISON.md，机器可读对比在 `retrain-comparison.json`。
- 第 2 轮增加 `source-spread` 页面选择并跳过已挖页，从 12 本不同漫画各扫描 1 页；277908 个窗口产生 120 个候选，AI 辅助复核接受 106、剔除 14。两轮增强清单共 6875 条，其中困难负样本 215 个（109+106），公共读取器校验通过。
- 跨书目不加权模型的三级累计验证 FPR 为 0.4556，优于单书目模型 0.4750，但仍差于基线 0.4111。随后完成困难样本权重 1/2/3/5 扫描；增强模型最优为权重 3（FPR 0.4361，困难样本通过 203/215），权重 5 过拟合到 0.5167。验证正样本最终均通过 249/252，因此保留原基线，不继续按验证集试权重。
- 完成候选容量 512→1024 的预注册对照。原清单上的 1024 容量基线把三级累计验证 FPR 从 0.4111 降到 0.3333，正样本最终通过仍为 249/252；同容量加入 215 个困难样本并赋权 3 后为 0.3694，同时将困难样本通过数从 215 降到 191。冻结 `baseline-region-c1024` 作为测试候选，不再按验证集继续调容量。
- 冻结后完成固定 6 页 test 整页预览。6 页来自 6 本不同漫画，两个模型的页面、清单 SHA256 和扫描参数一致。512 基线为 TP/FP/FN=23/2255/56、P/R/F1=0.0101/0.2911/0.0195、耗时 199.66 秒；1024 基线为 29/2236/50、0.0128/0.3671/0.0247、耗时 308.73 秒。1024 三项指标均改善，但绝对误报仍很高；该结果只覆盖 6/108 页，不是完整 test 结论。
- 将逐窗口 Python 扫描改为等价分批 NumPy 计算，并用流式 top-K 限制 NMS 候选内存；同一真实 test 页 TP/FP/FN 保持 8/375/14，耗时从约 51 秒降到 6.5 秒。批量与逐窗口特征有合同测试，统一分数阈值也已接入公共 `scan_image`，不只存在于评价脚本。
- 在固定 6 个 validation 页完成 step=6/12、scale=1.3/1.5 的 2×2 消融；最大未过滤 F1 为 step=6、scale=1.3 的 0.0220。随后用同一配置扫描全部 60 页 validation，并选择统一分数阈值 13.8538650925。
- 最终冻结模型在 108/108 test 页完成 13,514,148 个窗口，遗漏页 0；TP/FP/FN=69/960/1360，P/R/F1=0.0671/0.0483/0.0561，总耗时 1774.11 秒、平均 16.43 秒/页。结果真实但质量有限，主要问题是阈值后仍有来源相关误报，同时召回很低。

## 当前阻塞与下一步

1. B 的规定任务已完成：真实训练、至少三级 Cascade、逐 Stage 日志、多尺度/NMS、两轮训练页困难负样本、重训对比、步长/尺度实验和完整 test 评价均有可复现证据。
2. 486 张训练页全量困难负样本挖掘属于可继续扩展项；当前已完成 18 页/13 本漫画、240 个候选全覆盖复核，不能称为 486 页全量。
3. 若团队后续继续改进检测质量，应重新注册 validation 协议或扩充训练来源；不得使用 108 页 test 反复调参。
