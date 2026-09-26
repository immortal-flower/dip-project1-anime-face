# 基础接口约定 v1

这是团队工程约定，不是额外课程要求。变更需同步调用方与测试。

## 图像与坐标

- 外部图像：H×W×3、uint8、OpenCV BGR；灰度图为二维 uint8。
- 点使用 (x,y)，数组索引使用 [y,x]，坐标从 0 开始。
- bbox=[x1,y1,x2,y2]，右下边界不包含，宽高为 x2-x1、y2-y1。
- compute_11_channels(gray) 返回 11 张同尺寸 uint8 图，中间 int32。四值均值取 `(a+b+c+d+2)//4`，尚未与参考工程逐位对齐。
- 仅完整依赖区域有效时计算，其他置零。C1、C2、C3–6、C7–10 的右/下无效宽度分别为 1、3、3、7。
- 候选坐标暂限于 24×24 窗口左上 17×17，使所有通道依赖都在窗口内，保证裁剪训练与整图扫描一致。扩大时需按通道处理有效边界。
- 完整通道定义保存在 `configs/feature_definition.json`，版本 `dip11-int-v1`。基础训练导出复制该文件并在config记录版本、文件名及SHA256；B独立训练入口也应同步保存定义，避免模型与特征版本混淆。
- `compute_11_channels_float(gray, quantize=True)` 返回 11 张 float32 图，仅供实验。True 每一步同整数版取整；False 保留小数。B 的训练和推理继续使用整数版，不自动切换。

## 数据清单

load_manifest(path) 接收 UTF-8 JSON 数组，每条记录为一个标注区域：

```json
[{"image":"images/example.jpg","source_id":"original-page-0001","split":"train","label":1,"bbox":[10,20,110,120]}]
```

- image 相对清单文件；source_id 表示原始图/页，同源裁剪不能跨 split。
- split=train/val/test，label=1 人脸或 -1 背景。bbox 为图内整数坐标。
- 关键点记录额外含 landmarks（28×2 有限数值数组）和 visibility（28 个 0/1）。点坐标属于原图，点序由编号说明固定。
- 老师原文用 `{x,y,visibility}` 点对象，导入时显式转换为上述内部格式，不能直接混用。
- detection_samples 会裁剪并缩放为 24×24；训练不重新分配 split。
- 本清单用于区域训练。正式全图评价需要每张原图的完整真值框集合，不能用一个裁剪标签代替整图真值。
- v2正例的 `source_gt_bbox` 是原始XML真值，`source_bbox` 是扩边后的实际裁剪框，`bbox` 仍表示24×24训练窗口。`pages.json`的真值保持原始坐标，不随扩边改变。候选清单的 `needs_review` 不能解释为已经人工验收。

## B 检测输出

scan_image(bgr, detector_model, config) 返回 (predictions, logs)。predictions 为按 score 降序、已 NMS 的 `{bbox,score}` 列表，原图坐标，无脸空列表。

score 是各通过阶段 AdaBoost 分数裕量之和，不是概率。logs 当前记录各层尺寸、窗口数、各级通过数、候选数和总耗时，逐阶段耗时与误拒/误放待补。

train_cascade 使用训练集拟合，各级阈值取当前存活验证正样本最低分。基础版保召回但没有目标误报率控制。

## A 的整集检测评价

`evaluate_dataset(pages, predictions, threshold=0.5)` 中 pages 为含 `page_id`、`bboxes` 的整页记录列表；predictions 为以实际 page_id 为键的 JSON 对象：

```json
{"实际page_id": [{"bbox": [10, 20, 50, 60], "score": 1.2}], "另一实际page_id": []}
```

坐标为原图 xyxy，分数越高越先匹配；无检测明确写空列表。预测框和真值框必须具有有限坐标与正面积，score 为有限数值。每个预测匹配当前未匹配真值中 IoU 最高且不低于阈值的一个框，真值最多匹配一次；重复检测算 FP。

先逐页匹配，再累计 TP/FP/FN 计算 micro Precision、Recall、F1，不对每页 F1 求平均。分母为零时对应指标为 0。遗漏预测页按空检测计入并列入 `missing_prediction_pages`，避免只统计有结果的图片；未知 page_id 报错。正式报告应先核查遗漏列表，区分无检测与未运行。

```powershell
python -m scripts.evaluate_detection_results --pages data/processed/manga109_detection_v2_margin10/pages.json --predictions results/test_predictions.json --split test --output results/detection_test_metrics.json
```

`test_predictions.json` 需要由 B 真实运行后生成，只包含选定集合的 page_id；不要填模拟框冒充实验结果。评估使用整页全部标注，不能使用少量采样正例的框作为全部真值。

## C 回归与最终接口

- predict_shape(model, gray, bbox) 返回原图坐标 28×2 数组。
- 形状按框左上角及宽高归一化；仅可见点拟合，每个点至少一个可见训练样本。
- nme(prediction, truth, visibility, normalizer) 显式传入正归一化距离。全部不可见返回 None，汇总应剔除并记录数量。
- AnimeFaceDetector(model_path) 接收模型目录，detect(bgr) 返回 `[{bbox,score,landmarks}]`，分数降序、原图坐标、无脸空列表、不弹窗。

## 模型导出

- detector.json：候选通道/坐标、Stage 弱树/权重/阈值、训练种子和日志。
- landmark.npz：平均形状、采样偏移和回归矩阵，不使用 pickle。
- config.json：格式版本、窗口、通道取整/边界、金字塔、NMS、点数和 synthetic 标记。
- splits.json：图像、来源、划分、标签和 bbox。真实数据版本与内容校验待补。
- feature_definition.json：完整通道定义；config中新增feature_version、feature_definition_file、feature_definition_sha256。现有检测读取接口仍兼容，多出的配置不改变当前预测行为。

这套基础命名不同于老师建议目录；正式交付需补齐真实点序编号、数据说明和实验配置。
