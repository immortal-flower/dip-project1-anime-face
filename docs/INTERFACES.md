# 基础接口约定 v1

这是团队工程约定，不是额外课程要求。变更需同步调用方与测试。

## 图像与坐标

- 外部图像：H×W×3、uint8、OpenCV BGR；灰度图为二维 uint8。
- 点使用 (x,y)，数组索引使用 [y,x]，坐标从 0 开始。
- bbox=[x1,y1,x2,y2]，右下边界不包含，宽高为 x2-x1、y2-y1。
- compute_11_channels(gray) 返回 11 张同尺寸 uint8 图，中间 int32。四值均值取 `(a+b+c+d+2)//4`，尚未与参考工程逐位对齐。
- 仅完整依赖区域有效时计算，其他置零。C1、C2、C3–6、C7–10 的右/下无效宽度分别为 1、3、3、7。
- 候选坐标暂限于 24×24 窗口左上 17×17，使所有通道依赖都在窗口内，保证裁剪训练与整图扫描一致。扩大时需按通道处理有效边界。

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

## B 检测输出

scan_image(bgr, detector_model, config) 返回 (predictions, logs)。predictions 为按 score 降序、已 NMS 的 `{bbox,score}` 列表，原图坐标，无脸空列表。

score 是各通过阶段 AdaBoost 分数裕量之和，不是概率。logs 当前记录各层尺寸、窗口数、各级通过数、候选数和总耗时，逐阶段耗时与误拒/误放待补。

train_cascade 使用训练集拟合，各级阈值取当前存活验证正样本最低分。基础版保召回但没有目标误报率控制。

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

这套基础命名不同于老师建议目录；正式交付需补齐真实点序编号、数据说明和实验配置。
