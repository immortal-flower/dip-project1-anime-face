# C：Manga109真值框内的28点回归

## 目的

本实验只检查C的关键点回归，不评价B的人脸检测。程序直接读取Manga109 `annotations.v2020.12.18/*.xml` 中的官方 `face` 框，并在每个框内预测28个关键点。

Manga109只提供人脸框，不提供本项目采用的 `hysts28-v1` 28点真值。因此漫画结果只能人工观察，不能计算关键点NME，也不能代替AnimeFace的固定测试集评价。

## v4算法

1. 使用人脸框映射平均形状。
2. 运行HOG + PCA + Ridge整体回归。
3. 运行5轮局部像素对差分形状回归。
4. 使用验证集选择HOG权重0.55和残差增益1.1。
5. 融合12.5%的Fern/LBF非线性预测。
6. 在半径为人脸框短边5%的局部窗口内，根据梯度强度和空间距离进行受限边缘修正；移动强度0.5，只保留得分最高的20%候选像素。

所有参数只由AnimeFace验证集选择，测试集不参与选择。边缘修正用于让眼缘、眉毛和嘴线附近的点轻微贴近漫画线稿，搜索范围受到限制，避免远距离吸附到文字或分镜线上。

## 结果

| 模型 | 验证集NME | 保留测试集NME |
|---|---:|---:|
| 原HOG/形状50%融合 | 0.07182 | 0.08635 |
| v4 | **0.06733** | **0.08251** |

验证集相对改善6.25%，测试集相对改善4.45%。本地另在固定的18张Manga109测试页上读取181个官方人脸框并全部输出28点；由于没有漫画28点真值，该数字只代表处理覆盖率。

## 运行

先从 `deliverables/c-landmark-hybrid-edge-v4.zip` 解压模型，再运行：

```powershell
python -m scripts.annotate_manga109_landmarks `
  --manga-root data/Manga109_released_2026_05_21 `
  --model-dir models/landmark-hybrid-edge-v4 `
  --pages Akuhamu:031 BurariTessenTorimonocho:025 `
  --output results/manga109-oracle-landmarks-v4
```

每一页会生成一张整页可视化和一份JSON。JSON明确记录 `bbox_source=Manga109 annotations.v2020.12.18 face`，便于区分检测框实验。

## 限制

- 训练关键点仍来自190张AnimeFace图像，与Manga109黑白线稿存在明显域差异。
- 小脸、侧脸、旋转脸、严重遮挡和夸张表情仍可能接近平均形状。
- 局部边缘只知道线条强弱，不理解线条是五官、头发、网点还是阴影。
- 若要显著提升Manga109效果，需要补充少量代表性漫画人脸28点标注进行目标域适配。
