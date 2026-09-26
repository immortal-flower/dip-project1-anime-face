# 十一通道：六张图的观察与整数／浮点对照

当前结果为Python3.10环境下的 `results/channel_experiment_v2_color/`，第一格保留原始彩色，C0仍为灰度；下表采用该版measurements.json。旧图已移到本地历史归档。平滑抑噪和亮度鲁棒性的公式、受控实验与边界条件见 [A_REPORT.md](A_REPORT.md)。

本记录对应本地 `results/channel_experiment_v2_color/`。六张输入覆盖老师要求的四种情形：两张真实正面／近正面人脸、两张近正面动漫人脸、一张头发遮挡人脸、一张复杂漫画背景。它们用于观察通道，不加入训练或测试清单。

## 看图方法

本地 `index.md` 汇总六组拼图；每组还保存 `gray.png` 和 `C00.png`～`C10.png` 原尺寸通道。拼图按比例显示，白色留边属于排版；通道右侧／下侧黑边属于无效区域补零。每张图按原始 0～255 显示，没有分别拉伸对比度。拼图的 original 保留输入颜色，C0是转换后的灰度。

| 通道 | 内容 | 有效区域之外的处理 |
|---|---|---|
| C0 | 灰度原图 | 全图有效 |
| C1 | 2×2 平滑 | 右／下各 1 像素补零 |
| C2 | C1 上跨 2 像素平滑 | 右／下各 3 像素补零 |
| C3～C6 | C1 上水平、垂直、两种对角方向差分 | 右／下各 3 像素补零 |
| C7～C10 | C2 上更长距离的四方向差分 | 右／下各 7 像素补零 |

差分按 `(delta+255)//2` 编码，零差分对应 127。中灰不代表没有算出来，黑边也不能当作真实图像边缘让模型学习。当前 B 的候选特征限于 24×24 窗口左上 17×17 的共同有效区。

## 样本与逐图观察

| 编号 | 来源与输入尺寸 | 从图中观察到的现象 |
|---|---|---|
| real_hopper | 2.749 | 3.264 |
| real_astronaut | 1.027 | 1.259 |
| anime_frontal_1 | 0.334 | 0.662 |
| anime_frontal_2 | 0.183 | 0.130 |
| anime_hair_occlusion | 0.171 | 0.130 |
| manga_complex | 8.124 | 12.145 |

真实人像示例来源说明见 [Matplotlib 图像示例](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html) 和 [scikit-image astronaut 数据说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)。原始图片保存在本地，没有加入公开代码仓库。两个人像观察区分别为 xyxy `[135,105,395,375]`、`[145,25,300,195]`，不是训练真值。

这些图显示，真实人脸与动漫人脸都存在眼周、鼻部、下颌的方向变化；动漫图像常有更集中的线条与大块明暗，但发型、眼型和遮挡也可能破坏稳定结构。通道让 B 有机会选取“哪个位置相对另一个位置更亮”的弱特征；这些特征是否有效，还需真实训练和测试来证明。六张观察图不能证明动漫检测一定比真实人脸容易。

## 数值对照

`measurements.json` 保存逐图、逐通道的差异与本机耗时。实现使用三种方式：

1. 整数版：uint8 输出、int32 中间计算，保持现有检测接口。
2. float32 同取整版：每一步按整数版相同规则取整，用于检验运算的一致性。
3. float32 不取整版：保留中间小数，用于观察量化影响。

六张图的 11 个通道，整数版与同取整浮点版的最大绝对差均为 **0**，边界也完全相同。不取整浮点与整数输出的最大绝对差如下，统计时排除了无效边框：

| C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5 | 1 | 0.875 | 0.875 | 0.875 | 0.875 | 1.21875 | 1.21875 | 1.1875 | 1.1875 |

这验证的是本项目两个实现的对照，不代表已经与参考工程逐位对齐。取消取整可能改变后续弱树阈值判断；本轮没有切换已训练检测器的通道，也没有进行真实检测结果的一致性实验。

耗时为预热一次后七次调用的中位数，单位毫秒：

| 图像 | 整数版 | 浮点同取整 |
|---|---:|---:|
| real_hopper | 5.186 | 4.846 |
| real_astronaut | 1.773 | 1.478 |
| anime_frontal_1 | 0.609 | 0.789 |
| anime_frontal_2 | 0.277 | 0.194 |
| anime_hair_occlusion | 0.239 | 0.151 |
| manga_complex | 15.914 | 16.165 |

这是本机 Python/NumPy 的示例测量，受数组分配、调度、图像尺寸等影响，不能推导整数运算在 MCU 上的优势，也不能作为严格性能结论。验证环境为 Python 3.10.21、NumPy 2.2.6、OpenCV 4.14.0.94；耗时仅是当前机器的示例测量。

## 复现

若本地已有 `data/channel_examples_v1/samples.json`，直接运行（新目录保护既有结果）：

```powershell
python -m scripts.channel_experiment --manifest data/channel_examples_v1/samples.json --output results/channel_experiment_rerun
```

队友在自己的电脑上先备齐两张指定人像原图和课程数据，再准备观察样本。下面两个照片路径是占位符，需要替换为实际文件位置；脚本不会下载图片，也不要求安装 Matplotlib 或 scikit-image：

```powershell
python -m scripts.prepare_channel_examples --hopper "照片路径/grace_hopper.jpg" --astronaut "照片路径/astronaut.png" --output data/channel_examples_v1
```

默认 AnimeFace 和 Manga109 路径见 A_DATA_PREPARATION，也可通过 `--anime-root`、`--manga-root` 指定。想换图观察，可另建 JSON 数组清单，每条包含 `id`、`category`、`image`、`source`，可选 `preparation`；image 相对清单文件。图片需至少 8×8。原始数据、生成通道和计时 JSON 均保存在被 Git 忽略的目录。
