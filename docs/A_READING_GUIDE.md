# A 部分从哪里开始读

先看 [A当前文件入口](A_CURRENT_FILES.md) 找到最新数据。工作经过见 [A_WORK_LOG.md](A_WORK_LOG.md)；下文已统一到当前版本。

你负责的是：把原始图片整理成可信的训练材料，计算十一通道，并提供检测评分方法。B 用这些材料训练检测器；C 做 28 点回归。现在已经有可运行代码，你可以先顺着一条样本的经历理解它。

## 先看结果，再读实现

1. 打开本地 `results/quality_review_v2/sheet_01.jpg` 等复核图，看裁剪样本与原图上下文。它们是抽查结果，不代表全部人工验收。
2. 打开 `data/processed/manga109_detection_v2_margin10/manifest_candidate.json`，找刚才图片的文件名。`label=1` 是正样本，`label=-1` 是负样本；`split` 是它被分到的集合。
3. 看 `source_image` 和 `source_bbox`：它从哪张漫画页、哪个框裁下来。`bbox=[0,0,24,24]` 则属于裁好的小图，两者坐标不能混用。
4. 打开本地 `results/channel_experiment_v2_color/index.md` 查看六组通道图，再看 [通道观察与实验](CHANNEL_EXPERIMENT.md)。每张原图都有 C0～C10 十一张变换结果。
5. 最后按下面的顺序读代码。已经在各文件入口、函数和关键计算处添加中文注释。

## 第一站：正负样本怎么产生

主入口是 [prepare_manga_detection_data.py](../scripts/prepare_manga_detection_data.py) 的 `prepare()`。

```text
读取漫画页和老师数据集已有的 XML 人脸框
  → 按漫画书／可识别系列划分 train、val、test
  → 从人脸框裁正样本，从避开人脸的区域裁负样本
  → 缩成 24×24，保存图片、来源、坐标和集合
```

文件名中的 `pos` / `neg` 是保存时生成的名字，分别对应 positive / negative。同一个漫画书文件夹里有两种样本，所以看到人脸不一定说明负样本采错，先看文件名和清单。

核心辅助函数在 [data_preparation.py](../src/data_preparation.py)：

| 函数 | 阅读时关注什么 |
|---|---|
| `book_group()` / `split_groups()` | 同一本书及识别出的同系列先归组，再分集合 |
| `stable_rng()` | 固定种子让采样能够复现 |
| `sample_background()` | 随机候选框避开全部已标注人脸，过滤过于空白的区域 |
| `audit_rows()` | 检查同来源、相同像素有没有跨集合 |

这一步读取已有 XML 标注，不是靠我们自己的检测器寻找人脸。XML 若漏标，人脸仍可能被采进背景，所以自动检查后还要复核。

## 第二站：数据划分是什么

`train` 用于学习，`val` 用于选参数和调整阈值，`test` 留到最终测效果。先分原始来源再裁图，避免同一幅漫画的相似裁剪一张用于学习、一张用于考试。

当前 Manga109 的固定划分已经生成在 `book_splits.json`，6,460 个样本沿用它。不要为了凑数量随意重新分配。C的清单已接入联合版本，保留原split；253张通过格式与边界检查，3张待确认。完全重复别名绑定C原划分，近似关系另列待查。

## 第三站：十一通道到底是什么

打开 [channels11.py](../src/channels11.py)，先读 `compute_11_channels(gray)`：

- C0：灰度原图。
- C1、C2：两种范围的平滑，让细碎变化减弱。
- C3～C6：较短距离的四个方向像素差。
- C7～C10：较长距离的四个方向像素差。

通道是确定的公式，不需要训练。梯度通道的中灰约 127 表示没有亮度差，右侧／下侧黑边是不能完整计算而补的零。它们是供 B 挑选特征的材料，还不是最终检测结果。

`compute_11_channels_float()` 是对照实验版本。先理解整数版，再看它为什么保持相同取整就能得到相同结果。

## 第四站：交给 B 什么

- `manifest_candidate.json`：6407条漫画候选的路径、标签和固定划分；联合使用时读 `joint_a_c_v1/joint_manifest_candidate.json`。最终人工验收后另导出版本。
- `pages.json`：整页图片和该页全部人脸框，用于检测评价。
- `compute_11_channels()`：运行时计算通道，不必提前给每张训练图保存十一张 PNG。
- [detection_metrics.py](../src/detection_metrics.py)：把 B 的预测与真值逐图匹配，再汇总 Precision、Recall、F1。格式见 [接口约定](INTERFACES.md)。

## 当前做到哪了

| 内容 | 当前状态 |
|---|---|
| 当前漫画候选 | 2,522 张正样本、3,885 张负样本；原6460条完整记录保留 |
| Manga109 数据划分 | 已固定；文件、来源、像素跨集合和负样本几何避让检查通过 |
| 人工复核 | 工具已提供，未把任何样本擅自标成已人工验收；35 张缩小后低纹理负样本已单列 |
| 十一通道 | 整数实现、六张观察图、观察分析和浮点对照已完成首版 |
| 检测评价 | 逐图和整集评价代码已验证，等待 B 的真实预测 |
| AnimeFace + C 标注 | 256张已收到，253张通过检查，联合清单6660条；3张边界例保留原样待确认 |

接下来你回看时，可以先只完成“结果预览 → 一条清单 → `prepare()` → `compute_11_channels()`”。复核操作和运行命令在 [A 数据准备说明](A_DATA_PREPARATION.md)，不用一次读完 B、C 的训练代码。
