# A 部分：数据位置、负样本与划分

最新版本：`data/processed/manga109_detection_v2_margin10/` 保留原样本身份和划分，加入正例扩边；其中 `manifest_candidate.json` 隔离53张已知问题后保留6407张，仍待最终验收。v1保留用于历史溯源，下文初始数量不变。新版方法、逐项问题和运行命令见 [A_WORK_LOG.md](A_WORK_LOG.md)。

C已交付256张，现已接入 `data/processed/joint_a_c_v1/`；253张通过格式及边界检查，与漫画候选合并为6660条。原C点与split未修改，3张越界例单列。见 [C_DATA_INTEGRATION.md](C_DATA_INTEGRATION.md)。下文C分支快照是交付前的历史信息。

## 本地数据位置

以下路径相对仓库根目录，数据只保存在本地：

| 内容 | 路径 |
|---|---|
| AnimeFace 原图 | `data/animeface/images/` |
| AnimeFace 读取与完全重复检查 | `data/inspection/animeface/` |
| Manga109 压缩包 | 已不在仓库 data 中；重新校验时传入实际 ZIP 路径 |
| Manga109 实际数据根目录 | `data/Manga109_released_2026_05_21/` |
| Manga109 解压完整性记录 | `data/inspection/manga109/archive_verification.json` |
| 首批检测样本与划分 | `data/processed/manga109_detection_v1/` |

2026-09-25 已按用户整理后的目录更新：AnimeFace 去掉 raw 层，Manga109 去掉重复的同名外层。给脚本的根目录应直接包含 `annotations/` 和 `images/`。只使用本版的 `annotations/`，不要混用 `annotations.v2020...` 等旧标注。

```text
data/
├── animeface/images/
├── Manga109_released_2026_05_21/
│   ├── annotations/
│   ├── images/
│   └── books.txt
├── inspection/
└── processed/manga109_detection_v1/
```

## 解压核验

2026-09-25 检查：Manga109 压缩包的 11,149 个普通文件已完整解压，逐文件大小与 CRC 校验全部通过。其中 `images/` 包含 10,602 个图像文件；漫画双页图文件数不等于单页计数。

```powershell
python -m scripts.verify_dataset_archive --archive "实际压缩包路径.zip" --destination data --report data/inspection/manga109/archive_verification_new.json
```

上面的压缩包路径是占位符，需要替换为真实文件位置。ZIP 内部本身包含一层 `Manga109_released_2026_05_21/`，因此当前目录结构下 `--destination` 应为 `data`，不要再传数据集根目录，否则会重复嵌套。旧 CRC 记录保留为迁移前的历史验证，不表示迁移后重新读取了 ZIP。

如果有缺失文件，可加 `--extract-missing`。该选项只补缺失文件，已有但不一致的文件会报告异常而不是覆盖。读取压缩包不需要先移动压缩包或处理 WPS 文件关联。

AnimeFace 之前已检查 63,565 张图全部可读，按解码后像素内容识别出 41,522 个唯一图像和 22,043 个重复副本。原图仍全部保留；这只是完全重复检查，不代表近似重复或来源分组已完成。

## 首批 Manga109 检测数据

生成命令（输出目录必须为空或不存在，以保护已有划分与复核结果）：

```powershell
python -m scripts.prepare_manga_detection_data --root data/Manga109_released_2026_05_21 --output data/processed/manga109_detection_v2
```

规则：

已有 v1 数据可直接使用，上面以 v2 示范重新生成时的新输出位置；目录迁移不需要重新生成样本或随机划分。

- 固定种子 42，先按书／系列分组再划分。能够从名称识别的 `volNN` 分卷归入同组，防止同系列不同卷跨集合；其他书名作为独立组，语义上同系列但命名不同的情况仍需核对。
- 109 本书识别为 104 组，约 75%/10%/15% 分到 train/val/test。按组分配后，实际书本数及样本比例不会恰好等于该比例。
- 每本选择 6 个有画框和人脸标注的内页；每页最多 4 个尺寸至少 24×24 的人脸正样本和 6 个负样本。
- 负样本来自真实漫画图案、文字、衣物及背景，候选框边长 24–192 像素；与任何人脸框向外扩展 15% 后的范围有交集即排除。不能仅按较小的 IoU 判断无脸，因为一个大框也可能包含整张小脸。
- 在原尺寸候选区域过滤灰度标准差低于 12 的近乎空白区域；缩到 24×24 后仍可能变成弱纹理，后续检查单列此类样本。负样本是初始背景采样，不是经过检测器挖掘的困难负样本。
- XML 的最大坐标保守加 1 后转换到内部右下边界不含的约定，并裁剪到图内；此约定写入元数据。图片尺寸与 XML 必须匹配，防止版本错配。
- 存储 24×24 PNG 样本，避免当前读取器反复把大幅漫画页加载进内存。每个样本保留原图路径、原图框、页号和分组信息，可回溯。
- 对生成的样本进行完全相同像素去重；当前批次没有发现重复裁剪。近似重复与原标注漏标仍需复核。

本轮结果：

| 集合 | 漫画书数 | 正样本 | 负样本 | 合计 |
|---|---:|---:|---:|---:|
| train | 81 | 1,880 | 2,916 | 4,796 |
| val | 10 | 232 | 360 | 592 |
| test | 18 | 424 | 648 | 1,072 |
| 总计 | 109 | 2,536 | 3,924 | 6,460 |

原始来源共 654 个页面。此批数据是待复核的第一批真实检测样本，不表示最终课程数据集或正式模型评估已完成。

## 输出文件与使用

- `manifest.json`：兼容现有 `load_manifest()` 的正负样本清单，图像路径相对该清单。
- `book_splits.json`：固定的分组和书本分配；后续同源图的所有采样都应沿用。
- `pages.json`：所选整页图片和每页完整人脸框集合，供 B 进行全图检测评价。全图评价不能只用被选中的 4 个正样本框。
- `review.csv`：初始待复核索引，状态均为 `needs_review`。正式判断用下方工具另存 `decisions.csv`，避免把初始索引当成已经复核的结果。
- `preview_negative.jpg` / `preview_positive.jpg`：各自前 48 个样本预览，用于快速检查；不是全数据集人工验收。
- `report.json`：生成参数、注释内容指纹、数量和限制。

人工复核时，负样本要检查原 XML 漏标的人脸及不合适区域；正样本要筛查过度侧脸、旋转、遮挡等是否符合实验范围。已有划分应保留，不因筛选后数量变化而重新随机分组。

这批数据不含关键点，不能直接交给要求关键点标注的 `scripts/train_baseline.py`。B 可使用 `load_manifest()`、`detection_samples()` 和 `train_cascade()` 单独开展检测训练；本次工作仅准备数据，尚未运行真实模型训练。

## 自动检查与人工复核

先运行一致性检查，不需要重新采样：

```powershell
python -m scripts.audit_detection_data --manifest data/processed/manga109_detection_v1/manifest.json --pages data/processed/manga109_detection_v1/pages.json --output results/a_data_audit_v1.json
```

本轮 6,460 张全部通过，详细记录在上述 JSON；其中 `low_texture_after_resize` 单列 35 张缩小后灰度标准差低于 12 的负样本。`results/low_texture_review_v1.png` 是本次生成的本地总览，编号按 JSON 中该列表顺序对应。逐图看过总览，主要是低对比度网点、少量线条及大块空白；这只是辅助观察，没有据此删除或宣称完成人工验收。

自动避开的是已有标注框，不保证 XML 没有漏脸。人工还需要看源图上下文和课程要求。下面的命令会打开复核窗口，需要人操作，适合你准备好之后运行：

```powershell
python -m scripts.review_detection_samples --manifest data/processed/manga109_detection_v1/manifest.json --output data/processed/manga109_detection_v1/decisions.csv --label all
```

- 左侧显示 24×24 样本，右侧显示原图附近区域，绿色框是裁剪位置。
- A 保留，R 剔除，N 暂时跳过，P 上一张，Q 保存并退出；每次判断也会自动保存。
- 再次运行相同命令，会从首个未记录判断的样本继续。只看负样本可用 `--label negative`；正样本用 `--label positive`，判断可以保存在同一份 CSV。
- `decisions.csv` 是新判断记录；不要把原始 `review.csv` 作为这个输出文件。

人工复核后，另存已接受样本清单：

```powershell
python -m scripts.apply_sample_review --manifest data/processed/manga109_detection_v1/manifest.json --review data/processed/manga109_detection_v1/decisions.csv --output data/processed/manga109_detection_v1/manifest_reviewed_v1.json
```

只导出明确 `accepted` 的样本，未判断和 `rejected` 都不混入；路径随新清单位置转换，标签、坐标和 split 保持原样。原始文件不变，输出目录中另有数量汇总。只复核一部分时导出的是子集，B 使用前要核对各集合的正负数量；导出成功不代表最终数据集齐备。当前复核窗口尚待用户实操，导出逻辑已做自动测试。

## 通道与评价的交付

六张图的通道图、观察和整数／浮点对照已完成首版，见 [CHANNEL_EXPERIMENT.md](CHANNEL_EXPERIMENT.md)。日常训练直接调用 `compute_11_channels()`，不要求给全部 6,460 张图保存通道 PNG。

已提供 `scripts/evaluate_detection_results.py`，输入 B 的整页预测与 `pages.json`，逐图匹配后汇总 P/R/F1；预测格式和命令见 [INTERFACES.md](INTERFACES.md)。本轮没有运行真实检测训练，没有生成真实效果指标。

## 与 C 队友对接

已取得远程 `feature/landmarks-demo` 分支（2026-09-25 检查时提交为 `5e40ccc`）。其文档定义点序 `hysts28-v1`，记录 256 张预标注的 192/26/38 临时划分，并注明人工复核仍待完成。用户告知 C 已完成标注，实际最新状态需以 C 交付的文件为准；不要根据旧分支记录否定最新工作，也不能把模型预标注自动标成已人工确认。

同步代码不会下载被 Git 忽略的标注数据。目前本地未发现 C 的实际清单；AnimeFace 的最终联合划分暂未重新生成，避免覆盖其已有成果。

拿到 C 的清单时需要：

1. 获取实际 manifest 和已有划分，确认 `image`、`source_id`、`split`、`bbox`、`landmarks`、`visibility`、`landmark_order` 及人工复核状态。
2. 将 C 电脑的图像路径映射到本机 AnimeFace 图像，保留 28 点和复核记录。
3. 按已有像素指纹检查同图及重复别名是否跨集合；再检查近似重复。存在冲突应输出问题清单，与 C 一起确定修正，不能直接忽略或静默重分。
4. 在保持无冲突的既有划分前提下，扩充尚未使用的 AnimeFace 图像和 Manga109 数据，形成共同使用的版本。
5. 若联合使用 AnimeFace 正样本和 Manga109 负样本，保留 Manga109 正样本及来源统计，检查模型是否只学会区分彩色头像与黑白漫画。

## 验证与仓库范围

截至 2026-09-26，12 项单元测试通过；6,460 个样本全部可读取，3,924 个负样本均不与扩展的人脸标注相交；样本划分与整页划分一致，没有同来源或相同裁剪像素跨集合。合成训练与模型导出／重新加载流程通过，不能替代真实效果评价。

目录迁移后已修复 `manifest.json` 的 6,460 个原图引用及 `pages.json` 的 654 个整页引用。标注内容指纹、样本标签、坐标和集合分配保持一致。迁移前清单备份及更新记录保存在本地 `data/inspection/path_migration_20260925_backup/` 和 `path_migration_20260925.json`。

原始图片、采样图片、标注副本和预览均留在被 Git 忽略的 `data/`。代码与说明可随 A 分支协作，Manga109 数据不进入公开仓库。
