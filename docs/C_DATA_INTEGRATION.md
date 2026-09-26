# A 接入 C 的 28 点交付记录（2026-09-26）

输入来自用户指定的 `data/processed/animeface/`：`human_corrected_manifest.json` 和 `images.zip`。本轮只处理路径、格式、边界、重复和划分，不重新训练C模型，也不修改原标注。

## 收到的内容及检查结果

- 256条正例，点序全部为 `hysts28-v1`，均含28个坐标和28个可见性标志；C的标记为 `annotation_status=human_reviewed`、`reviewed_landmarks=true`，已原样保留，A不据此声称自己重新完成人工标注。
- C原划分为train192、val26、test38。原JSON中的路径来自队友电脑，已映射到本机图像，只有导出副本的image改为可移植相对路径。
- ZIP含完整63565张头像及一个目录项，CRC检测通过；256张选中图片与本机已有图片逐文件大小和CRC一致，所以没有重复解压全部图库，只在联合版本复制256张所需图片。
- 256张内部没有完全重复像素。按现有近似候选图构建关系连通组，未发现涉及C标注的跨集合候选冲突；这不等于证明所有近似重复都已穷尽。
- 3张图片合计4个“可见点”越界，保留原坐标并单列；其余253张通过格式和边界检查。问题明细见 [C_ANNOTATION_FEEDBACK.md](C_ANNOTATION_FEEDBACK.md)。
- 已对12张标注叠加图进行辅助观察，含上述3张。图片在 `results/c_integration/landmark_overlay_review.jpg`；这只是接口和边界检查，不取代C的全量人工修正。

## 输出目录与使用

`data/processed/joint_a_c_v1/`：

| 文件 | 用途 |
|---|---|
| `images/` | 本次256张C交付图片副本；与本地原图和压缩包一致 |
| `landmarks_imported.json` | 全部256条原标注的相对路径副本，保留原split、点、bbox、visibility、状态及其他字段；额外记录来源和检查结果 |
| `landmarks_validated.json` | 253条通过格式与可见点边界检查的记录；train190、val25、test38 |
| `annotation_issues.json` | 3张图片、4个越界可见点的精确坐标与图像尺寸 |
| `joint_manifest_candidate.json` | Manga109候选6407条＋C有效253条＝6660条，供共同代码读取 |
| `animeface_split_registry.csv` | 对现有63565张AnimeFace文件记录完全重复组、C锁定split和近似关联保留区 |
| `near_split_conflicts.json` | 本轮为空，供以后新增标注后继续检查 |
| `integration_report.json` | 输入SHA256、计数、原文件未改证明和检查状态 |

联合区域样本数量：

| split | 正例 | 负例 | 合计 |
|---|---:|---:|---:|
| train | 2067 | 2893 | 4960 |
| val | 252 | 360 | 612 |
| test | 456 | 632 | 1088 |
| 总计 | 2775 | 3885 | 6660 |

全部6660条已通过公共读取器，三个集合均能生成24×24检测样本，C的训练记录中每个点都有可见监督，各点可见样本数为46～186。记录见 `results/c_integration/interface_check.json`。这证明数据接口接通，不代表真实模型训练或准确率已完成。

## 防止后续扩充泄漏

注册表中542个文件属于C图像的完全重复别名，`locked_split`绑定C原有split；其中包括原256张。另有35个文件与C存在近似候选关系，标为 `near_candidate_hold`，应先确认关系再使用；剩余62988个文件是未分配的数据池。

未标注池没有被随意分成三个集合，避免未来与C发生冲突。当前实际使用的数据已有固定split。近似候选关联只是保留区，不会把C的坐标传播给相似图。原标注中的source_id也未被改写。

AnimeFace历史清点的像素指纹是 `sha256(BGR像素字节 + UTF-8的“高度,宽度”)`，导出中通过 `pixel_hash_policy` 标明；Manga109裁剪历史指纹为纯像素字节SHA256。两种格式不能直接混用，跨数据源比对时需先统一重新计算。

## 数据用途的边界

C交付的256个bbox都是整张头像图范围 `[0,0,width,height]`，适合现有裁剪头像和框内关键点的接口，不等于原始复杂场景中的完整人脸检测真值。联合清单用于区域训练和对接；整页检测评价仍使用Manga109的 `pages.json`，正式运行前需核对漏标与目标范围。

Manga109的6407条候选仍未全量质量验收，所以联合清单仍叫candidate。3张C越界例不改坐标、不自动置visibility为0，也没有强行夹到边缘；C确认后另导出v2即可恢复对应记录，沿用原split。

## 复现

使用已配置的`.venv`，输出目录必须不存在：

```powershell
python -m scripts.integrate_c_landmarks --manifest data/processed/animeface/human_corrected_manifest.json --archive data/processed/animeface/images.zip --output data/processed/joint_a_c_v1
```

默认使用本地AnimeFace图像、已有清点CSV、近似候选报告与Manga109候选清单，可通过命令行参数替换。脚本发现确定的完全重复跨split、错误点序、缺图、ZIP与图片不一致时会停止；不擅自重新划分。

包含C数据的新阶段包可用：

```powershell
python -m scripts.export_a_delivery --dataset data/processed/manga109_detection_v2_margin10 --channels results/channel_experiment_v2_color --joint data/processed/joint_a_c_v1 --output results/a_delivery_latest
```

测试已增加可见点越界不被截断、间接近似关系跨集合检测、C原字段保留与联合副本导出，当前共19项测试通过。原JSON的SHA256为 `b421c252260c2a5008f2524f720e5560da8a06d913b3b9e332f5862efbcb4c9e`，本轮结束前再次核对。
