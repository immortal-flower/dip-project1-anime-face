# A 工作记录：查漏补缺（2026-09-26）

本轮承接 `13a3adf` 后的要求核查。目标是补齐可以独立完成的缺口，同时把实际发现的质量问题留痕。老师原文未改，未推送 GitHub，未改 C 的点标注或已有划分。

## 本轮结果

| 工作 | 做了什么 | 实际验证／产物 |
|---|---|---|
| 正样本扩边 | 在原框四侧默认扩 10%，碰到新邻脸则逐次减半，最多减 8 次后退回原框 | 新版 `data/processed/manga109_detection_v2_margin10/`；56 张减小扩边，86 张原框已与邻脸重叠并列为待查 |
| 保留旧数据与真值 | 新版沿用同一批 6,460 样本，负例直接复制，正例另外记录 `source_gt_bbox` | 3,924 张负例文件逐一不变；2,536 原人脸框保留；654 页真值和固定划分不变；`results/crop_version_validation.json` |
| 自动质量队列 | 全量计算纹理、边界、框形状、已知邻脸交叠提示 | `results/quality_review_v2/quality_queue.csv`；低纹理35、触边56、异常宽高比240、原框邻脸交叠86；提示可重叠 |
| 实际视觉抽查 | 按来源／标签／集合稳定抽取68张，生成9张上下文检查页并逐张查看 | 34正、34负，逐条判断保存在 `ai_observations.json`；明确标记为 Codex 的 AI 抽查，不是人工验收 |
| 隔离已知问题 | 14张明显侧脸／局部脸／无五官脸，4张含人脸或疑似微小人脸的负例，另加35张低纹理负例 | 共隔离53张；候选 `manifest_candidate.json` 留下6,407张，原始 `manifest.json` 未删改 |
| 近似重复筛查 | 64位差分指纹≤4位差，32×32灰度平均绝对差≤12，宽高比差≤15%；低纹理另列 | Manga109跨集合候选812对；AnimeFace的41,522张唯一像素图候选1,564对；全部图片可读 |
| 近似重复观察 | 查看两份报告各前24对 | AnimeFace前24对视觉高度相似，支持继续归组；漫画前24对多为相似边线，不能证明同源，不自动删或合并 |
| 系列归组辅助 | 109书记录已有系列组和split，按书名相似度≥0.75列潜在异名关系 | `series_group_review.csv`；本次没有额外书名相似候选，不等于语义同系列已全部查清 |
| 环境复现 | 项目本地工具安装uv，创建`.venv`，安装实际依赖并固定版本 | uv0.12.19、Python3.10.21、NumPy2.2.6、OpenCV4.14.0.94；`requirements-lock.txt` |
| 环境证据 | 保存真实命令输出、测试、合成流程及浏览器核验页截图 | `results/environment_a/`，16项测试通过；合成训练、导出/加载、检测、28点和空结果检查通过 |
| 特征定义导出 | 保存11通道来源、偏移、取整、边界和窗口配置；基础训练导出时同时保存定义和SHA256 | `configs/feature_definition.json`；用定义文件计算的参考通道与实现完全一致 |
| 报告与图 | 补平滑抑噪、亮度不变条件和反例；拼图首格保留原始颜色，C0仍是灰度 | `docs/A_REPORT.md`；`results/channel_experiment_v2_color/`、`results/channel_principles.json` |
| 阶段交接包 | 导出代码、24×24样本、清单、通道图、质量记录、环境证据，逐文件SHA256核验 | `scripts/export_a_delivery.py`；本地输出 `results/a_delivery_20260926/`，属于阶段候选包 |

## 候选数据数量

| 集合 | 原版正例 | 候选正例 | 原版负例 | 候选负例 |
|---|---:|---:|---:|---:|
| train | 1880 | 1877 | 2916 | 2893 |
| val | 232 | 227 | 360 | 360 |
| test | 424 | 418 | 648 | 632 |
| 总计 | 2536 | 2522 | 3924 | 3885 |

这只是训练区域清单的候选筛选。剩下6,407张仍为 `needs_review`，不是已通过全量验收的数据。筛选依据是预先明确的样本质量范围，不是任何模型的测试误差。整页评价真值保持原样，没有删掉难脸以抬高指标。

## 关键发现与处理依据

1. 自动避让XML框不足以保证负例没有脸。抽查编号40是漫画里的人物肖像，60、67包含人脸，43含微小人物头部。原XML可能遗漏或不包含此类描绘，已按疑似人脸背景隔离，不能继续称为“所有负样本都无脸”。对应来源页的真值完整性也需复核。
2. 正例有明显侧脸、分镜只露眼周和无五官脸。抽查18个隔离建议中14个属于正例；本轮没有把这些图改标成负例，避免引入相互矛盾的标签。
3. 六张通道示例不代表训练数据类型覆盖；本次抽查另发现短发／长发、闭眼、夸张眼型、轻微倾斜、墨镜和头发遮挡、不同线条画风。负例可见头发、衣物、建筑、车辆、机械、对白、拟声词。它们证明有实例，不是全数据集分布统计；手部等具体类别还要继续核对。
4. 漫画重复候选里的简单线条容易产生假阳性。没有用感知指纹自动改变固定划分；AnimeFace的候选关系应与C的原清单及完全重复别名一起对接。
5. 全图评价仍以原XML的全部脸为真值，包含小脸、侧脸和遮挡；近正面训练目标与全脸评估范围有差异。先共同确定正式目标／忽略规则，再运行最终指标；不要临时删真值。

## 复现本轮的主要命令

以下从仓库根目录运行，先激活 `.venv`。老师要求的uv在本机装于 `results/tools/uv/bin/uv.exe`；队友可用自己的uv命令。缓存、uv及下载的解释器也都留在本地results中，没有改系统Python。

```powershell
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
.venv/Scripts/Activate.ps1
python -m unittest discover -s tests -v
```

新数据目录必须不存在。已有v2可直接读，重做时把输出名改成新版本：

```powershell
python -m scripts.revise_detection_crops --source data/processed/manga109_detection_v1 --output data/processed/manga109_detection_v2_margin10 --positive-margin 0.10
python -m scripts.audit_detection_data --manifest data/processed/manga109_detection_v2_margin10/manifest.json --pages data/processed/manga109_detection_v2_margin10/pages.json --output results/a_data_audit_v2.json
python -m scripts.prepare_quality_review --dataset data/processed/manga109_detection_v2_margin10 --output results/quality_review_v2
python -m scripts.find_near_duplicates --manifest data/processed/manga109_detection_v2_margin10/manifest.json --output results/near_duplicates_manga_v2
python -m scripts.find_near_duplicates --inventory data/inspection/animeface/inventory.csv --image-root data/animeface --output results/near_duplicates_anime_v1
python -m scripts.build_detection_candidate --dataset data/processed/manga109_detection_v2_margin10 --observations results/quality_review_v2/ai_observations.json --output data/processed/manga109_detection_v2_margin10/manifest_candidate.json
python -m scripts.channel_experiment --manifest data/channel_examples_v1/samples.json --output results/channel_experiment_v2_color
python -m scripts.channel_principles_experiment --output results/channel_principles.json
python -m scripts.export_a_delivery --dataset data/processed/manga109_detection_v2_margin10 --channels results/channel_experiment_v2_color --output results/a_delivery_20260926
```

AI观察记录来自本次实际看图，不是脚本自动产生的分类；队友重新抽查时应保存自己的判断，不复制“已检查”的声明。新环境验证可用 `python -m scripts.verify_a_environment --uv 实际uv路径 --output 新的记录目录`。

## 尚未完成，不能写成完成

- 全量样本最终验收、近似重复候选逐对确认、系列语义归组最终核对。人工工具和候选资料已齐，图像质量仍需继续复核。
- C最新28点清单、点序和既有split联合对接。
- B真实模型预测、正式P/R/F1、困难负样本再训练、全系统整数／浮点输出比较。
- 最终课程整包。当前阶段包不包含完整漫画原图、全量AnimeFace、C数据和真实模型；原图复核与整页评价需要按数据来源另行获取。不要将阶段候选包当作最终提交。

## 运行中遇到的问题

默认网络沙箱阻止下载，按本轮环境复现授权联网将uv安装到项目工具目录。Python下载后出现Windows临时重命名重试，最终安装成功。uv首次冻结版本时默认缓存目录不可写，改为项目内缓存后已正确生成两项锁定依赖。一次包含中文的PowerShell管道脚本遇到编码错误，改为UTF-8文件执行后成功；没有因此修改原始数据。
