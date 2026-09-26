# A 数据准备：当前版本

更新：2026-09-26。完整路径索引见 [A_CURRENT_FILES.md](A_CURRENT_FILES.md)，历史经过见 [A_WORK_LOG.md](A_WORK_LOG.md)。以下均为仓库根目录下的相对路径。

## 数据位置

| 内容 | 路径 |
|---|---|
| AnimeFace原图 | `data/animeface/images/` |
| Manga109原图和XML | `data/Manga109_released_2026_05_21/`，直接含images、annotations、books.txt |
| 原始数据完整性与重复检查 | `data/inspection/` |
| 当前漫画检测数据 | `data/processed/manga109_detection_v2_margin10/` |
| C原始交付 | `data/processed/animeface/` |
| C导入与联合数据 | `data/processed/joint_a_c_v1/` |

## 正负样本及划分

先按漫画书／可识别系列归组，seed=42固定约75/10/15的来源划分，再裁样本。109本书对应104组、654页；同源裁剪不跨集合。实际样本数不强行凑比例。

负例来自真实漫画背景，避开扩展15%的人脸XML框，缩小前过滤低纹理。正例四侧默认扩10%，若引入新邻脸则减少扩边。24×24 PNG用于区域训练，source_gt_bbox保留XML原框，source_bbox记录实际裁剪，bbox属于小图；pages.json保持完整整页原真值。

完整manifest.json保留6460条；manifest_candidate.json隔离14张不适合正例、4张含脸或疑似含脸背景及35张缩小后低纹理负例，留下6407条：

| 集合 | 正例 | 负例 |
|---|---:|---:|
| train | 1877 | 2893 |
| val | 227 | 360 |
| test | 418 | 632 |

这是候选筛选，并未完成全量人工验收。XML可能漏脸，不能用“避开XML框”证明负例都无人脸。整页评估仍需核对漏标和目标范围，不删除难例来提高分数。

## 当前检查和人工复核入口

使用已配置的.venv，在项目根目录运行。下列输出用新名字保存，避免覆盖已有证据。

```powershell
python -m scripts.audit_detection_data --manifest data/processed/manga109_detection_v2_margin10/manifest.json --pages data/processed/manga109_detection_v2_margin10/pages.json --output results/a_data_audit_current.json
python -m scripts.review_detection_samples --manifest data/processed/manga109_detection_v2_margin10/manifest_candidate.json --output data/processed/manga109_detection_v2_margin10/decisions.csv --label all
```

第二条打开人工复核窗口：A接受、R剔除、N跳过、P上一张、Q保存退出。结合原图上下文判断，继续运行会接着未复核项。不要把初始needs_review或AI抽查当作人工已接受。

完成实际人工复核后才运行：

```powershell
python -m scripts.apply_sample_review --manifest data/processed/manga109_detection_v2_margin10/manifest_candidate.json --review data/processed/manga109_detection_v2_margin10/decisions.csv --output data/processed/manga109_detection_v2_margin10/manifest_reviewed.json
```

只导出明确accepted项，保持原标签和split。部分复核会产生子集，不自动成为最终完整数据集。

## 与B、C对接

B可用漫画候选或联合候选，通过load_manifest和detection_samples读取；整页检测评估使用pages.json及 [INTERFACES.md](INTERFACES.md) 中的预测格式。尚未生成真实P/R/F1。

C的256张交付均已保留。253张通过格式和可见点边界检查，原划分190/25/38，与漫画候选形成6660条联合记录；3张图4个可见点越界保留原样。原JSON、点序、坐标和split没有修改，详见 [C_DATA_INTEGRATION.md](C_DATA_INTEGRATION.md)。

联合清单：`data/processed/joint_a_c_v1/joint_manifest_candidate.json`；只用关键点：同目录 `landmarks_validated.json`。C的bbox是整张头像图范围，不作为复杂场景完整检测真值。后续扩充AnimeFace须遵守同目录划分注册表的重复组锁定和近似候选保留规则。

## 通道和交付

六张图的最新彩色原图与11通道在 `results/channel_experiment_v2_color/`，分析见 [CHANNEL_EXPERIMENT.md](CHANNEL_EXPERIMENT.md)。训练时调用compute_11_channels，无需预存每张训练图的十一通道PNG。

最新本地交接包为 `results/a_delivery_latest/`，不含完整原始漫画库或真实模型。数据、标注、结果不随Git推送，队友需要另行获取；旧版本位置和恢复方式见A_CURRENT_FILES。
