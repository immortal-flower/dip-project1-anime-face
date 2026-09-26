# A 当前文件入口与整理记录

更新：2026-09-26。日常使用以下版本；数据和结果仅在本地，GitHub 只包含代码、配置和说明。

## 现在使用什么

| 用途 | 当前路径（相对仓库根目录） |
|---|---|
| 漫画检测候选：2522正例、3885负例 | `data/processed/manga109_detection_v2_margin10/manifest_candidate.json` |
| 完整裁剪记录及固定划分 | 同目录 `manifest.json`、`book_splits.json` |
| 整页检测评价真值 | 同目录 `pages.json` |
| C 的全部原始交付 | `data/processed/animeface/`，保持不变 |
| C 的253条格式／边界检查通过记录 | `data/processed/joint_a_c_v1/landmarks_validated.json` |
| 联合候选：6660条 | 同目录 `joint_manifest_candidate.json` |
| 全部256张C图片及导入记录（含3张待确认图） | 同目录 `images/`、`landmarks_imported.json` |
| 六张观察图输入 | `data/channel_examples_v1/`（这是当前输入，没有被新版输入取代） |
| 最新十一通道结果 | `results/channel_experiment_v2_color/index.md` |
| 样本质量复核 | `results/quality_review_v2/` |
| 近似重复候选 | `results/near_duplicates_manga_v2/`、`results/near_duplicates_anime_v1/` |
| C接入检查 | `results/c_integration/` |
| 环境验证证据 | `results/environment_a/` |
| 最新本地交接包 | `results/a_delivery_latest/START_HERE.md` |

文件名中的 v1 不一定过时：联合数据、AnimeFace近似检查和观察图输入目前只有这一版。只归档已经被替代的内容，不按名字批量删除。

## 整理了什么

旧产物移到仓库旁边的 `../A历史归档/20260926/`，归档中保留原相对目录；具体清单和文件校验见归档内 `archive_manifest.json`。它不属于Git仓库，也不会进入新交接包。

- 旧漫画裁剪 `data/processed/manga109_detection_v1/`。
- 旧灰度首格通道结果 `results/channel_experiment_v1/`。
- 两个旧交接包 `results/a_delivery_20260926/`、`results/a_delivery_with_c_20260926/`。
- 三次合成演示输出 `results/smoke/`、`results/smoke_after_a_updates/`、`results/smoke_uv_environment/`。
- 已被新版记录取代的单图／检查文件：`a_data_audit_v1.json`、`animeface_first_channels.png`、`low_texture_review_v1.png`。

原始数据、C的三张问题图、最新版的完整清单和被隔离样本都保留。`manifest_candidate.json` 的筛选不等于删除原始材料。

旧目录恢复时，应按归档清单移回原相对路径，且先确认目标不存在；旧清单中的相对路径以原位置为准，不能在归档位置直接训练。A_WORK_LOG中旧命令是历史记录，日常以本页和A_DATA_PREPARATION为准。

## 为什么保留一些基础脚本

`scripts/smoke_test.py`、`scripts/train_baseline.py`、`demo.py` 是共享接口验证／训练／演示入口，仍被团队使用，并非可随意删除的废弃版本。本轮不改动它们，也不改动B和C的实现。合成演示输出可按需重新生成。

A的原始数据生成、扩边迁移、质量复核脚本也保留：它们分别承担不同步骤，是复现当前数据的工具，不是重复模型版本。交接导出工具只复制选定的最新通道结果。

## 当前限制与验证

当前数据仍是候选集：漫画质量及疑似重复关系尚待确认，C的3张图4个可见点越界仍保留原样；真实训练和最终指标由相应模块后续完成。

整理验证包括：旧产物移动前后文件SHA256一致、当前6660条联合图片可读、654个整页原图路径存在、C原JSON未变化、B/C和共享代码与整理前提交 `43d02c5` 相同、19项测试通过。交接包另有逐文件 `SHA256SUMS.json`。

用户随后明确要求合入 `main`。已检查远程状态并将A分支快进合入本地main，本轮发布目标相应更新为 `origin/main`；采用普通推送，不强制覆盖历史，不改动队友分支。更新发布后队友直接同步main即可。图片、标注和交接包仍需单独共享，克隆仓库不会自动获得它们。
