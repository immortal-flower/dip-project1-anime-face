# dip-project1-anime-face

数字图像处理项目一：基于特征工程的动漫人脸检测与 28 点关键点回归。

三位同学平等协作，先运行共同基础，再各自完善模块。已准备首批 Manga109 样本并完成六张图的通道观察；**模型完整流程仍仅有合成样例验证，尚未验证真实动漫数据检测与回归效果。**

## 阅读入口

- **[A 当前文件入口](docs/A_CURRENT_FILES.md)**：最新版数据、结果、交接包和旧产物归档位置。

- [A 本轮工作记录](docs/A_WORK_LOG.md)：新增功能、发现的问题、数据版本和复现命令。
- [C 标注接入与联合数据](docs/C_DATA_INTEGRATION.md)：256张交付、3个边界例、6660条联合候选和划分注册表。
- [A 报告初稿](docs/A_REPORT.md)：数据方法、特征原理、真实观察与实验边界。
- [A 对照老师要求的遗漏核查](docs/A_REQUIREMENTS_AUDIT.md)：区分已经验证、尚未收尾和可选实验。
- [A 从哪里开始读](docs/A_READING_GUIDE.md)：先看结果，再顺着样本、划分和通道读中文注释。
- [十一通道观察与浮点对照](docs/CHANNEL_EXPERIMENT.md)：六张实际图像的观察、数值结果和复现方式。
- [A 部分：数据位置、负样本与划分](docs/A_DATA_PREPARATION.md)：本地 Manga109/AnimeFace、首批检测样本及 C 标注对接。
- [三人工作流程与 A / B / C 对接说明](docs/TEAM_WORKFLOW.md)：一起讨论项目顺序、数据依赖与阶段成果。
- [给 Codex 的协作提示词与分工](docs/CODEX_COLLABORATION.md)
- [接口约定](docs/INTERFACES.md)
- [当前进度与限制](docs/STATUS.md)
- 老师要求：[PDF](docs/项目一%20基于特征工程的动漫人脸检测与关键点回归.pdf)、[详细说明](docs/项目一_基于特征工程的动漫人脸检测与关键点回归_详细说明.md)。原文保持不变。

## 克隆与环境

这是公开仓库，无需邀请即可查看和克隆。直接推送需要协作者写入权限；没有权限可以 Fork 后提交 Pull Request。

```bash
git clone https://github.com/immortal-flower/dip-project1-anime-face.git
cd dip-project1-anime-face
uv venv .venv --python 3.10
uv pip install -r requirements.txt
```

2026-09-26 已在独立 Python 3.10.21 环境验证；如需匹配本轮 NumPy/OpenCV 版本，可改用 `uv pip install -r requirements-lock.txt`。本机 uv 位于 `results/tools/uv/bin/uv.exe`，环境记录在 `results/environment_a/`；这些本地目录不会随 Git 克隆。

需要先安装 uv。Windows PowerShell 激活 `.venv\Scripts\Activate.ps1`；macOS/Linux 使用 `source .venv/bin/activate`。以下命令都在仓库根目录运行。

## 可选：检查共享接口

日常A工作直接从上面的最新文件入口开始。以下合成示例用于排查接口问题，旧输出已归档，运行时会重新生成；共享脚本继续保留。

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke_test
python demo.py --image results/smoke/images/test_1_0.png --model-dir results/smoke/models --output results/smoke/demo.png
python -m scripts.visualize_channels --image results/smoke/images/test_1_0.png --output results/smoke/channels.png
```

smoke_test 自动生成微型合成数据，训练三级 Cascade 和三级形状回归，检查保存后重新加载结果一致。图片、模型与 JSON 在 `results/smoke/` 中，再次执行会更新该目录。配置和 JSON 标记 `synthetic: true`。

**合成图案和椭圆上的 28 点没有真实五官语义，不能用于课程效果报告。** 这只是接口连接检查，不替代真实数据训练和人工标注。

准备好符合接口约定的真实数据清单后：

```bash
python -m scripts.train_baseline --manifest data/manifest.json --output models/baseline
python demo.py --image data/example.jpg --model-dir models/baseline --output results/example.png
```

基础训练入口使用 train 训练、val 校准阈值；test 不参与。它不负责下载数据、人工修正或正式测试集评估。

## 实际代码框架

| 模块 | 文件与职责 |
|---|---|
| A | `src/channels11.py`：整数通道；`src/data_io.py`：图像和清单；`src/detection_metrics.py`：检测指标 |
| B | `src/depth2_tree.py`、`adaboost.py`、`cascade.py`：检测训练；`pyramid.py`、`sliding_window.py`、`grouping.py`：搜索与 NMS |
| C | `src/shape_regression.py`：多级回归；`landmark_metrics.py`：NME；`detector.py`：统一接口；`demo.py`：结果图与 JSON |
| A 的数据入口 | `scripts/prepare_manga_detection_data.py`：采样和划分；`audit_detection_data.py`：一致性检查；`review_detection_samples.py` / `apply_sample_review.py`：人工复核和导出 |
| A 的实验入口 | `scripts/prepare_channel_examples.py`：准备观察图；`channel_experiment.py`：批量通道与浮点对照；`visualize_channels.py`：单图拼图；`evaluate_detection_results.py`：汇总 B 的整页检测结果 |
| 共享入口 | `scripts/train_baseline.py`：训练两个模型；`scripts/smoke_test.py`：合成端到端检查 |
| 测试 | `tests/test_contracts.py`：基础接口；`test_data_preparation.py`：采样与划分；`test_a_workflow.py`：通道对照、整集指标与复核导出 |
| 协作文件 | `AGENTS.md`：Codex 阅读入口；`docs/`：老师要求、协作提示词、接口和进度 |

基础模型导出为 detector.json、landmark.npz、config.json、splits.json。真实点序编号、数据说明和课程材料还需完善。

模型现在还会导出 `feature_definition.json`，并在配置中记录通道版本和校验指纹。A 当前候选清单为 `data/processed/manga109_detection_v2_margin10/manifest_candidate.json`，尚未全量验收，不能直接作为最终数据发布；原版保留用于溯源。新工具及本地阶段包使用说明见 A_WORK_LOG。

C交付已经接入；当前联合入口为 `data/processed/joint_a_c_v1/joint_manifest_candidate.json`。只做关键点模块可读取同目录 `landmarks_validated.json`。这两个文件仍是本地数据，不会随Git下载；所有原C标注均保留，3张可见点越界例另列待确认。

## 日常协作

开始前保护本地修改并同步主分支，每人使用独立分支，通过 Pull Request 合并。README 随代码更新，STATUS 区分实现、合成验证和真实实验。保存文件不会自动提交或推送。

本地环境、密钥、data/、models/、results/ 不上传到仓库。真实数据和大型模型另行约定共享位置，在仓库记录获取方式；课程最终提交仍需包含老师要求的数据集和模型。
