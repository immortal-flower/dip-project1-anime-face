# dip-project1-anime-face

数字图像处理项目一：基于特征工程的动漫人脸检测与 28 点关键点回归。

三位同学平等协作，先运行共同基础，再各自完善模块。**当前只有合成样例流程验证，尚未验证真实动漫数据效果。**

## 阅读入口

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

需要先安装 uv。Windows PowerShell 激活 `.venv\Scripts\Activate.ps1`；macOS/Linux 使用 `source .venv/bin/activate`。以下命令都在仓库根目录运行。

## 先检查完整流程

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
| A 的入口 | `scripts/visualize_channels.py`：原图与 11 通道拼图 |
| 共享入口 | `scripts/train_baseline.py`：训练两个模型；`scripts/smoke_test.py`：合成端到端检查 |
| 共享测试 | `tests/test_contracts.py`：通道、树、指标和坐标接口验证 |
| 协作文件 | `AGENTS.md`：Codex 阅读入口；`docs/`：老师要求、协作提示词、接口和进度 |

基础模型导出为 detector.json、landmark.npz、config.json、splits.json。真实点序编号、数据说明和课程材料还需完善。

## 日常协作

开始前保护本地修改并同步主分支，每人使用独立分支，通过 Pull Request 合并。README 随代码更新，STATUS 区分实现、合成验证和真实实验。保存文件不会自动提交或推送。

本地环境、密钥、data/、models/、results/ 不上传到仓库。真实数据和大型模型另行约定共享位置，在仓库记录获取方式；课程最终提交仍需包含老师要求的数据集和模型。
