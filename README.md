# dip-project1-anime-face

数字图像处理项目一：基于特征工程的动漫人脸检测与 28 点关键点回归。

三位同学平等协作，先运行共同基础，再各自完善模块。公共检测基础仍以合成样例验证为主；C模块已在真实AnimeFace裁剪图上完成教师预标注和临时回归训练，但未经人工修正的伪标签不能作为正式准确率。

## C模块：28点关键点流程

正式点序采用与HRNetV2教师一致的 `hysts28-v1`，见 [编号和修正规则](docs/LANDMARKS.md)。教师预标注不是真值，必须保留未审核状态；测试集只有人工逐张确认后才能报告NME。

```powershell
# 1. GPU预标注（需先按 requirements-prelabel.txt 准备教师环境）
python -m scripts.prelabel_animeface --images data/raw/anime_faces/images `
  --detector-source third_party/anime-face-detector `
  --landmark-model models/pretrained/anime-face-detector-hrnetv2.safetensors `
  --output data/landmarks/prelabels --device cuda:0 --limit 256

# 2. 人工拖动修正；支持中断后继续
python -m scripts.correct_landmarks --input data/landmarks/prelabels/manifest.json `
  --output data/landmarks/corrected/manifest.json

# 3. 单独训练和评价C模块
python -m scripts.train_landmark --manifest data/landmarks/corrected/manifest.json `
  --output models/landmark
python -m scripts.evaluate_landmark --manifest data/landmarks/corrected/manifest.json `
  --model models/landmark --output results/landmark-test

# 4. 暂时没有B检测模型时，可传入一个或多个框演示
python demo.py --image example.jpg --model-dir models/landmark `
  --boxes-json example-boxes.json --output results/example.jpg
```

当前256张临时数据的验证集选择结果为每点2对像素差、三级更新、Ridge=10，已设为训练入口默认值；人工修正数据扩大后应重新选择。评价默认使用双眼中心距离；眼部可见点不足时回退到真值框对角线，结果文件会统计两种方法各自使用次数。`--allow-unreviewed` 只用于检查软件流程，不会把教师伪标签标成真实测试结果。

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
| C | `src/landmark_schema.py`：固定点序；`shape_regression.py`：多级回归；`landmark_metrics.py`：双眼/框归一化NME；`detector.py`：统一接口；`demo.py`：结果图与 JSON |
| A 的入口 | `scripts/visualize_channels.py`：原图与 11 通道拼图 |
| C 的入口 | `scripts/prelabel_animeface.py`、`correct_landmarks.py`、`train_landmark.py`、`evaluate_landmark.py` |
| 共享入口 | `scripts/train_baseline.py`：训练两个模型；`scripts/smoke_test.py`：合成端到端检查 |
| 共享测试 | `tests/test_contracts.py`：通道、树、指标和坐标接口验证 |
| 协作文件 | `AGENTS.md`：Codex 阅读入口；`docs/`：老师要求、协作提示词、接口和进度 |

基础模型导出为 detector.json、landmark.npz、config.json、splits.json。真实点序编号、数据说明和课程材料还需完善。

## 日常协作

开始前保护本地修改并同步主分支，每人使用独立分支，通过 Pull Request 合并。README 随代码更新，STATUS 区分实现、合成验证和真实实验。保存文件不会自动提交或推送。

本地环境、密钥、data/、models/、results/ 不上传到仓库。真实数据和大型模型另行约定共享位置，在仓库记录获取方式；课程最终提交仍需包含老师要求的数据集和模型。
