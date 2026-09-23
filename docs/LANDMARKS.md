# 28点编号与人工修正规则

正式点序名称为 `hysts28-v1`，与 `anime-face-detector` 的 HRNetV2 教师模型输出完全一致。编号从0开始，左右均指图像中的左右。训练清单、模型配置、评价结果和Demo JSON都必须记录此名称，禁止与旧的 `course28-v1` 混用。

| 编号 | 区域与顺序 |
|---|---|
| 0–4 | 脸部轮廓：图像左侧上缘、左下颌、下巴、右下颌、右侧上缘 |
| 5–7 | 图像左眉，从图像左侧到右侧 |
| 8–10 | 图像右眉，从图像左侧到右侧 |
| 11–16 | 图像左眼，严格按编号图位置 |
| 17–22 | 图像右眼，严格按编号图位置 |
| 23 | 鼻尖 |
| 24–27 | 嘴部：左嘴角、上唇中心、右嘴角、下唇中心 |

水平翻转对应点为：`0↔4, 1↔3, 5↔10, 6↔9, 7↔8, 11↔19, 12↔18, 13↔17, 14↔22, 15↔21, 16↔20, 24↔26`。2、23、25、27位于中部，不交换。

编号示意图见 `landmark_numbering.png`。它只定义语义和顺序，不是训练图片。

## 推荐标注流程

1. 用 `scripts/prelabel_animeface.py` 生成HRNetV2预标注。预标注必须保留 `annotation_status=model_prelabel_unreviewed`。
2. 运行下面的修正工具。左键拖动最近的点，右键切换可见性；Enter接受当前图片，P返回，U撤销，S保存，Q保存并退出。

```powershell
python -m scripts.correct_landmarks `
  --input data/landmarks/prelabels/manifest.json `
  --output data/landmarks/corrected/manifest.json
```

按Enter接受后，记录才会标记为 `annotation_status=human_reviewed` 和 `reviewed_landmarks=true`。测试集必须逐张人工确认；未经确认的教师输出不能当作真值NME。

左键点代表可见点。点被头发或画面边界遮挡时，位置可保留为估计值，但需右键标成不可见；训练损失和NME只统计 `visibility=1` 的点。

完全从头标注整图时可运行 `python -m scripts.annotate --images ... --output ...`。该工具也使用 `hysts28-v1`。同一项目中不再使用旧的八轮廓点自定义顺序。
