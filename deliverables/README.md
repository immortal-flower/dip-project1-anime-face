# B 冻结模型下载包

`b-final-model.zip` 是 `results/b-final/model/` 的完整可运行副本，用于把 B 的冻结检测器交给 C。仓库仍继续忽略本地 `results/` 和 `models/`；只有这个经过检查的小型交付包被纳入版本控制。

## 下载后恢复

在仓库根目录执行：

```powershell
New-Item -ItemType Directory -Force results\b-final | Out-Null
Expand-Archive -LiteralPath deliverables\b-final-model.zip -DestinationPath results\b-final -Force
```

压缩包自带顶层 `model/` 目录，恢复后的路径为 `results/b-final/model/`。

## 包内文件

```text
model/
├── B_MODEL_MANIFEST.json
├── config.json
├── detector.json
├── feature_definition.json
├── landmark.npz
└── splits.json
```

其中 `detector.json` 才包含 Cascade 的弱树参数。`landmark.npz` 是现有联合包中的关键点基线，B 没有验证其真实 NME；C 应按照 `docs/B_TO_C_HANDOFF.md` 在新的组合目录中替换为自己训练并验证过的最终关键点模型。

## 完整性校验

压缩包 SHA256：

```text
8000810243b1b7be69a27d6ca201a803ea87416becf6d2428f38e4885babefb1
```

Windows PowerShell：

```powershell
(Get-FileHash -Algorithm SHA256 deliverables\b-final-model.zip).Hash.ToLowerInvariant()
```

包内五个模型文件的逐文件 SHA256 保存在 `model/B_MODEL_MANIFEST.json`。交付前已从 ZIP 数据流重新计算，五项全部一致。

压缩包不含 Manga109 原页、训练裁剪图或其他大型数据集文件。
