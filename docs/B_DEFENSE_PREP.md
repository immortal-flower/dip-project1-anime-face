# B 模块答辩前数字冻结与证据索引

> 冻结日期：2026-09-29。Manga109 的 108 页和 AnimeFace 的 38 张 test 均已暴露，本文只汇总冻结结果；新增 step/scale 实验只使用 1 张预先固定的 validation 页。除 `integration/b-c` 的正式 C 汇总外，所有数字均可追溯到 `results/defense-prep/frozen/DEFENSE_NUMBERS.json` 及其中列出的原始 result。

## 答辩推荐使用数字

1. 检测输入窗口：24×24。
2. 通道数：11。
3. 候选像素差特征：1024。
4. 原冻结模型：3 个 Stage，弱树数 [19, 20, 19]。
5. V2：5 个 Stage，弱树数 [19, 20, 19, 13, 14]；前三级完全冻结。
6. Stage 0 在 108 页 test 上早拒绝 68.90% 的窗口。
7. 原模型 Manga109-108：P/R/F1=6.71%/4.83%/5.61%。
8. V2 Manga109-108：P/R/F1=8.53%/17.28%/11.43%。
9. 原→V2 Recall：4.83%→17.28%，约 3.58 倍。
10. 前两轮 hard negatives：接受 215 个。
11. V2 新增 train hard negatives：接受 287 个；累计 train 接受 502 个。
12. V2 val hard negatives：88 个，只用于选择/校准；test 为 0。
13. C 最终 ensemble 在 oracle bbox 下 mean NME=0.0864（C 的 interocular/必要时 bbox diagonal 协议）。
14. 旧 B + C 正式端到端成功：2/38=5.26%。

这些 Manga109 TP/FP/FN 是冻结预测与真值的 IoU≥0.5 自动一对一匹配，尚未完成逐页用户人工批准；答辩时应称为“冻结自动统计”。

## 绝对不能混用的数字

- old 3-stage B detector and V2 5-stage detector。
- Manga109 full-page detection and AnimeFace cropped-face detection。
- AnimeFace full-image bbox and user-approved local-face bbox。
- C interocular/bbox-diagonal-fallback NME and B helper sqrt(box-area) NME。
- C final ensemble landmark model and B package landmark.npz。
- automatic IoU metrics and any future human-reviewed metrics。

## 1. 课程要求的 step / scale 小型实验

固定 validation 页：`manga109:HanzaiKousyouninMinegishiEitarou:061`。选择规则是现有 `source-spread` 顺序的第一张 val 页，先冻结后运行；没有依据效果挑页。模型、阈值、weighted NMS、min support=2、框校准和 IoU 都保持 V2 不变。

| 对照 | step | scale | windows | TP | FP | FN | Precision | Recall | F1 | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step | 1 | 1.2 | 5954996 | 7 | 79 | 9 | 0.0814 | 0.4375 | 0.1373 | 965.23 |
| step | 2 | 1.2 | 1491184 | 6 | 97 | 10 | 0.0583 | 0.3750 | 0.1008 | 263.34 |
| scale | 2 | 1.1 | 2617173 | 6 | 89 | 10 | 0.0632 | 0.3750 | 0.1081 | 569.16 |
| scale | 2 | 1.2 | 1491184 | 6 | 97 | 10 | 0.0583 | 0.3750 | 0.1008 | 263.34 |
| scale | 2 | 1.3 | 1118885 | 4 | 110 | 12 | 0.0351 | 0.2500 | 0.0615 | 370.03 |

在 scale=1.2 的这一页上，step 1 的窗口数是 step 2 的 3.99 倍，耗时是 3.67 倍。scale=1.1/1.2/1.3 的召回变化只反映这一固定页，不能外推为总体性能结论。每级 evaluated/passed/rejected、raw candidates 和 NMS 后数量保存在 `results/defense-prep/step-scale/summary.json`。

## 2. 六个困难负样本图例

| 编号 | 类型 | page_id | round | score | 为什么容易误检 |
|---|---|---|---:|---:|---|
| hn_01 | dialogue_text | `manga109:AisazuNihaIrarenai:040` | 1 | 11.55 | 对白框中的字符形成眼睛或眉毛般的局部黑白梯度。 |
| hn_02 | high_contrast_hair | `manga109:AkkeraKanjinchou:051` | 2 | 12.02 | 成束高对比斜线与动漫刘海、眼周边缘的局部结构相似。 |
| hn_03 | clothing_texture | `manga109:AisazuNihaIrarenai:040` | 1 | 12.06 | 衣服条纹与轮廓在小窗口内形成近似五官的明暗组合。 |
| hn_04 | dense_manga_lines | `manga109:AkkeraKanjinchou:051` | 2 | 11.51 | 密集竖线和斜边产生大量强梯度，触发局部像素差弱分类器。 |
| hn_05 | eye_like_local_structure | `manga109:Arisa:020` | 3 | 16.08 | 衣领扣件的双圆形结构在低分辨率下呈现类似双眼的布局。 |
| hn_06 | building_object_outline | `manga109:AkkeraKanjinchou:051` | 2 | 10.97 | 屋顶和物体交界形成封闭轮廓及中心暗区，外观近似脸部布局。 |

六项的源清单均为 `review_status=accepted`；被复核剔除的疑似完整脸没有进入图例。拼图：`results/defense-prep/hard-negatives/hard_negatives_sheet.png`。

## 3. 冻结检测案例

| 类别 | page_id | TP/FP/FN | 选择规则 |
|---|---|---:|---|
| success | `manga109:UltraEleven:036` | 9/32/14 | V2 frozen test page with the highest TP count; selected match has the highest IoU. |
| success | `manga109:DollGun:025` | 8/50/16 | V2 frozen test page with the second-highest TP count; selected match has the highest IoU. |
| fp | `manga109:YoumaKourin:099` | 1/54/14 | V2 frozen test page with the largest FP count; selected FP has the highest score. |
| fp | `manga109:Joouari:001` | 0/37/4 | High-FP page with zero TP; selected FP has the highest score. |
| fn | `manga109:MoeruOnisan_vol01:068` | 7/35/33 | V2 frozen test page with the largest FN count; selected FN is the smallest unmatched GT. |
| fn | `manga109:TetsuSan:028` | 1/29/29 | Another high-FN page from a different source; selected FN is the smallest unmatched GT. |

案例全部来自已经冻结的 V2 Manga109 108 页 test JSON，只用于解释，不据此修改阈值、NMS、step、scale 或模型。成功页仍同时展示 FP/FN，避免只报最好看的局部。

## 4. 冻结数字 A：原 B 模型，Manga109 108 test

- TP/FP/FN=69/960/1360；P/R/F1=0.0670554/0.0482855/0.0561432。
- 窗口 13514148；总耗时 1774.11s；每页 16.43s。
- step=6，scale=1.3，threshold=13.8538650924998，NMS IoU=0.3。
- raw candidates=3215310，NMS 后=33138，阈值后=1029。
- Stage 0/1/2 evaluated/passed/rejected：13514148/4202378/9311770; 4202378/3637752/564626; 3637752/3215310/422442。
- 评价输入 detector/config SHA256：`9914cc517c23d1e5f9e208403917dac0cf2e40812fc5713dcd2cb283948d8486` / `51388ca000cbe7234cbc7e1ff7a83e79b600acf89078a5094b1033465fd385b9`。交付包 config 后续写入部署阈值，因此包内 config SHA256 为 `1a117ad6416c35d05157f07808a98ee87a0280de6e1ecb9b1dc0d11e51b9940e`，两者不可假设相同。

## 5. 冻结数字 B：V2，Manga109 108 test

- TP/FP/FN=247/2647/1182；P/R/F1=0.0853490/0.1728481/0.1142725。
- 5 stages；threshold=14.724508431559716；weighted NMS；min support=2。
- box calibration：`{"dx": 6.086241666486392e-05, "dy": 0.06027198164383903, "scale_x": 0.9611113619164823, "scale_y": 1.0247802449039334}`。
- 模型 ZIP SHA256：`1e15e385dbdecd6b5e2ce0ec4bb9eb94b86f5c87d42ffa288ead268fc7abba0f`。
- 数据边界：train hard negatives 拟合；val 只选择阈值/后处理/框校准；test hard negatives=0，冻结后只评价一次。

## 6. 冻结数字 C：hard-negative 真实正负结果

- Round 1：mined/accepted/rejected=120/109/11。
- Round 2：mined/accepted/rejected=120/106/14。
- validation negative cumulative FPR：512 baseline=0.4111；1024 baseline=0.3333；1024 + HN weight 3=0.3694。
- 困难负样本并非每轮都改善 validation；权重 3 模型仍差于普通 1024 模型，所以旧冻结模型保留普通 1024 版本。该负结果不得删除。

## 7. 冻结数字 D：B+C / AnimeFace 两种协议

### Protocol 1：正式 integration/b-c

- 旧 B detector + C 最终 50/50 ensemble；原 bbox 是 AnimeFace 裁剪整图。TP/FP/FN=3/7/35，P/R/F1=0.3000/0.0789/0.1250。
- oracle mean NME=0.0864；matched detected mean NME=0.1281；端到端成功=2/38=5.26%。
- NME 优先使用双眼中心距，眼点不足时回退 bbox diagonal。

### Protocol 2：feature/detection 的用户批准局部脸框

- 原 B：TP/FP/FN=6/4/32，P/R/F1=0.6000/0.1579/0.2500。
- V2：TP/FP/FN=3/22/35，P/R/F1=0.1200/0.0789/0.0952。
- `scripts/evaluate_c_end_to_end.py` 使用 mean visible error / sqrt(local box area)，且 B 包的 `landmark.npz` 不是 C 最终 ensemble；所以 Protocol 2 的 NME 不能与 Protocol 1 横向比较。

## 8. 复现与证据

- step/scale 命令：`results/defense-prep/step-scale/commands.txt`。
- 完整本地冻结数字：`results/defense-prep/frozen/DEFENSE_NUMBERS.json`。
- 六个 hard negatives：`results/defense-prep/hard-negatives/metadata.json`。
- 六个检测案例：`results/defense-prep/detection-cases/metadata.json`。
- 汇总脚本：`python -m scripts.prepare_b_defense`。

## 9. 剩余风险

- step/scale 只有 1 张固定 validation 页，仅满足小型验收和复杂度趋势展示，不能声称总体提升。
- Manga109 108 页和两套 C-38 的 TP/FP/FN 目前是自动 IoU 匹配；用户逐页人工复核尚未完成。
- V2 提升 Manga109 recall，但在局部框 C-38 上退化，说明仍有明显跨域和框语义问题。
- test 已暴露，不能继续据其结果调整任何参数。
