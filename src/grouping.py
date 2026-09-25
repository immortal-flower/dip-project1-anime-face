# B 的基础版：按得分保留较可靠的框，抑制与其高度重叠的重复框。
from .detection_metrics import iou


# 按 score 从高到低处理，仅保留与已保留框重叠不太大的候选。
def nms(predictions, threshold=0.3):
    kept = []
    for item in sorted(predictions, key=lambda p: p['score'], reverse=True):
        if all(iou(item['bbox'], prev['bbox']) <= threshold for prev in kept):
            kept.append(item)
    return kept
