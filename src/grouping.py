from .detection_metrics import iou


def nms(predictions, threshold=0.3):
    kept = []
    for item in sorted(predictions, key=lambda p: p['score'], reverse=True):
        if all(iou(item['bbox'], prev['bbox']) <= threshold for prev in kept):
            kept.append(item)
    return kept
