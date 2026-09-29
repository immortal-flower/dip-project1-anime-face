"""Detection grouping: hard NMS and score-weighted box voting."""
import numpy as np

from .detection_metrics import iou


# 按 score 从高到低处理，仅保留与已保留框重叠不太大的候选。
def nms(predictions, threshold=0.3):
    kept = []
    for item in sorted(predictions, key=lambda p: p['score'], reverse=True):
        if all(iou(item['bbox'], prev['bbox']) <= threshold for prev in kept):
            kept.append(item)
    return kept


def weighted_nms(predictions, threshold=0.3, min_support=1):
    """Merge overlapping candidates and expose cluster support for verification.

    The highest-score candidate seeds a cluster.  Coordinates are averaged with
    positive score-derived weights while the peak score is retained, so score
    calibration remains compatible with hard NMS.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('NMS threshold must be in [0, 1]')
    if not isinstance(min_support, int) or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    remaining = sorted(predictions, key=lambda p: p['score'], reverse=True)
    kept = []
    while remaining:
        seed = remaining[0]
        cluster = [item for item in remaining
                   if iou(seed['bbox'], item['bbox']) > threshold]
        cluster_ids = {id(item) for item in cluster}
        remaining = [item for item in remaining if id(item) not in cluster_ids]
        if len(cluster) < min_support:
            continue
        scores = np.asarray([item['score'] for item in cluster], dtype=float)
        # Softplus-like shift keeps every finite candidate weight positive.
        weights = np.maximum(scores - scores.min() + 1e-6, 1e-6)
        boxes = np.asarray([item['bbox'] for item in cluster], dtype=float)
        merged = dict(seed)
        merged['bbox'] = np.average(boxes, axis=0, weights=weights).tolist()
        merged['score'] = float(scores.max())
        merged['support'] = len(cluster)
        merged['mean_score'] = float(scores.mean())
        kept.append(merged)
    return kept


def calibrate_box(box, calibration, image_shape=None):
    """Apply a global center/size correction learned without test data."""
    x1, y1, x2, y2 = map(float, box)
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx += float(calibration.get('dx', 0.0)) * width
    cy += float(calibration.get('dy', 0.0)) * height
    width *= float(calibration.get('scale_x', 1.0))
    height *= float(calibration.get('scale_y', 1.0))
    result = [cx - width / 2, cy - height / 2,
              cx + width / 2, cy + height / 2]
    if image_shape is not None:
        image_height, image_width = image_shape[:2]
        result = [max(0.0, min(result[0], image_width - 1.0)),
                  max(0.0, min(result[1], image_height - 1.0)),
                  max(1.0, min(result[2], float(image_width))),
                  max(1.0, min(result[3], float(image_height)))]
    return result
