"""A: score-ordered one-to-one matching; call per image before aggregation."""
import numpy as np


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    union = np.prod(np.maximum(0, a[2:]-a[:2])) + np.prod(np.maximum(0, b[2:]-b[:2])) - area
    return float(area / union) if union > 0 else 0.0


def evaluate_detection(predictions, ground_truth, threshold=0.5):
    matched = set()
    for prediction in sorted(predictions, key=lambda p: p['score'], reverse=True):
        candidates = [(iou(prediction['bbox'], box), j) for j, box in enumerate(ground_truth) if j not in matched]
        if candidates:
            overlap, j = max(candidates)
            if overlap >= threshold:
                matched.add(j)
    tp, fp, fn = len(matched), len(predictions)-len(matched), len(ground_truth)-len(matched)
    p, r = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0)
