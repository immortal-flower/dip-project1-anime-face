# A：比较预测框和真实框，统计检出、误检和漏检；每张图独立匹配。
"""A: score-ordered one-to-one matching; call per image before aggregation."""
import numpy as np


# 计算两个人脸框的交并比：相交面积除以合并面积，范围为 0～1。
def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    union = np.prod(np.maximum(0, a[2:]-a[:2])) + np.prod(np.maximum(0, b[2:]-b[:2])) - area
    return float(area / union) if union > 0 else 0.0


# 在一张图内按得分依次匹配真值；同一个真值最多算一次检出。
def evaluate_detection(predictions, ground_truth, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    # 拒绝坏坐标或非有限分数，避免静默污染整个测试集的统计。
    for box in [p['bbox'] for p in predictions] + list(ground_truth):
        value = np.asarray(box, dtype=float)
        if value.shape != (4,) or not np.isfinite(value).all() or np.any(value[2:] <= value[:2]):
            raise ValueError('Expected finite xyxy boxes with positive area')
    if any(not np.isfinite(p['score']) for p in predictions):
        raise ValueError('Detection scores must be finite')
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


def evaluate_dataset(pages, predictions, threshold=0.5):
    """逐图匹配后累计 TP/FP/FN，再计算整个集合的 P/R/F1。

    predictions 是 {page_id: [{bbox, score}, ...]}；缺失页按无检测计入漏检，
    并在结果里明确列出。禁止把不同图片的框放在一起匹配。
    """
    identifiers = [page['page_id'] for page in pages]
    if len(identifiers)!=len(set(identifiers)) or not pages:
        raise ValueError('Expected nonempty pages with unique page_id')
    unknown = set(predictions)-set(identifiers)
    if unknown:
        raise ValueError(f'Predictions refer to unknown evaluation pages: {sorted(unknown)[:3]}')
    individual = {page['page_id']:evaluate_detection(predictions.get(page['page_id'],[]),page['bboxes'],threshold)
                  for page in pages}
    tp,fp,fn = [sum(row[key] for row in individual.values()) for key in ('tp','fp','fn')]
    precision,recall = tp/max(tp+fp,1),tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                page_count=len(pages),missing_prediction_pages=sorted(set(identifiers)-set(predictions)),
                iou_threshold=threshold,per_page=individual)
