# C 的共同基础版：只评价可见点，由调用方明确提供归一化距离。
import numpy as np

from .landmark_schema import LEFT_EYE, RIGHT_EYE, validate_landmarks


# 计算可见点平均定位误差，再除以归一化距离；无可见点时返回 None。
def nme(prediction, truth, visibility, normalizer):
    """Caller specifies eye-center distance or explicitly documented bbox diagonal."""
    p, t = np.asarray(prediction), np.asarray(truth)
    mask = np.asarray(visibility, dtype=bool)
    if p.shape != (28, 2) or t.shape != (28, 2) or mask.shape != (28,) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Expected finite 28x2 points and 28 visibility flags')
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('Normalizer must be finite and positive')
    if not mask.any():
        return None
    return float(np.linalg.norm(p[mask]-t[mask], axis=1).mean()/normalizer)


def bbox_diagonal(bbox):
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError('Expected one finite xyxy bbox')
    width, height = bbox[2:] - bbox[:2]
    if width <= 0 or height <= 0:
        raise ValueError('bbox must have positive width and height')
    return float(np.hypot(width, height))


def interocular_distance(truth, visibility, min_visible_per_eye=2):
    """Distance between visible-point means of the two eye regions.

    ``None`` means that one eye lacks enough visible points or that the two
    computed centers coincide.  Callers can then use an explicitly recorded
    bbox-diagonal fallback.
    """
    truth, visibility = validate_landmarks(truth, visibility)
    if min_visible_per_eye < 1:
        raise ValueError('min_visible_per_eye must be positive')
    centers = []
    for indices in (LEFT_EYE, RIGHT_EYE):
        selected = np.asarray(indices)[visibility[list(indices)]]
        if len(selected) < min_visible_per_eye:
            return None
        centers.append(truth[selected].mean(axis=0))
    distance = float(np.linalg.norm(centers[0] - centers[1]))
    return distance if np.isfinite(distance) and distance > 0 else None


def nme_details(prediction, truth, visibility, bbox, preferred='interocular',
                fallback='bbox_diagonal'):
    """Return NME plus the normalization method and distance actually used."""
    if preferred not in ('interocular', 'bbox_diagonal'):
        raise ValueError('preferred must be interocular or bbox_diagonal')
    if fallback not in (None, 'bbox_diagonal'):
        raise ValueError('fallback must be bbox_diagonal or None')
    method = preferred
    distance = interocular_distance(truth, visibility) if preferred == 'interocular' else bbox_diagonal(bbox)
    if distance is None:
        if fallback is None:
            return dict(nme=None, normalization=None, normalizer=None)
        method, distance = fallback, bbox_diagonal(bbox)
    return dict(nme=nme(prediction, truth, visibility, distance),
                normalization=method, normalizer=distance)
