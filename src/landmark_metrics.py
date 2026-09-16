import numpy as np


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
