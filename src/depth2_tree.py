"""B: greedy weighted depth-2 tree over a fixed random candidate pool."""
import numpy as np


def fit_tree(x, y, weights, depth=2):
    leaf = 1 if np.dot(weights, y) >= 0 else -1
    if depth == 0 or len(y) < 2 or len(np.unique(y)) == 1:
        return {'leaf': leaf}
    best = None
    for j in range(x.shape[1]):
        values = np.unique(x[:, j])
        if len(values) < 2:
            continue
        thresholds = (values[:-1].astype(float) + values[1:]) / 2
        # Bounded baseline search; B can improve search efficiency/quality.
        if len(thresholds) > 16:
            thresholds = thresholds[np.linspace(0, len(thresholds)-1, 16).astype(int)]
        for threshold in thresholds:
            left = x[:, j] <= threshold
            error = sum(min(weights[mask & (y == 1)].sum(), weights[mask & (y == -1)].sum()) for mask in (left, ~left))
            if best is None or error < best[0]:
                best = (error, j, float(threshold), left)
    if best is None:
        return {'leaf': leaf}
    _, j, threshold, left = best
    return {'feature': j, 'threshold': threshold,
            'left': fit_tree(x[left], y[left], weights[left], depth-1),
            'right': fit_tree(x[~left], y[~left], weights[~left], depth-1)}


def predict_tree(tree, x):
    if 'leaf' in tree:
        return np.full(len(x), tree['leaf'], dtype=float)
    left = x[:, tree['feature']] <= tree['threshold']
    out = np.empty(len(x))
    out[left] = predict_tree(tree['left'], x[left])
    out[~left] = predict_tree(tree['right'], x[~left])
    return out
