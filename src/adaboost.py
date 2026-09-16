"""B: discrete AdaBoost with depth-2 weak learners."""
import numpy as np
from .depth2_tree import fit_tree, predict_tree


def stage_scores(stage, x):
    return sum((item['alpha'] * predict_tree(item['tree'], x) for item in stage['trees']), np.zeros(len(x)))


def train_stage(x, y, rounds=5):
    weights = np.full(len(y), 1 / len(y))
    trees = []
    for _ in range(rounds):
        tree = fit_tree(x, y, weights)
        prediction = predict_tree(tree, x)
        error = float(weights[prediction != y].sum())
        if error >= 0.5 - 1e-12:
            break
        alpha = float(0.5 * np.log((1-error) / max(error, 1e-9)))
        trees.append(dict(tree=tree, alpha=alpha))
        weights *= np.exp(-alpha*y*prediction)
        weights /= weights.sum()
        if error <= 1e-9:
            break
    if not trees:
        raise ValueError('No useful weak learner; improve data or increase candidate features')
    return dict(trees=trees, threshold=0.0)
