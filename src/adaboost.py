"""B: discrete AdaBoost with depth-2 weak learners."""
import numpy as np

from .depth2_tree import fit_tree, predict_tree


def stage_scores(stage, x):
    """Return the weighted vote of every weak tree in one stage."""
    return sum(
        (item['alpha'] * predict_tree(item['tree'], x) for item in stage['trees']),
        np.zeros(len(x)),
    )


def train_stage(x, y, rounds=5, initial_weights=None):
    """Fit one AdaBoost stage and retain an auditable per-round log."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError('x must be a non-empty 2-D array aligned with y')
    if set(np.unique(y).tolist()) != {-1, 1}:
        raise ValueError('AdaBoost training requires both -1 and +1 labels')
    if rounds < 1:
        raise ValueError('rounds must be positive')

    if initial_weights is None:
        weights = np.full(len(y), 1 / len(y))
    else:
        weights = np.asarray(initial_weights, dtype=float).copy()
        if (
            weights.shape != (len(y),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError('initial_weights must be finite nonnegative sample weights')
        weights /= weights.sum()
    trees, boosting_log = [], []
    stop_reason = 'max_rounds_reached'
    for round_index in range(rounds):
        tree = fit_tree(x, y, weights)
        prediction = predict_tree(tree, x)
        error = float(weights[prediction != y].sum())
        if error >= 0.5 - 1e-12:
            stop_reason = 'no_useful_weak_learner'
            break

        safe_error = min(max(error, 1e-12), 1 - 1e-12)
        alpha = float(0.5 * np.log((1 - safe_error) / safe_error))
        trees.append(dict(tree=tree, alpha=alpha))
        boosting_log.append(
            dict(round=round_index, weighted_error=error, alpha=alpha)
        )
        if error <= 1e-12:
            stop_reason = 'perfect_weak_learner'
            break

        weights *= np.exp(-alpha * y * prediction)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError('AdaBoost sample weights became invalid')
        weights /= total

    if not trees:
        raise ValueError('No useful weak learner; improve data or increase candidate features')
    return dict(
        trees=trees,
        threshold=0.0,
        boosting_log=boosting_log,
        training_stop_reason=stop_reason,
    )
