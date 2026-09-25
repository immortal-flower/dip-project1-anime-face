# B 的基础版：让后续弱树更关注分错的样本，用加权投票组成一个阶段。
"""B: discrete AdaBoost with depth-2 weak learners."""
import numpy as np
from .depth2_tree import fit_tree, predict_tree


# 累加本阶段所有弱树的 alpha×预测值，得到每个窗口的阶段分数。
def stage_scores(stage, x):
    return sum((item['alpha'] * predict_tree(item['tree'], x) for item in stage['trees']), np.zeros(len(x)))


# 训练若干弱树；分错的样本权重增加，下一轮会更关注这些样本。
def train_stage(x, y, rounds=5):
    weights = np.full(len(y), 1 / len(y))
    trees = []
    for _ in range(rounds):
        tree = fit_tree(x, y, weights)
        prediction = predict_tree(tree, x)
        error = float(weights[prediction != y].sum())
        # 错误率达到一半就没有正向投票价值；越准确的树，alpha 投票权越大。
        if error >= 0.5 - 1e-12:
            break
        alpha = float(0.5 * np.log((1-error) / max(error, 1e-9)))
        trees.append(dict(tree=tree, alpha=alpha))
        # y 与预测同号时降低权重，异号时提高权重，然后归一化。
        weights *= np.exp(-alpha*y*prediction)
        weights /= weights.sum()
        if error <= 1e-9:
            break
    if not trees:
        raise ValueError('No useful weak learner; improve data or increase candidate features')
    return dict(trees=trees, threshold=0.0)
