# B 的基础版：在像素差候选中寻找分割阈值，构建最大深度为 2 的弱树。
"""B: greedy weighted depth-2 tree over a fixed random candidate pool."""
import numpy as np


# 递归寻找使加权分类错误较少的切分；纯类别分支可以提前成为叶子。
def fit_tree(x, y, weights, depth=2):
    # x 每行是一个样本、每列是一个候选像素差；叶子预测加权票数较多的类别。
    leaf = 1 if np.dot(weights, y) >= 0 else -1
    if depth == 0 or len(y) < 2 or len(np.unique(y)) == 1:
        return {'leaf': leaf}
    best = None
    for j in range(x.shape[1]):
        values = np.unique(x[:, j])
        if len(values) < 2:
            continue
        thresholds = (values[:-1].astype(float) + values[1:]) / 2
        # 用相邻取值的中点作阈值；基础版每个特征最多尝试 16 个，控制训练量。
        # Bounded baseline search; B can improve search efficiency/quality.
        if len(thresholds) > 16:
            thresholds = thresholds[np.linspace(0, len(thresholds)-1, 16).astype(int)]
        for threshold in thresholds:
            left = x[:, j] <= threshold
            # 左右子集各预测多数类，少数类权重之和就是这次切分的代价。
            error = sum(min(weights[mask & (y == 1)].sum(), weights[mask & (y == -1)].sum()) for mask in (left, ~left))
            if best is None or error < best[0]:
                best = (error, j, float(threshold), left)
    if best is None:
        return {'leaf': leaf}
    _, j, threshold, left = best
    # 选中当前层最优切分后，再分别建立左右子树，每深入一层 depth 减 1。
    return {'feature': j, 'threshold': threshold,
            'left': fit_tree(x[left], y[left], weights[left], depth-1),
            'right': fit_tree(x[~left], y[~left], weights[~left], depth-1)}


# 把特征矩阵沿树节点分流，返回每个样本的叶子分数。
def predict_tree(tree, x):
    if 'leaf' in tree:
        return np.full(len(x), tree['leaf'], dtype=float)
    left = x[:, tree['feature']] <= tree['threshold']
    out = np.empty(len(x))
    out[left] = predict_tree(tree['left'], x[left])
    out[~left] = predict_tree(tree['right'], x[~left])
    return out
