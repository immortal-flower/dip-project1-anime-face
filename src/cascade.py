# B 的基础版：多个阶段串联，窗口在任一阶段不合格就提前退出。
"""B: baseline cascade training and real early rejection at inference."""
import numpy as np
from .channels11 import compute_11_channels
from .adaboost import train_stage, stage_scores


# 按候选的通道和两点坐标取像素差；x/y 是当前窗口左上角。
def sample_features(channels, features, x=0, y=0):
    return np.asarray([int(channels[c][y+y1, x+x1])-int(channels[c][y+y2, x+x2])
                       for c, x1, y1, x2, y2 in features], dtype=float)


# 用训练样本学习阶段，用验证正样本设置阈值；微型数据回退会记入日志。
def train_cascade(patches, labels, val_patches, val_labels, seed=42, candidates=64, stages=3, rounds=5):
    if stages < 3 or candidates < 1 or rounds < 1:
        raise ValueError('Require at least 3 stages and positive candidates/rounds')
    rng = np.random.default_rng(seed)
    # All channels have valid support in top-left 17x17 of a 24x24 patch.
    features = np.column_stack((rng.integers(0, 11, candidates), rng.integers(0, 17, (candidates, 4)))).tolist()
    # 先为每张小图计算候选像素差；后续树训练使用这个矩阵，避免反复算通道。
    x = np.stack([sample_features(compute_11_channels(p), features) for p in patches])
    vx = np.stack([sample_features(compute_11_channels(p), features) for p in val_patches])
    if set(labels.tolist()) != {-1, 1} or set(val_labels.tolist()) != {-1, 1}:
        raise ValueError('Train and validation must both contain +/-1 labels')
    trained, logs = [], []
    active = np.ones(len(labels), dtype=bool)
    val_active = np.ones(len(val_labels), dtype=bool)
    for _ in range(stages):
        subset = active.copy()
        # Tiny datasets may exhaust negatives. Explicitly log the fallback.
        # 微型数据的负样本可能耗尽；记录回退，正式训练应接入困难负样本。
        fallback = len(np.unique(labels[subset])) < 2
        if fallback:
            subset[:] = True
        stage = train_stage(x[subset], labels[subset], rounds)
        positives = val_active & (val_labels == 1)
        if not positives.any():
            raise ValueError('No validation positives survive; cannot calibrate next stage')
        # 阈值取当前存活验证正例最低分，保住这些正例；这是基础版的校准策略。
        stage['threshold'] = float(np.min(stage_scores(stage, vx[positives])))
        before = int(active.sum())
        # &= 累积前面所有阶段的筛选结果；已经被拒绝的样本不会重新变为通过。
        active &= stage_scores(stage, x) >= stage['threshold']
        val_active &= stage_scores(stage, vx) >= stage['threshold']
        logs.append(dict(train_input=before, train_pass=int(active.sum()),
                         val_pass=int(val_active.sum()), reused_training_pool=fallback))
        trained.append(stage)
    return dict(features=features, stages=trained, seed=seed, training_log=logs)


# 逐阶段判断窗口，通过返回分数，否则立即拒绝并记录已通过级数。
def predict_window(model, channels, x, y):
    features = sample_features(channels, model['features'], x, y)[None, :]
    score = 0.0
    passed = 0
    for stage in model['stages']:
        current = float(stage_scores(stage, features)[0])
        if current < stage['threshold']:
            return False, score, passed
        passed += 1
        score += current - stage['threshold']
    return True, score, passed
