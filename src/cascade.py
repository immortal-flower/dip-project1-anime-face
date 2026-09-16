"""B: baseline cascade training and real early rejection at inference."""
import numpy as np
from .channels11 import compute_11_channels
from .adaboost import train_stage, stage_scores


def sample_features(channels, features, x=0, y=0):
    return np.asarray([int(channels[c][y+y1, x+x1])-int(channels[c][y+y2, x+x2])
                       for c, x1, y1, x2, y2 in features], dtype=float)


def train_cascade(patches, labels, val_patches, val_labels, seed=42, candidates=64, stages=3, rounds=5):
    if stages < 3 or candidates < 1 or rounds < 1:
        raise ValueError('Require at least 3 stages and positive candidates/rounds')
    rng = np.random.default_rng(seed)
    # All channels have valid support in top-left 17x17 of a 24x24 patch.
    features = np.column_stack((rng.integers(0, 11, candidates), rng.integers(0, 17, (candidates, 4)))).tolist()
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
        fallback = len(np.unique(labels[subset])) < 2
        if fallback:
            subset[:] = True
        stage = train_stage(x[subset], labels[subset], rounds)
        positives = val_active & (val_labels == 1)
        if not positives.any():
            raise ValueError('No validation positives survive; cannot calibrate next stage')
        stage['threshold'] = float(np.min(stage_scores(stage, vx[positives])))
        before = int(active.sum())
        active &= stage_scores(stage, x) >= stage['threshold']
        val_active &= stage_scores(stage, vx) >= stage['threshold']
        logs.append(dict(train_input=before, train_pass=int(active.sum()),
                         val_pass=int(val_active.sum()), reused_training_pool=fallback))
        trained.append(stage)
    return dict(features=features, stages=trained, seed=seed, training_log=logs)


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
