"""B: target-controlled cascade training and early rejection at inference."""
import time

import numpy as np

from .adaboost import stage_scores, train_stage
from .channels11 import compute_11_channels


def sample_features(channels, features, x=0, y=0):
    """Evaluate candidate channel pixel differences at one window origin."""
    return np.asarray(
        [
            int(channels[c][y + y1, x + x1])
            - int(channels[c][y + y2, x + x2])
            for c, x1, y1, x2, y2 in features
        ],
        dtype=float,
    )


def calibrate_threshold(scores, labels, target_recall):
    """Choose the highest threshold that retains the requested positive recall."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 1 or len(scores) != len(labels) or len(scores) == 0:
        raise ValueError('scores and labels must be aligned non-empty 1-D arrays')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0, 1]')
    positive_scores = scores[labels == 1]
    if len(positive_scores) == 0:
        raise ValueError('Threshold calibration requires validation positives')

    required = max(1, int(np.ceil(target_recall * len(positive_scores) - 1e-12)))
    threshold = float(np.sort(positive_scores)[::-1][required - 1])
    passed = scores >= threshold
    positive = labels == 1
    negative = labels == -1
    recall = float(np.mean(passed[positive]))
    false_positive_rate = float(np.mean(passed[negative])) if negative.any() else None
    return threshold, recall, false_positive_rate


def _stage_stats(labels, input_mask, pass_mask):
    positive = input_mask & (labels == 1)
    negative = input_mask & (labels == -1)
    passed_positive = int((pass_mask & positive).sum())
    passed_negative = int((pass_mask & negative).sum())
    input_positive = int(positive.sum())
    input_negative = int(negative.sum())
    return dict(
        input_total=int(input_mask.sum()),
        input_positive=input_positive,
        input_negative=input_negative,
        pass_total=int((pass_mask & input_mask).sum()),
        pass_positive=passed_positive,
        pass_negative=passed_negative,
        rejected_positive=input_positive - passed_positive,
        rejected_negative=input_negative - passed_negative,
        false_reject=input_positive - passed_positive,
        false_accept=passed_negative,
        recall=passed_positive / input_positive if input_positive else None,
        false_positive_rate=(
            passed_negative / input_negative if input_negative else None
        ),
    )


def _exhaustion_reason(labels, active, val_labels, val_active):
    missing = []
    if not np.any(active & (labels == 1)):
        missing.append('train_positives')
    if not np.any(active & (labels == -1)):
        missing.append('train_negatives')
    if not np.any(val_active & (val_labels == 1)):
        missing.append('validation_positives')
    if not np.any(val_active & (val_labels == -1)):
        missing.append('validation_negatives')
    return '_and_'.join(missing) + '_exhausted' if missing else None


def train_cascade(
    patches,
    labels,
    val_patches,
    val_labels,
    seed=42,
    candidates=64,
    stages=3,
    rounds=5,
    target_recall=1.0,
    target_false_positive_rate=0.5,
    sample_weights=None,
):
    """Train stages until the requested count or a safe stop condition is reached."""
    if stages < 3 or candidates < 1 or rounds < 1:
        raise ValueError('Require at least 3 stages and positive candidates/rounds')
    if not 0 < target_recall <= 1:
        raise ValueError('target_recall must be in (0, 1]')
    if not 0 <= target_false_positive_rate < 1:
        raise ValueError('target_false_positive_rate must be in [0, 1)')

    labels = np.asarray(labels)
    val_labels = np.asarray(val_labels)
    if sample_weights is None:
        sample_weights = np.ones(len(labels), dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if (
            sample_weights.shape != (len(labels),)
            or not np.isfinite(sample_weights).all()
            or np.any(sample_weights <= 0)
        ):
            raise ValueError('sample_weights must contain one positive finite value per sample')
    rng = np.random.default_rng(seed)
    features = np.column_stack(
        (rng.integers(0, 11, candidates), rng.integers(0, 17, (candidates, 4)))
    ).tolist()
    x = np.stack(
        [sample_features(compute_11_channels(patch), features) for patch in patches]
    )
    vx = np.stack(
        [sample_features(compute_11_channels(patch), features) for patch in val_patches]
    )
    if set(labels.tolist()) != {-1, 1} or set(val_labels.tolist()) != {-1, 1}:
        raise ValueError('Train and validation must both contain +/-1 labels')

    trained, logs = [], []
    active = np.ones(len(labels), dtype=bool)
    val_active = np.ones(len(val_labels), dtype=bool)
    cascade_start = time.perf_counter()
    stop_reason = 'requested_stages_completed'

    for stage_index in range(stages):
        exhausted = _exhaustion_reason(labels, active, val_labels, val_active)
        if exhausted:
            stop_reason = exhausted
            break

        stage_start = time.perf_counter()
        fitted = train_stage(
            x[active], labels[active], rounds,
            initial_weights=sample_weights[active],
        )
        active_val_x = vx[val_active]
        active_val_labels = val_labels[val_active]
        calibration = []
        for tree_count in range(1, len(fitted['trees']) + 1):
            prefix = dict(trees=fitted['trees'][:tree_count], threshold=0.0)
            threshold, recall, false_positive_rate = calibrate_threshold(
                stage_scores(prefix, active_val_x), active_val_labels, target_recall
            )
            candidate = dict(
                weak_trees=tree_count,
                threshold=threshold,
                recall=recall,
                false_positive_rate=false_positive_rate,
                target_met=(
                    recall + 1e-12 >= target_recall
                    and false_positive_rate <= target_false_positive_rate + 1e-12
                ),
            )
            calibration.append(candidate)
            if candidate['target_met']:
                break

        selected = next((item for item in calibration if item['target_met']), None)
        if selected is None:
            selected = min(
                calibration,
                key=lambda item: (
                    item['false_positive_rate'],
                    -item['recall'],
                    item['weak_trees'],
                ),
            )
        tree_count = selected['weak_trees']
        selection_reason = (
            'target_met' if selected['target_met'] else 'best_available_within_limit'
        )
        stage = dict(
            trees=fitted['trees'][:tree_count],
            threshold=selected['threshold'],
            boosting_log=fitted['boosting_log'][:tree_count],
            training_stop_reason=fitted['training_stop_reason'],
            target_recall=target_recall,
            target_false_positive_rate=target_false_positive_rate,
            target_met=selected['target_met'],
            selection_reason=selection_reason,
        )

        train_before = active.copy()
        val_before = val_active.copy()
        active &= stage_scores(stage, x) >= stage['threshold']
        val_active &= stage_scores(stage, vx) >= stage['threshold']
        train_stats = _stage_stats(labels, train_before, active)
        validation_stats = _stage_stats(val_labels, val_before, val_active)
        logs.append(
            dict(
                stage=stage_index,
                weak_trees_selected=tree_count,
                weak_trees_fitted=len(fitted['trees']),
                threshold=stage['threshold'],
                target_recall=target_recall,
                target_false_positive_rate=target_false_positive_rate,
                target_met=selected['target_met'],
                selection_reason=selection_reason,
                train=train_stats,
                validation=validation_stats,
                calibration=calibration,
                seconds=time.perf_counter() - stage_start,
                train_input=train_stats['input_total'],
                train_pass=train_stats['pass_total'],
                val_input=validation_stats['input_total'],
                val_pass=validation_stats['pass_total'],
            )
        )
        trained.append(stage)

        if (
            stage_index + 1 < stages
            and train_stats['rejected_negative'] == 0
            and validation_stats['rejected_negative'] == 0
        ):
            stop_reason = 'no_negative_reduction'
            break

    summary = dict(
        requested_stages=stages,
        trained_stages=len(trained),
        stopped_early=len(trained) < stages,
        stop_reason=stop_reason,
        all_stage_targets_met=bool(trained)
        and all(item['target_met'] for item in logs),
        target_recall=target_recall,
        target_false_positive_rate=target_false_positive_rate,
        max_weak_trees=rounds,
        candidates=candidates,
        weighted_training=not np.allclose(sample_weights, sample_weights[0]),
        seconds=time.perf_counter() - cascade_start,
    )
    return dict(
        features=features,
        stages=trained,
        seed=seed,
        training_log=logs,
        training_summary=summary,
    )


def continue_cascade(
    model, patches, labels, val_patches, val_labels, added_stages=2,
    rounds=10, target_recall=0.98, target_false_positive_rate=0.5,
    sample_weights=None,
):
    """Append late stages while preserving every existing feature/stage exactly."""
    if added_stages < 1 or rounds < 1:
        raise ValueError('added_stages and rounds must be positive')
    if not model.get('features') or not model.get('stages'):
        raise ValueError('A fitted base cascade is required')
    labels, val_labels = np.asarray(labels), np.asarray(val_labels)
    if set(labels.tolist()) != {-1, 1} or set(val_labels.tolist()) != {-1, 1}:
        raise ValueError('Train and validation must both contain +/-1 labels')
    if sample_weights is None:
        sample_weights = np.ones(len(labels), dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.shape != (len(labels),) or np.any(sample_weights <= 0):
        raise ValueError('sample_weights must be positive and aligned')

    features = model['features']
    x = np.stack([sample_features(compute_11_channels(p), features) for p in patches])
    vx = np.stack([sample_features(compute_11_channels(p), features) for p in val_patches])
    active, val_active = np.ones(len(labels), bool), np.ones(len(val_labels), bool)
    for stage in model['stages']:
        active &= stage_scores(stage, x) >= stage['threshold']
        val_active &= stage_scores(stage, vx) >= stage['threshold']

    appended, logs = [], []
    started = time.perf_counter()
    stop_reason = 'requested_stages_completed'
    base_count = len(model['stages'])
    for offset in range(added_stages):
        exhausted = _exhaustion_reason(labels, active, val_labels, val_active)
        if exhausted:
            stop_reason = exhausted
            break
        stage_start = time.perf_counter()
        fitted = train_stage(x[active], labels[active], rounds,
                             initial_weights=sample_weights[active])
        calibration = []
        for tree_count in range(1, len(fitted['trees']) + 1):
            prefix = dict(trees=fitted['trees'][:tree_count], threshold=0.0)
            threshold, recall, fpr = calibrate_threshold(
                stage_scores(prefix, vx[val_active]), val_labels[val_active],
                target_recall,
            )
            candidate = dict(
                weak_trees=tree_count, threshold=threshold, recall=recall,
                false_positive_rate=fpr,
                target_met=recall + 1e-12 >= target_recall and
                fpr <= target_false_positive_rate + 1e-12,
            )
            calibration.append(candidate)
            if candidate['target_met']:
                break
        selected = next((c for c in calibration if c['target_met']), None)
        if selected is None:
            selected = min(calibration, key=lambda c: (
                c['false_positive_rate'], -c['recall'], c['weak_trees']))
        count = selected['weak_trees']
        stage = dict(
            trees=fitted['trees'][:count], threshold=selected['threshold'],
            boosting_log=fitted['boosting_log'][:count],
            training_stop_reason=fitted['training_stop_reason'],
            target_recall=target_recall,
            target_false_positive_rate=target_false_positive_rate,
            target_met=selected['target_met'],
            selection_reason=('target_met' if selected['target_met'] else
                              'best_available_within_limit'),
            late_stage=True,
        )
        before, val_before = active.copy(), val_active.copy()
        active &= stage_scores(stage, x) >= stage['threshold']
        val_active &= stage_scores(stage, vx) >= stage['threshold']
        train_stats = _stage_stats(labels, before, active)
        val_stats = _stage_stats(val_labels, val_before, val_active)
        logs.append(dict(
            stage=base_count + offset, frozen_early_stages=base_count,
            weak_trees_selected=count, weak_trees_fitted=len(fitted['trees']),
            threshold=stage['threshold'], target_recall=target_recall,
            target_false_positive_rate=target_false_positive_rate,
            target_met=selected['target_met'],
            selection_reason=stage['selection_reason'], train=train_stats,
            validation=val_stats, calibration=calibration,
            seconds=time.perf_counter() - stage_start,
            train_input=train_stats['input_total'], train_pass=train_stats['pass_total'],
            val_input=val_stats['input_total'], val_pass=val_stats['pass_total'],
        ))
        appended.append(stage)
        if (offset + 1 < added_stages and
                train_stats['rejected_negative'] == 0 and
                val_stats['rejected_negative'] == 0):
            stop_reason = 'no_negative_reduction'
            break

    result = dict(model)
    result['stages'] = list(model['stages']) + appended
    result['training_log'] = list(model.get('training_log', [])) + logs
    result['continuation_summary'] = dict(
        frozen_early_stages=base_count, requested_added_stages=added_stages,
        added_stages=len(appended), stop_reason=stop_reason,
        target_recall=target_recall,
        target_false_positive_rate=target_false_positive_rate,
        max_weak_trees=rounds, seconds=time.perf_counter() - started,
    )
    return result


def predict_window(model, channels, x, y, collect_timing=False):
    """Evaluate one window, optionally returning feature and per-stage timings."""
    feature_start = time.perf_counter() if collect_timing else None
    features = sample_features(channels, model['features'], x, y)[None, :]
    feature_seconds = time.perf_counter() - feature_start if collect_timing else None
    score = 0.0
    passed = 0
    stage_seconds = []
    for stage in model['stages']:
        stage_start = time.perf_counter() if collect_timing else None
        current = float(stage_scores(stage, features)[0])
        if collect_timing:
            stage_seconds.append(time.perf_counter() - stage_start)
        if current < stage['threshold']:
            result = (False, score, passed)
            return result + (feature_seconds, stage_seconds) if collect_timing else result
        passed += 1
        score += current - stage['threshold']
    result = (True, score, passed)
    return result + (feature_seconds, stage_seconds) if collect_timing else result
