"""B: image-pyramid scanning, per-stage timing and NMS."""
import heapq
import time

import cv2
import numpy as np

from .adaboost import stage_scores
from .channels11 import compute_11_channels
from .grouping import nms
from .pyramid import image_pyramid


def _batch_features(channels, features, xs, ys):
    """Evaluate every candidate feature for a batch of window origins."""
    feature_array = np.asarray(features, dtype=np.intp)
    channel_array = np.asarray(channels)
    channel = feature_array[:, 0][None, :]
    x1, y1, x2, y2 = (
        feature_array[:, index][None, :] for index in range(1, 5)
    )
    xs = np.asarray(xs, dtype=np.intp)[:, None]
    ys = np.asarray(ys, dtype=np.intp)[:, None]
    first = channel_array[channel, ys + y1, xs + x1].astype(np.int16)
    second = channel_array[channel, ys + y2, xs + x2].astype(np.int16)
    return first - second


def _push_candidate(heap, limit, serial, detection):
    """Keep the highest scoring candidates while retaining stable tie order."""
    if limit is None:
        heap.append((0.0, serial, detection))
        return
    item = (float(detection['score']), -serial, detection)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def scan_image(image, model, config):
    """Scan every pyramid level and return original-coordinate detections and logs."""
    step = config['step']
    if not isinstance(step, int) or step < 1:
        raise ValueError('step must be a positive integer')
    pre_nms_limit = config.get('pre_nms_limit')
    if pre_nms_limit is not None and (
        not isinstance(pre_nms_limit, int) or pre_nms_limit < 1
    ):
        raise ValueError('pre_nms_limit must be a positive integer when supplied')
    batch_size = config.get('scan_batch_size', 4096)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('scan_batch_size must be a positive integer')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidate_heap, logs = [], []
    raw_candidate_count, candidate_serial = 0, 0
    for level, sx, sy in image_pyramid(gray, config['scale_factor']):
        start = time.perf_counter()
        channels = compute_11_channels(level)
        stage_count = len(model['stages'])
        count, passed, evaluated = 0, [0] * stage_count, [0] * stage_count
        stage_seconds = [0.0] * stage_count
        feature_seconds, candidates = 0.0, 0

        x_positions = np.arange(0, level.shape[1] - 23, step, dtype=np.intp)
        y_positions = np.arange(0, level.shape[0] - 23, step, dtype=np.intp)
        level_windows = len(x_positions) * len(y_positions)
        count = level_windows
        for batch_start in range(0, level_windows, batch_size):
            batch_stop = min(batch_start + batch_size, level_windows)
            linear = np.arange(batch_start, batch_stop, dtype=np.intp)
            xs = x_positions[linear % len(x_positions)]
            ys = y_positions[linear // len(x_positions)]

            feature_start = time.perf_counter()
            values = _batch_features(channels, model['features'], xs, ys)
            feature_seconds += time.perf_counter() - feature_start
            active = np.ones(len(xs), dtype=bool)
            scores = np.zeros(len(xs), dtype=float)
            for stage_index, stage in enumerate(model['stages']):
                active_indices = np.flatnonzero(active)
                if not len(active_indices):
                    break
                stage_start = time.perf_counter()
                current = stage_scores(stage, values[active_indices])
                stage_seconds[stage_index] += time.perf_counter() - stage_start
                evaluated[stage_index] += len(active_indices)
                accepted = current >= stage['threshold']
                passed[stage_index] += int(accepted.sum())
                scores[active_indices[accepted]] += (
                    current[accepted] - stage['threshold']
                )
                active[active_indices[~accepted]] = False

            for index in np.flatnonzero(active):
                candidates += 1
                raw_candidate_count += 1
                detection = dict(
                    bbox=[
                        float(xs[index] * sx), float(ys[index] * sy),
                        float((xs[index] + 24) * sx),
                        float((ys[index] + 24) * sy),
                    ],
                    score=float(scores[index]),
                )
                _push_candidate(
                    candidate_heap, pre_nms_limit, candidate_serial, detection
                )
                candidate_serial += 1

        stage_log = [
            dict(
                stage=stage_index,
                evaluated=evaluated[stage_index],
                passed=passed[stage_index],
                rejected=evaluated[stage_index] - passed[stage_index],
                seconds=stage_seconds[stage_index],
                average_seconds=(
                    stage_seconds[stage_index] / evaluated[stage_index]
                    if evaluated[stage_index]
                    else None
                ),
            )
            for stage_index in range(stage_count)
        ]
        logs.append(
            dict(
                width=level.shape[1],
                height=level.shape[0],
                windows=count,
                stage_pass=passed,
                stage_evaluated=evaluated,
                stage_rejected=[
                    evaluated[index] - passed[index] for index in range(stage_count)
                ],
                stage_seconds=stage_seconds,
                stages=stage_log,
                feature_seconds=feature_seconds,
                candidates=candidates,
                seconds=time.perf_counter() - start,
            )
        )
    if pre_nms_limit is None:
        predictions = [item[2] for item in candidate_heap]
    else:
        predictions = [item[2] for item in sorted(candidate_heap, key=lambda item: -item[1])]
    kept = nms(predictions, config['nms_threshold'])
    detections_before_score_filter = len(kept)
    score_threshold = config.get('score_threshold')
    if score_threshold is not None:
        if not np.isfinite(score_threshold):
            raise ValueError('score_threshold must be finite when supplied')
        kept = [item for item in kept if item['score'] >= score_threshold]
    if logs:
        logs[-1]['scan_summary'] = dict(
            raw_candidates=raw_candidate_count,
            candidates_sent_to_nms=len(predictions),
            pre_nms_truncated=raw_candidate_count - len(predictions),
            detections_after_nms=detections_before_score_filter,
            detections_after_score_filter=len(kept),
            pre_nms_limit=pre_nms_limit,
            score_threshold=score_threshold,
        )
    return kept, logs
