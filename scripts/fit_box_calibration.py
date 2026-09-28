"""Fit global box center/size correction on train matches and verify on val."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data_io import write_json
from src.detection_metrics import evaluate_dataset, iou
from src.grouping import calibrate_box


def _load_eval(path):
    value = json.loads(Path(path).read_text('utf-8'))
    if value.get('split') not in ('train', 'val'):
        raise ValueError('Calibration accepts only train/val evaluations')
    return value


def _pages(pages_path, selected):
    rows = json.loads(Path(pages_path).read_text('utf-8'))
    by_id = {row['page_id']: row for row in rows}
    return [by_id[page_id] for page_id in selected]


def _pairs(rows, predictions, minimum_iou=.2):
    pairs = []
    for page in rows:
        available = set(range(len(page['bboxes'])))
        for prediction in sorted(predictions.get(page['page_id'], []),
                                 key=lambda item: item['score'], reverse=True):
            candidates = [(iou(prediction['bbox'], page['bboxes'][j]), j)
                          for j in available]
            if not candidates:
                continue
            overlap, index = max(candidates)
            if overlap >= minimum_iou:
                available.remove(index)
                pairs.append((prediction['bbox'], page['bboxes'][index], overlap))
    return pairs


def _median_calibration(pairs):
    values = []
    for predicted, truth, _ in pairs:
        px1, py1, px2, py2 = predicted
        tx1, ty1, tx2, ty2 = truth
        pw, ph, tw, th = px2 - px1, py2 - py1, tx2 - tx1, ty2 - ty1
        values.append([
            ((tx1 + tx2) - (px1 + px2)) / (2 * pw),
            ((ty1 + ty2) - (py1 + py2)) / (2 * ph), tw / pw, th / ph,
        ])
    median = np.median(np.asarray(values), axis=0)
    return dict(dx=float(median[0]), dy=float(median[1]),
                scale_x=float(median[2]), scale_y=float(median[3]))


def _transform(predictions, calibration):
    return {page_id: [dict(item, bbox=calibrate_box(item['bbox'], calibration))
                      for item in items]
            for page_id, items in predictions.items()}


def fit(train_evaluation, validation_evaluation, pages, output,
        score_threshold=None):
    train, validation = _load_eval(train_evaluation), _load_eval(validation_evaluation)
    train_pages = _pages(pages, train['selected_page_ids'])
    val_pages = _pages(pages, validation['selected_page_ids'])
    train_predictions = train['predictions']
    val_predictions = validation['predictions']
    if score_threshold is not None:
        train_predictions = {key: [item for item in items
                                    if item['score'] >= score_threshold]
                             for key, items in train_predictions.items()}
        val_predictions = {key: [item for item in items
                                  if item['score'] >= score_threshold]
                           for key, items in val_predictions.items()}
    pairs = _pairs(train_pages, train_predictions)
    if len(pairs) < 5:
        raise ValueError('At least five train-only matched boxes are required')
    proposal = _median_calibration(pairs)
    candidates = []
    for strength in np.linspace(0, 1.25, 6):
        calibration = dict(
            dx=proposal['dx'] * strength, dy=proposal['dy'] * strength,
            scale_x=1 + (proposal['scale_x'] - 1) * strength,
            scale_y=1 + (proposal['scale_y'] - 1) * strength,
        )
        metrics = evaluate_dataset(
            val_pages, _transform(val_predictions, calibration), .5)
        candidates.append(dict(strength=float(strength), calibration=calibration,
                               metrics={k: metrics[k] for k in
                                        ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')}))
    selected = max(candidates, key=lambda item: (
        item['metrics']['f1'], item['metrics']['recall'],
        -item['metrics']['fp'], -abs(item['strength'])))
    result = dict(
        schema_version=1, fit_split='train', selection_split='val',
        test_data_used=False, train_match_count=len(pairs),
        score_threshold=score_threshold,
        train_minimum_match_iou=.2, train_median_proposal=proposal,
        candidates=candidates, selected=selected,
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-evaluation', required=True)
    parser.add_argument('--validation-evaluation', required=True)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--score-threshold', type=float)
    args = parser.parse_args()
    print(json.dumps(fit(args.train_evaluation, args.validation_evaluation,
                         args.pages, args.output,
                         args.score_threshold)['selected'], ensure_ascii=False))


if __name__ == '__main__':
    main()
