"""Compare two cascade models on identical validation and hard-negative rows."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.adaboost import stage_scores
from src.cascade import sample_features
from src.channels11 import compute_11_channels
from src.data_io import load_manifest, write_json


def _patch(row):
    x1, y1, x2, y2 = row['bbox']
    gray = cv2.cvtColor(row['_image'], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray[y1:y2, x1:x2], (24, 24))


def _survival(rows, model):
    if not rows:
        raise ValueError('Evaluation subset is empty')
    x = np.stack([
        sample_features(compute_11_channels(_patch(row)), model['features'])
        for row in rows
    ])
    active = np.ones(len(rows), dtype=bool)
    result = []
    for index, stage in enumerate(model['stages']):
        input_count = int(active.sum())
        active &= stage_scores(stage, x) >= stage['threshold']
        passed = int(active.sum())
        result.append(dict(
            stage=index,
            input=input_count,
            passed=passed,
            rejected=input_count - passed,
            conditional_pass_rate=passed / input_count if input_count else None,
            cumulative_pass_rate=passed / len(rows),
        ))
    return result


def _training_validation(model):
    initial_negative = model['training_log'][0]['validation']['input_negative']
    return [dict(
        stage=item['stage'],
        weak_trees=item['weak_trees_selected'],
        target_met=item['target_met'],
        conditional_false_positive_rate=item['validation']['false_positive_rate'],
        cumulative_false_positive_rate=(
            item['validation']['pass_negative'] / initial_negative
        ),
        conditional_recall=item['validation']['recall'],
        pass_negative=item['validation']['pass_negative'],
        pass_positive=item['validation']['pass_positive'],
    ) for item in model['training_log']]


def compare(original_dir, retrained_dir, base_manifest, hard_negative_manifest):
    original_dir, retrained_dir = Path(original_dir), Path(retrained_dir)
    original = json.loads((original_dir/'detector.json').read_text(encoding='utf-8'))
    retrained = json.loads((retrained_dir/'detector.json').read_text(encoding='utf-8'))
    base_rows = list(load_manifest(base_manifest))
    hard_rows = [
        row for row in load_manifest(hard_negative_manifest)
        if row.get('hard_negative')
    ]
    subsets = dict(
        validation_negative=[r for r in base_rows if r['split'] == 'val' and r['label'] == -1],
        validation_positive=[r for r in base_rows if r['split'] == 'val' and r['label'] == 1],
        accepted_hard_negative=hard_rows,
    )
    if not hard_rows:
        raise ValueError('Hard-negative manifest contains no hard negatives')
    for row in hard_rows:
        if row['split'] != 'train' or row['label'] != -1 or not row.get('hard_negative'):
            raise ValueError('Hard-negative manifest must contain accepted train negatives only')

    result = dict(
        original_model=str(original_dir.resolve()),
        retrained_model=str(retrained_dir.resolve()),
        base_manifest=str(Path(base_manifest).resolve()),
        hard_negative_manifest=str(Path(hard_negative_manifest).resolve()),
        features_identical=original['features'] == retrained['features'],
        seed_identical=original.get('seed') == retrained.get('seed'),
        original_training_summary=original['training_summary'],
        retrained_training_summary=retrained['training_summary'],
        training_validation=dict(
            original=_training_validation(original),
            retrained=_training_validation(retrained),
        ),
        independent_evaluation={},
    )
    for name, rows in subsets.items():
        result['independent_evaluation'][name] = dict(
            count=len(rows),
            original=_survival(rows, original),
            retrained=_survival(rows, retrained),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-model', required=True)
    parser.add_argument('--retrained-model', required=True)
    parser.add_argument('--base-manifest', required=True)
    parser.add_argument('--hard-negative-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(
        args.original_model, args.retrained_model, args.base_manifest,
        args.hard_negative_manifest,
    )
    write_json(args.output, result)
    print(f'Comparison saved: {args.output}')


if __name__ == '__main__':
    main()
