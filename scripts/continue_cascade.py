"""Append late cascade stages; early stages/features are exactly preserved."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from src.cascade import continue_cascade
from src.data_io import detection_samples, load_manifest, write_json


def _sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def train(base_model_dir, manifest, output, added_stages=2, rounds=10,
          target_recall=.98, target_fpr=.5, hard_negative_weight=3.0):
    base_model_dir, output = Path(base_model_dir), Path(output)
    rows = list(load_manifest(manifest))
    patches, labels = detection_samples(rows, 'train')
    val_patches, val_labels = detection_samples(rows, 'val')
    train_rows = [row for row in rows if row['split'] == 'train']
    weights = np.asarray([
        hard_negative_weight if row.get('hard_negative') else 1.0
        for row in train_rows
    ])
    base = json.loads((base_model_dir / 'detector.json').read_text('utf-8'))
    frozen = dict(features=base['features'], stages=base['stages'])
    frozen_digest = _sha(frozen)
    model = continue_cascade(
        base, patches, labels, val_patches, val_labels,
        added_stages=added_stages, rounds=rounds,
        target_recall=target_recall, target_false_positive_rate=target_fpr,
        sample_weights=weights,
    )
    prefix = dict(features=model['features'],
                  stages=model['stages'][:len(base['stages'])])
    if _sha(prefix) != frozen_digest:
        raise AssertionError('Frozen early cascade changed')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'feature_definition.json', 'landmark.npz'):
        shutil.copy2(base_model_dir / name, output / name)
    write_json(output / 'detector.json', model)
    config = json.loads((output / 'config.json').read_text('utf-8'))
    config['late_stage_continuation'] = dict(
        frozen_early_stages=len(base['stages']), added_stages=added_stages,
        max_weak_trees=rounds, target_recall=target_recall,
        target_false_positive_rate=target_fpr,
        hard_negative_weight=hard_negative_weight,
        frozen_early_sha256=frozen_digest,
    )
    write_json(output / 'config.json', config)
    write_json(output / 'splits.json', [
        {k: row[k] for k in ('image', 'source_id', 'split', 'label', 'bbox')}
        for row in rows
    ])
    return model['continuation_summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--added-stages', type=int, default=2)
    parser.add_argument('--max-weak-trees', type=int, default=10)
    parser.add_argument('--target-recall', type=float, default=.98)
    parser.add_argument('--target-fpr', type=float, default=.5)
    parser.add_argument('--hard-negative-weight', type=float, default=3.0)
    args = parser.parse_args()
    result = train(
        args.base_model_dir, args.manifest, args.output, args.added_stages,
        args.max_weak_trees, args.target_recall, args.target_fpr,
        args.hard_negative_weight,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
