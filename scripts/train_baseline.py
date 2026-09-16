"""Shared integration entry point; run from repo root with python -m."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from src.data_io import load_manifest, detection_samples, write_json
from src.cascade import train_cascade
from src.shape_regression import train_shape


def train(manifest, output, synthetic=False):
    rows = list(load_manifest(manifest))
    patches, labels = detection_samples(rows, 'train')
    val, val_labels = detection_samples(rows, 'val')
    model = train_cascade(patches, labels, val, val_labels)
    landmark_rows = [r for r in rows if r['split'] == 'train' and r['label'] == 1 and 'landmarks' in r]
    if not landmark_rows:
        raise ValueError('Training needs positive rows with 28-point annotations')
    landmark = train_shape([cv2.cvtColor(r['_image'], cv2.COLOR_BGR2GRAY) for r in landmark_rows],
                           [r['bbox'] for r in landmark_rows], [r['landmarks'] for r in landmark_rows],
                           [r['visibility'] for r in landmark_rows])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'detector.json', model)
    np.savez_compressed(output/'landmark.npz', **landmark)
    write_json(output/'config.json', dict(schema_version=1, synthetic=synthetic,
        window_size=[24, 24], num_channels=11, scale_factor=1.2, step=1, nms_threshold=0.3,
        landmark_count=28, seed=42, coordinate_convention='xyxy-exclusive',
        landmark_order='synthetic ellipse 0..27' if synthetic else 'manifest order; supply numbering diagram',
        feature_rounding='floor((sum_of_four+2)/4)', feature_border='zero outside full valid support'))
    write_json(output/'splits.json', [{k: r[k] for k in ('image', 'source_id', 'split', 'label', 'bbox')} for r in rows])
    print(f'Model saved: {output}; synthetic={synthetic}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    train(args.manifest, args.output, args.synthetic)


if __name__ == '__main__':
    main()
