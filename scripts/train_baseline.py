# 共享训练入口：同一清单中同时需要检测正负样本和关键点标注，单有背景图片不足以运行。
"""Shared integration entry point; run from repo root with python -m."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from src.data_io import load_manifest, detection_samples, write_json
from src.cascade import train_cascade
from src.shape_regression import train_shape


# 串联检测与关键点训练，保存模型、配置、划分记录和是否合成数据的标记。
def train(manifest, output, synthetic=False, stages=3, max_weak_trees=5,
          target_recall=1.0, target_false_positive_rate=0.5,
          candidates=64, seed=42, hard_negative_weight=1.0):
    if hard_negative_weight <= 0:
        raise ValueError('hard_negative_weight must be positive')
    rows = list(load_manifest(manifest))
    patches, labels = detection_samples(rows, 'train')
    val, val_labels = detection_samples(rows, 'val')
    train_weights = np.asarray([
        hard_negative_weight if row.get('hard_negative') else 1.0
        for row in rows if row['split'] == 'train'
    ])
    model = train_cascade(
        patches, labels, val, val_labels, seed=seed, candidates=candidates,
        stages=stages, rounds=max_weak_trees, target_recall=target_recall,
        target_false_positive_rate=target_false_positive_rate,
        sample_weights=train_weights)
    model['training_summary']['hard_negative_weight'] = hard_negative_weight
    landmark_rows = [r for r in rows if r['split'] == 'train' and r['label'] == 1 and 'landmarks' in r]
    if not landmark_rows:
        raise ValueError('Training needs positive rows with 28-point annotations')
    landmark = train_shape([cv2.cvtColor(r['_image'], cv2.COLOR_BGR2GRAY) for r in landmark_rows],
                           [r['bbox'] for r in landmark_rows], [r['landmarks'] for r in landmark_rows],
                           [r['visibility'] for r in landmark_rows])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # 随模型保存完整通道定义与指纹，避免队友加载时猜测取整／边界版本。
    definition = (Path(__file__).resolve().parents[1]/'configs/feature_definition.json').read_bytes()
    (output/'feature_definition.json').write_bytes(definition)
    write_json(output/'detector.json', model)
    np.savez_compressed(output/'landmark.npz', **landmark)
    write_json(output/'config.json', dict(schema_version=1, synthetic=synthetic,
        feature_version='dip11-int-v1', feature_definition_file='feature_definition.json',
        feature_definition_sha256=hashlib.sha256(definition).hexdigest(),
        window_size=[24, 24], num_channels=11, scale_factor=1.2, step=1, nms_threshold=0.3,
        landmark_count=28, seed=seed, coordinate_convention='xyxy-exclusive',
        landmark_order='synthetic ellipse 0..27' if synthetic else 'manifest order; supply numbering diagram',
        feature_rounding='floor((sum_of_four+2)/4)', feature_border='zero outside full valid support',
        cascade_training=dict(requested_stages=stages,
                              max_weak_trees=max_weak_trees,
                              target_recall=target_recall,
                              target_false_positive_rate=target_false_positive_rate,
                              candidates=candidates,
                              hard_negative_weight=hard_negative_weight)))
    write_json(output/'splits.json', [{k: r[k] for k in ('image', 'source_id', 'split', 'label', 'bbox')} for r in rows])
    print(f'Model saved: {output}; synthetic={synthetic}')


# 命令行入口：读取参数、调用主要处理函数，并将结果保存到指定位置。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--stages', type=int, default=3)
    parser.add_argument('--max-weak-trees', type=int, default=5)
    parser.add_argument('--target-recall', type=float, default=1.0)
    parser.add_argument('--target-fpr', type=float, default=0.5)
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hard-negative-weight', type=float, default=1.0)
    args = parser.parse_args()
    train(args.manifest, args.output, args.synthetic, stages=args.stages,
          max_weak_trees=args.max_weak_trees,
          target_recall=args.target_recall,
          target_false_positive_rate=args.target_fpr,
          candidates=args.candidates, seed=args.seed,
          hard_negative_weight=args.hard_negative_weight)


if __name__ == '__main__':
    main()
