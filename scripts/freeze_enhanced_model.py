"""Copy a selected model and freeze validation-selected postprocessing."""
import argparse
import json
import shutil
from pathlib import Path

from src.data_io import write_json


def freeze(source, output, threshold_selection, box_calibration):
    source, output = Path(source), Path(output)
    threshold = json.loads(Path(threshold_selection).read_text('utf-8'))
    calibration = json.loads(Path(box_calibration).read_text('utf-8'))
    if threshold.get('selection_split') != 'val':
        raise ValueError('Threshold must be selected on validation')
    if calibration.get('fit_split') != 'train' or calibration.get('selection_split') != 'val':
        raise ValueError('Box calibration must fit train and select validation')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('detector.json', 'feature_definition.json', 'landmark.npz',
                 'splits.json'):
        shutil.copy2(source / name, output / name)
    config = json.loads((source / 'config.json').read_text('utf-8'))
    config.update(
        score_threshold=threshold['selected']['threshold'],
        nms_method='weighted', min_box_support=2,
        box_calibration=calibration['selected']['calibration'],
        pre_nms_limit=500, scan_batch_size=4096,
    )
    config['enhanced_selection'] = dict(
        training_hard_negatives=502, validation_hard_negatives=88,
        test_hard_negatives=0, threshold_metric='F2',
        threshold_validation_result=str(Path(threshold_selection).resolve()),
        box_calibration_result=str(Path(box_calibration).resolve()),
        leakage_policy='train fits; val selects; test evaluated once after freeze',
    )
    write_json(output / 'config.json', config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold-selection', required=True)
    parser.add_argument('--box-calibration', required=True)
    args = parser.parse_args()
    config = freeze(args.source, args.output, args.threshold_selection,
                    args.box_calibration)
    print(json.dumps({key: config[key] for key in
                      ('score_threshold', 'nms_method', 'min_box_support',
                       'box_calibration')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
