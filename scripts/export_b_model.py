"""Export the frozen B detector package with selected inference parameters."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.data_io import write_json


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_model(model_dir, threshold_result, output):
    model_dir, threshold_result, output = (
        Path(model_dir), Path(threshold_result), Path(output)
    )
    threshold_data = json.loads(threshold_result.read_text(encoding='utf-8'))
    selected = threshold_data['selected']
    validation_result_path = Path(threshold_data['validation_result'])
    if not validation_result_path.is_absolute():
        validation_result_path = threshold_result.parent.parent.parent / validation_result_path
    validation = json.loads(validation_result_path.read_text(encoding='utf-8'))
    scan_config = validation['scan_config']
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ('detector.json', 'feature_definition.json', 'splits.json', 'landmark.npz'):
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, output / name)
            copied.append(name)

    config = json.loads((model_dir/'config.json').read_text(encoding='utf-8'))
    config.update(
        step=scan_config['step'],
        scale_factor=scan_config['scale_factor'],
        nms_threshold=scan_config['nms_threshold'],
        pre_nms_limit=scan_config['pre_nms_limit'],
        scan_batch_size=scan_config['scan_batch_size'],
        score_threshold=selected['threshold'],
    )
    config['b_final_selection'] = dict(
        selection_split='val',
        selection_rule=threshold_data['selection_rule'],
        validation_pages=len(threshold_data['selected_page_ids']),
        validation_f1=selected['f1'],
        iou_threshold=threshold_data['iou_threshold'],
    )
    write_json(output/'config.json', config)
    copied.append('config.json')
    manifest = dict(
        schema_version=1,
        source_model=str(model_dir),
        threshold_result=str(threshold_result),
        files={
            name: dict(size=(output/name).stat().st_size, sha256=_sha256(output/name))
            for name in copied
        },
    )
    write_json(output/'B_MODEL_MANIFEST.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--threshold-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export_model(args.model_dir, args.threshold_result, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
