"""Frozen evaluation on C's user-approved 38 local face boxes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_json
from src.detection_metrics import evaluate_dataset, iou
from src.detector import AnimeFaceDetector
from src.shape_regression import predict_shape


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(manifest, image_root, model_dir, output, iou_threshold=.5,
             nme_threshold=.1):
    manifest, image_root = Path(manifest), Path(image_root)
    rows = json.loads(manifest.read_text('utf-8'))
    if len(rows) != 38 or any(row.get('split') != 'test' for row in rows):
        raise ValueError('Expected the frozen 38-image C test manifest')
    detector = AnimeFaceDetector(model_dir)
    predictions, page_rows, details = {}, [], []
    nmes = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        image = read_image(image_root / row['image'])
        detections = detector.detect(image)
        page_id = row['source_id']
        predictions[page_id] = detections
        page_rows.append(dict(page_id=page_id, source_id=page_id,
                              bboxes=[row['bbox']]))
        best = max(detections, key=lambda item: iou(item['bbox'], row['bbox']),
                   default=None)
        overlap, nme = 0.0, None
        if best is not None:
            overlap = iou(best['bbox'], row['bbox'])
            if overlap >= iou_threshold:
                truth = np.asarray(row['landmarks'], dtype=float)
                visible = np.asarray(row['visibility'], dtype=bool)
                predicted = np.asarray(best['landmarks'], dtype=float)
                width = row['bbox'][2] - row['bbox'][0]
                height = row['bbox'][3] - row['bbox'][1]
                nme = float(np.linalg.norm(predicted[visible] - truth[visible], axis=1).mean()
                            / np.sqrt(width * height))
                nmes.append(nme)
        details.append(dict(source_id=page_id, detections=len(detections),
                            best_iou=overlap, nme=nme,
                            end_to_end_success=nme is not None and nme <= nme_threshold))
        print(json.dumps(dict(image=index + 1, total=len(rows),
                              detections=len(detections), best_iou=overlap)), flush=True)
    detection = evaluate_dataset(page_rows, predictions, iou_threshold)
    successes = sum(item['end_to_end_success'] for item in details)
    result = dict(
        schema_version=1, purpose='frozen C 38-image end-to-end evaluation',
        split='test', test_images=len(rows), manifest_sha256=_sha(manifest),
        detector_sha256=_sha(Path(model_dir) / 'detector.json'),
        config_sha256=_sha(Path(model_dir) / 'config.json'),
        iou_threshold=iou_threshold, nme_definition='mean visible point error / sqrt(box area)',
        nme_threshold=nme_threshold, detection_metrics=detection,
        landmark_evaluable=len(nmes), mean_nme=float(np.mean(nmes)) if nmes else None,
        end_to_end_successes=successes,
        end_to_end_success_rate=successes / len(rows), details=details,
        seconds=time.perf_counter() - started,
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.image_root, args.model_dir, args.output)
    print(json.dumps({k: result[k] for k in ('detection_metrics', 'landmark_evaluable',
          'mean_nme', 'end_to_end_successes', 'end_to_end_success_rate')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
