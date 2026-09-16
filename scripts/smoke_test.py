"""Synthetic integration check only; never report as anime-face performance."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from src.data_io import write_image, write_json, read_image
from src.detector import AnimeFaceDetector
from src.landmark_metrics import nme
from .train_baseline import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='results/smoke')
    args = parser.parse_args()
    root = Path(args.output_dir)
    rng = np.random.default_rng(42)
    rows = []
    angles = np.linspace(0, 2*np.pi, 28, endpoint=False)
    template = np.column_stack((12+6*np.cos(angles), 12+7*np.sin(angles)))
    for split, count in [('train', 12), ('val', 4), ('test', 2)]:
        for label in (1, -1):
            for j in range(count):
                image = np.full((24, 24, 3), int(rng.integers(160, 220)), dtype=np.uint8)
                if label == 1:
                    cv2.circle(image, (7, 8), 3, (20, 20, 20), -1)
                    cv2.circle(image, (16, 8), 3, (20, 20, 20), -1)
                    cv2.line(image, (8, 17), (16, 17), (30, 30, 30), 2)
                else:
                    cv2.rectangle(image, (4, 12), (19, 20), (20, 20, 20), -1)
                name = f'images/{split}_{label}_{j}.png'
                write_image(root/name, image)
                row = dict(image=name, source_id=name, bbox=[0, 0, 24, 24], label=label, split=split)
                if label == 1:
                    row.update(landmarks=(template+rng.normal(0, 0.2, (28, 2))).tolist(), visibility=[1]*28)
                rows.append(row)
    write_json(root/'manifest.json', rows)
    train(root/'manifest.json', root/'models', synthetic=True)
    detector = AnimeFaceDetector(root/'models')
    test = next(r for r in rows if r['split'] == 'test' and r['label'] == 1)
    image = read_image(root/test['image'])
    prediction = detector.detect(image)
    assert prediction, 'Expected at least one detection on the synthetic held-out fixture'
    assert np.asarray(prediction[0]['landmarks']).shape == (28, 2)
    assert all(np.isfinite(p['score']) for p in prediction)
    assert detector.detect(np.zeros((12, 12, 3), dtype=np.uint8)) == []
    # Loading exported models a second time must preserve outputs exactly.
    assert AnimeFaceDetector(root/'models').detect(image) == prediction
    write_json(root/'check.json', dict(synthetic=True, purpose='integration only; not coursework evaluation',
        detections=prediction, nme_bbox_diagonal=nme(prediction[0]['landmarks'], test['landmarks'], test['visibility'], np.hypot(24, 24))))
    print('PASS: synthetic training, export/reload, detection, 28 points, empty result, JSON')


if __name__ == '__main__':
    main()
