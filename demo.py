import argparse
from pathlib import Path
import cv2
from src.data_io import read_image, write_image, write_json
from src.detector import AnimeFaceDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    image = read_image(args.image)
    detector = AnimeFaceDetector(args.model_dir)
    results = detector.detect(image)
    canvas = image.copy()
    for item in results:
        x1, y1, x2, y2 = map(round, item['bbox'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for x, y in item['landmarks']:
            cv2.circle(canvas, (round(x), round(y)), 1, (0, 0, 255), -1)
    write_image(args.output, canvas)
    write_json(Path(args.output).with_suffix('.json'), dict(
        synthetic=detector.config.get('synthetic', False), faces=results, scan=detector.last_scan_log))
    print(f'Saved {len(results)} detections to {args.output}; synthetic={detector.config.get("synthetic", False)}')


if __name__ == '__main__':
    main()
