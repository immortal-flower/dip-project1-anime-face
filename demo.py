# 演示入口：读输入图、调用检测与回归、绘制结果，并输出 JSON。
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.data_io import read_image, write_image, write_json
from src.detector import AnimeFaceDetector, LandmarkRegressor


# 命令行入口：读取参数、调用主要处理函数，并将结果保存到指定位置。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--boxes-json', help='Optional JSON boxes; bypasses the face detector')
    args = parser.parse_args()
    image = read_image(args.image)
    if args.boxes_json:
        raw = json.loads(Path(args.boxes_json).read_text(encoding='utf-8'))
        boxes = raw.get('faces', raw.get('boxes', [])) if isinstance(raw, dict) else raw
        regressor = LandmarkRegressor(args.model_dir)
        results = regressor.predict(image, boxes)
        scan, config, mode = [], regressor.config, 'supplied_boxes'
    else:
        detector = AnimeFaceDetector(args.model_dir)
        results = detector.detect(image)
        scan, config, mode = detector.last_scan_log, detector.config, 'detector'
    canvas = image.copy()
    for item in results:
        x1, y1 = np.floor(item['bbox'][:2]).astype(int)
        x2, y2 = np.ceil(item['bbox'][2:]).astype(int) - 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if 'score' in item:
            cv2.putText(canvas, f"{float(item['score']):.3f}", (x1, max(12, y1-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        for x, y in item['landmarks']:
            cv2.circle(canvas, (round(x), round(y)), 1, (0, 0, 255), -1)
    write_image(args.output, canvas)
    write_json(Path(args.output).with_suffix('.json'), dict(
        synthetic=config.get('synthetic', False), mode=mode,
        landmark_order=config.get('landmark_order'), faces=results, scan=scan))
    print(f'Saved {len(results)} faces to {args.output}; mode={mode}; '
          f'synthetic={config.get("synthetic", False)}')


if __name__ == '__main__':
    main()
