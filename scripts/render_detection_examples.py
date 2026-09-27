"""Render representative full-page detector results without changing metrics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image
from src.detection_metrics import iou


def _annotate(image, truth, predictions, title):
    canvas = image.copy()
    unmatched = set(range(len(truth)))
    matched_predictions = set()
    for prediction_index, prediction in enumerate(predictions):
        if not unmatched:
            break
        truth_index = max(
            unmatched, key=lambda index: iou(prediction['bbox'], truth[index])
        )
        if iou(prediction['bbox'], truth[truth_index]) >= 0.5:
            unmatched.remove(truth_index)
            matched_predictions.add(prediction_index)
    for box in truth:
        x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 3)
    for index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = map(lambda value: int(round(value)), prediction['bbox'])
        color = (255, 120, 0) if index in matched_predictions else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"{prediction['score']:.1f}", (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(
        canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .72,
        (20, 20, 20), 2, cv2.LINE_AA,
    )
    return canvas


def render_examples(pages_path, result_path, output, count=6):
    pages_path, result_path, output = map(Path, (pages_path, result_path, output))
    pages = json.loads(pages_path.read_text(encoding='utf-8'))
    result = json.loads(result_path.read_text(encoding='utf-8'))
    by_id = {row['page_id']: row for row in pages}
    per_page = result['metrics']['per_page']
    worst = sorted(per_page, key=lambda page_id: (-per_page[page_id]['fp'], page_id))
    best = sorted(per_page, key=lambda page_id: (-per_page[page_id]['tp'], page_id))
    selected = []
    for page_id in worst[:max(1, count//2)] + best:
        if page_id not in selected:
            selected.append(page_id)
        if len(selected) == count:
            break

    output.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    index = []
    for number, page_id in enumerate(selected, 1):
        page = by_id[page_id]
        metrics = per_page[page_id]
        image = read_image((pages_path.parent/page['image']).resolve())
        title = (
            f"{page_id}  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
        )
        annotated = _annotate(
            image, page['bboxes'], result['predictions'][page_id], title
        )
        path = output/f'example-{number:02d}.png'
        write_image(path, annotated)
        scale = 520/annotated.shape[1]
        thumb = cv2.resize(
            annotated, (520, max(1, int(round(annotated.shape[0]*scale))))
        )
        thumbnails.append(thumb)
        index.append(dict(
            page_id=page_id, file=path.name, source_id=page['source_id'], **metrics
        ))

    width = 1040
    row_heights = []
    for offset in range(0, len(thumbnails), 2):
        row_heights.append(max(item.shape[0] for item in thumbnails[offset:offset+2]))
    sheet = np.full((sum(row_heights), width, 3), 255, dtype=np.uint8)
    top = 0
    for row_index, offset in enumerate(range(0, len(thumbnails), 2)):
        for column, thumb in enumerate(thumbnails[offset:offset+2]):
            sheet[top:top+thumb.shape[0], column*520:(column+1)*520] = thumb
        top += row_heights[row_index]
    write_image(output/'examples-sheet.png', sheet)
    (output/'index.json').write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args()
    result = render_examples(args.pages, args.result, args.output, args.count)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
