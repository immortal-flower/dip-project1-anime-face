"""Select a frozen detection-score threshold using validation predictions only."""
import argparse
import json
from pathlib import Path

from src.data_io import write_json
from src.detection_metrics import iou


def select_threshold(pages_path, validation_result, output, iou_threshold=0.5,
                     beta=1.0):
    if beta <= 0:
        raise ValueError('beta must be positive')
    pages_path = Path(pages_path)
    validation_result = Path(validation_result)
    result = json.loads(validation_result.read_text(encoding='utf-8'))
    if result.get('split') != 'val':
        raise ValueError('Threshold selection requires a validation result, never test')
    pages = json.loads(pages_path.read_text(encoding='utf-8'))
    page_by_id = {row['page_id']: row for row in pages}
    selected_ids = result['selected_page_ids']
    selected_pages = [page_by_id[page_id] for page_id in selected_ids]
    predictions = result['predictions']
    ranked = []
    for page_order, page_id in enumerate(selected_ids):
        for prediction_order, item in enumerate(predictions[page_id]):
            ranked.append((
                -float(item['score']), page_order, prediction_order,
                page_id, item,
            ))
    ranked.sort()
    if not ranked:
        raise ValueError('Validation result contains no detections')

    unmatched = {
        row['page_id']: set(range(len(row['bboxes']))) for row in selected_pages
    }
    truth = {row['page_id']: row['bboxes'] for row in selected_pages}
    total_truth = sum(len(items) for items in truth.values())
    sweep = []
    tp = fp = processed = 0
    index = 0
    while index < len(ranked):
        threshold = -ranked[index][0]
        while index < len(ranked) and -ranked[index][0] == threshold:
            _, _, _, page_id, prediction = ranked[index]
            available = unmatched[page_id]
            if available:
                best_index = max(
                    available,
                    key=lambda truth_index: iou(
                        prediction['bbox'], truth[page_id][truth_index]
                    ),
                )
                if iou(prediction['bbox'], truth[page_id][best_index]) >= iou_threshold:
                    available.remove(best_index)
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1
            processed += 1
            index += 1
        fn = total_truth - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_truth if total_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        beta2 = beta * beta
        fbeta = ((1 + beta2) * precision * recall /
                 (beta2 * precision + recall)
                 if precision + recall else 0.0)
        sweep.append(dict(
            threshold=threshold, detections=processed,
            tp=tp, fp=fp, fn=fn,
            precision=precision, recall=recall, f1=f1, fbeta=fbeta,
        ))
    best = max(
        sweep,
        key=lambda item: (
            item['fbeta'], item['recall'], -item['fp'],
            float('-inf') if item['threshold'] is None else -item['threshold'],
        ),
    )
    output_data = dict(
        schema_version=1,
        selection_split='val',
        selection_rule=f'maximum micro F{beta:g}; ties prefer recall, fewer FP, lower threshold',
        beta=beta,
        pages=str(pages_path),
        validation_result=str(validation_result),
        selected_page_ids=selected_ids,
        iou_threshold=iou_threshold,
        selected=best,
        sweep=sweep,
    )
    write_json(output, output_data)
    return output_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--validation-result', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    result = select_threshold(
        args.pages, args.validation_result, args.output, args.iou_threshold,
        args.beta,
    )
    print(json.dumps(result['selected'], ensure_ascii=False))


if __name__ == '__main__':
    main()
