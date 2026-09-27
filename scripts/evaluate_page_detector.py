"""Run a frozen cascade on a fixed full-page subset and compute detection metrics."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from src.data_io import read_image, write_json
from src.detection_metrics import evaluate_dataset
from src.sliding_window import scan_image


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_spread(rows, limit):
    groups = {}
    for row in rows:
        groups.setdefault(row['source_id'], []).append(row)
    selected = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for source_rows in groups.values():
            if round_index < len(source_rows):
                selected.append(source_rows[round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return selected


def evaluate_pages(
    pages_path, model_dir, output, split='test', max_pages=6,
    step=12, scale_factor=1.5, nms_threshold=0.3,
    pre_nms_limit=500, iou_threshold=0.5, scan_batch_size=4096,
    score_threshold=None,
):
    pages_path, model_dir, output = Path(pages_path), Path(model_dir), Path(output)
    pages = json.loads(pages_path.read_text(encoding='utf-8'))
    candidates = [row for row in pages if row['split'] == split]
    full_split = max_pages is None or max_pages >= len(candidates)
    selected = candidates if full_split else _source_spread(candidates, max_pages)
    if not full_split and len(selected) != max_pages:
        raise ValueError(f'Requested {max_pages} pages but selected {len(selected)}')
    if not full_split and len({row['source_id'] for row in selected}) != len(selected):
        raise ValueError('Selected subset does not have one distinct source per page')

    detector_path = model_dir/'detector.json'
    config_path = model_dir/'config.json'
    model = json.loads(detector_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.update(
        step=step, scale_factor=scale_factor, nms_threshold=nms_threshold,
        pre_nms_limit=pre_nms_limit, scan_batch_size=scan_batch_size,
    )
    if score_threshold is not None:
        config['score_threshold'] = score_threshold
    effective_score_threshold = config.get('score_threshold')
    predictions, page_logs = {}, []
    started = time.perf_counter()
    for page in selected:
        page_start = time.perf_counter()
        image_path = (pages_path.parent/page['image']).resolve()
        image = read_image(image_path)
        detections, scan_log = scan_image(image, model, config)
        summary = scan_log[-1].get('scan_summary', {}) if scan_log else {}
        detections_before_score_filter = summary.get(
            'detections_after_nms', len(detections)
        )
        predictions[page['page_id']] = detections
        page_logs.append(dict(
            page_id=page['page_id'], source_id=page['source_id'],
            ground_truth_faces=len(page['bboxes']), detections=len(detections),
            detections_before_score_filter=detections_before_score_filter,
            windows=sum(level['windows'] for level in scan_log),
            seconds=time.perf_counter() - page_start, scan=scan_log,
        ))
        print(json.dumps(dict(
            page=len(page_logs), total=len(selected), page_id=page['page_id'],
            detections=len(detections), windows=page_logs[-1]['windows'],
            seconds=round(page_logs[-1]['seconds'], 3),
        ), ensure_ascii=False), flush=True)

    result = dict(
        schema_version=1,
        purpose=(
            'complete frozen split evaluation' if full_split else
            'fixed full-page subset evaluation; not the complete split'
        ),
        split=split,
        selection='all' if full_split else 'source-spread',
        requested_pages=len(candidates) if full_split else max_pages,
        selected_page_ids=[row['page_id'] for row in selected],
        selected_source_ids=[row['source_id'] for row in selected],
        pages_sha256=_sha256(pages_path),
        detector_sha256=_sha256(detector_path),
        config_sha256=_sha256(config_path),
        scan_config=dict(
            step=step, scale_factor=scale_factor,
            nms_threshold=nms_threshold, pre_nms_limit=pre_nms_limit,
            scan_batch_size=scan_batch_size,
            score_threshold=effective_score_threshold,
        ),
        iou_threshold=iou_threshold,
        metrics=evaluate_dataset(selected, predictions, iou_threshold),
        predictions=predictions,
        pages=page_logs,
        seconds=time.perf_counter() - started,
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--max-pages', type=int, default=6)
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--scale-factor', type=float, default=1.5)
    parser.add_argument('--nms-threshold', type=float, default=0.3)
    parser.add_argument('--pre-nms-limit', type=int, default=500)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--scan-batch-size', type=int, default=4096)
    parser.add_argument('--score-threshold', type=float)
    args = parser.parse_args()
    if args.max_pages < 1:
        parser.error('--max-pages must be positive')
    result = evaluate_pages(
        args.pages, args.model_dir, args.output, args.split, args.max_pages,
        args.step, args.scale_factor, args.nms_threshold,
        args.pre_nms_limit, args.iou_threshold, args.scan_batch_size,
        args.score_threshold,
    )
    print(json.dumps(result['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    main()
