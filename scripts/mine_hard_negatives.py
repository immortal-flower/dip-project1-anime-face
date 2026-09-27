"""Mine detector false positives from training manga pages only."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image, write_json
from src.detection_metrics import iou
from src.sliding_window import scan_image


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path, parent):
    return Path(os.path.relpath(Path(path).resolve(), Path(parent).resolve())).as_posix()


def _load_json_list(path, description):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, list) or not value:
        raise ValueError(f'{description} must be a nonempty JSON list')
    return value


def _validate_manifest_rows(rows):
    groups = {}
    for index, row in enumerate(rows):
        required = {'image', 'source_id', 'split', 'label', 'bbox'}
        if not required <= set(row):
            raise ValueError(f'Manifest row {index} is missing required fields')
        if row['split'] not in ('train', 'val', 'test') or row['label'] not in (-1, 1):
            raise ValueError(f'Manifest row {index} has an invalid split or label')
        source = row['source_id']
        if groups.setdefault(source, row['split']) != row['split']:
            raise ValueError(f'Data leakage across manifest splits: {source}')


def _validate_pages(rows):
    page_ids, groups = set(), {}
    for index, row in enumerate(rows):
        required = {'image', 'source_id', 'page_id', 'split', 'bboxes'}
        if not required <= set(row):
            raise ValueError(f'Page row {index} is missing required fields')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Page row {index} has an invalid split')
        if not all(isinstance(row[key], str) and row[key] for key in ('image', 'source_id', 'page_id')):
            raise ValueError(f'Page row {index} has an invalid identifier or image path')
        if row['page_id'] in page_ids:
            raise ValueError(f'Duplicate page_id: {row["page_id"]}')
        page_ids.add(row['page_id'])
        source = row['source_id']
        if groups.setdefault(source, row['split']) != row['split']:
            raise ValueError(f'Data leakage across page splits: {source}')
        if not isinstance(row['bboxes'], list):
            raise ValueError(f'Page row {index} bboxes must be a list')
        for box in row['bboxes']:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in box)
                or not (box[0] < box[2] and box[1] < box[3])
            ):
                raise ValueError(f'Page row {index} contains an invalid xyxy bbox')


def _preflight_train_pages(pages, pages_path):
    prepared, missing = [], []
    for page in pages:
        if page['split'] != 'train':
            continue
        image_path = (Path(pages_path).parent / page['image']).resolve()
        if not image_path.is_file():
            missing.append(str(image_path))
        else:
            prepared.append((page, image_path))
    if not prepared and not missing:
        raise ValueError('pages.json contains no training pages')
    if missing:
        example = missing[0]
        raise FileNotFoundError(
            f'{len(missing)} training page image(s) are missing; first: {example}. '
            'Install the full Manga109 image tree referenced by pages.json before mining.'
        )
    return prepared


def _clip_box(box, width, height):
    x1 = max(0, min(width, int(np.floor(box[0]))))
    y1 = max(0, min(height, int(np.floor(box[1]))))
    x2 = max(0, min(width, int(np.ceil(box[2]))))
    y2 = max(0, min(height, int(np.ceil(box[3]))))
    return [x1, y1, x2, y2]


def _existing_keys(rows):
    keys = set()
    for row in rows:
        if row.get('hard_negative') and 'page_id' in row and 'origin_bbox' in row:
            keys.add((row['page_id'], tuple(row['origin_bbox'])))
    return keys


def _select_pages(train_pages, max_pages, strategy):
    if strategy == 'sequential':
        selected = train_pages
    elif strategy == 'source-spread':
        groups = {}
        for item in train_pages:
            groups.setdefault(item[0]['source_id'], []).append(item)
        selected = []
        round_index = 0
        while True:
            added = False
            for source_rows in groups.values():
                if round_index < len(source_rows):
                    selected.append(source_rows[round_index])
                    added = True
            if not added:
                break
            round_index += 1
    else:
        raise ValueError(f'Unknown page_selection: {strategy}')
    return selected[:max_pages] if max_pages is not None else selected


def _portable_original_rows(rows, manifest_path, output_path):
    portable = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != '_image'}
        item['image'] = _relative_path(Path(manifest_path).parent / row['image'], output_path)
        if 'source_image' in item:
            item['source_image'] = _relative_path(
                Path(manifest_path).parent / row['source_image'], output_path
            )
        portable.append(item)
    return portable


def _write_gallery(path, crops, columns=4):
    if not crops:
        return None
    tile = 96
    rows = int(np.ceil(len(crops) / columns))
    canvas = np.full((rows * tile, columns * tile, 3), 255, dtype=np.uint8)
    for index, (crop, score) in enumerate(crops):
        cell = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_NEAREST)
        cv2.putText(cell, f'{score:.2f}', (3, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, (0, 0, 255), 1, cv2.LINE_AA)
        y, x = divmod(index, columns)
        canvas[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = cell
    write_image(path, canvas)
    return Path(path).name


def mine_hard_negatives(
    manifest,
    pages,
    model_dir,
    output,
    score_threshold=0.0,
    max_face_iou=0.3,
    max_per_page=50,
    max_total=2000,
    mining_round=1,
    gallery_size=12,
    step=None,
    scale_factor=None,
    nms_threshold=None,
    pre_nms_limit=None,
    max_pages=None,
    page_selection='sequential',
    skip_mined_pages=False,
):
    """Scan train pages and export non-face detections as 24x24 negatives."""
    if not 0 <= max_face_iou <= 1:
        raise ValueError('max_face_iou must be in [0, 1]')
    if max_per_page < 1 or max_total < 1 or mining_round < 1 or gallery_size < 0:
        raise ValueError('limits/round must be positive and gallery_size nonnegative')
    if max_pages is not None and max_pages < 1:
        raise ValueError('max_pages must be positive when supplied')
    if pre_nms_limit is not None and pre_nms_limit < 1:
        raise ValueError('pre_nms_limit must be positive when supplied')

    manifest, pages = Path(manifest), Path(pages)
    model_dir, output = Path(model_dir), Path(output)
    base_rows = _load_json_list(manifest, 'Manifest')
    page_rows = _load_json_list(pages, 'pages.json')
    _validate_manifest_rows(base_rows)
    _validate_pages(page_rows)
    train_pages = _preflight_train_pages(page_rows, pages)
    train_pages_available = len(train_pages)
    previously_mined_pages = {
        row['page_id'] for row in base_rows
        if row.get('hard_negative') and row.get('page_id')
    }
    skipped_existing_pages = 0
    if skip_mined_pages:
        before = len(train_pages)
        train_pages = [
            item for item in train_pages
            if item[0]['page_id'] not in previously_mined_pages
        ]
        skipped_existing_pages = before - len(train_pages)
    train_pages = _select_pages(train_pages, max_pages, page_selection)

    detector_path = model_dir / 'detector.json'
    config_path = model_dir / 'config.json'
    if not detector_path.is_file() or not config_path.is_file():
        raise FileNotFoundError('model_dir must contain detector.json and config.json')
    model = json.loads(detector_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    scan_config = dict(config)
    if step is not None:
        scan_config['step'] = step
    if scale_factor is not None:
        scan_config['scale_factor'] = scale_factor
    if nms_threshold is not None:
        scan_config['nms_threshold'] = nms_threshold
    if pre_nms_limit is not None:
        scan_config['pre_nms_limit'] = pre_nms_limit
    if not isinstance(scan_config.get('step'), int) or scan_config['step'] < 1:
        raise ValueError('effective scan step must be a positive integer')
    if scan_config.get('scale_factor', 0) <= 1:
        raise ValueError('effective scale_factor must exceed 1')
    if not 0 <= scan_config.get('nms_threshold', -1) <= 1:
        raise ValueError('effective nms_threshold must be in [0, 1]')
    if not model.get('stages'):
        raise ValueError('Detector model has no cascade stages')

    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / 'images'
    existing = _existing_keys(base_rows)
    mined, gallery, page_logs = [], [], []
    overlap_rejected = duplicate_rejected = below_score = 0
    started = time.perf_counter()

    for page, image_path in train_pages:
        if len(mined) >= max_total:
            break
        page_start = time.perf_counter()
        image = read_image(image_path)
        height, width = image.shape[:2]
        if 'width' in page and int(page['width']) != width:
            raise ValueError(f'Width mismatch for page {page["page_id"]}')
        if 'height' in page and int(page['height']) != height:
            raise ValueError(f'Height mismatch for page {page["page_id"]}')
        for box in page['bboxes']:
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f'Out-of-bounds face bbox on page {page["page_id"]}')

        predictions, scan_log = scan_image(image, model, scan_config)
        page_mined = 0
        for prediction in sorted(predictions, key=lambda item: item['score'], reverse=True):
            if page_mined >= max_per_page or len(mined) >= max_total:
                break
            score = float(prediction['score'])
            if score < score_threshold:
                below_score += 1
                continue
            overlap = max((iou(prediction['bbox'], box) for box in page['bboxes']), default=0.0)
            if overlap > max_face_iou:
                overlap_rejected += 1
                continue
            crop_box = _clip_box(prediction['bbox'], width, height)
            if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
                continue
            key = (page['page_id'], tuple(crop_box))
            if key in existing:
                duplicate_rejected += 1
                continue
            existing.add(key)
            crop = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
            resized = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
            name = f'hnm-r{mining_round:02d}-{len(mined):06d}.png'
            write_image(image_dir / name, resized)
            row = dict(
                image=f'images/{name}',
                source_id=page['source_id'],
                page_id=page['page_id'],
                split='train',
                label=-1,
                bbox=[0, 0, 24, 24],
                hard_negative=True,
                mining_round=mining_round,
                mined_from=_relative_path(image_path, output),
                origin_bbox=crop_box,
                detector_bbox=[float(value) for value in prediction['bbox']],
                detector_score=score,
                max_face_iou=float(overlap),
                review_status='needs_review',
                annotation_status='detector_false_positive_candidate',
            )
            mined.append(row)
            page_mined += 1
            if len(gallery) < gallery_size:
                gallery.append((resized, score))

        page_logs.append(dict(
            page_id=page['page_id'], source_id=page['source_id'], image=page['image'],
            ground_truth_faces=len(page['bboxes']), detections=len(predictions),
            mined=page_mined, windows=sum(level['windows'] for level in scan_log),
            scan=scan_log, seconds=time.perf_counter() - page_start,
        ))

    write_json(output / 'mined_manifest.json', mined)
    augmented = _portable_original_rows(base_rows, manifest, output) + mined
    write_json(output / 'augmented_manifest.json', augmented)
    gallery_file = _write_gallery(output / 'gallery.png', gallery)
    summary = dict(
        schema_version=1,
        train_only=True,
        mining_round=mining_round,
        source_manifest=str(manifest.resolve()),
        source_manifest_sha256=_sha256(manifest),
        source_pages=str(pages.resolve()),
        source_pages_sha256=_sha256(pages),
        detector_sha256=_sha256(detector_path),
        config_sha256=_sha256(config_path),
        score_threshold=score_threshold,
        max_face_iou=max_face_iou,
        max_per_page=max_per_page,
        max_total=max_total,
        train_pages_available=train_pages_available,
        selected_train_pages=len(train_pages),
        max_pages=max_pages,
        page_selection=page_selection,
        skip_mined_pages=skip_mined_pages,
        skipped_existing_pages=skipped_existing_pages,
        scan_config=dict(
            step=scan_config['step'],
            scale_factor=scan_config['scale_factor'],
            nms_threshold=scan_config['nms_threshold'],
            pre_nms_limit=scan_config.get('pre_nms_limit'),
        ),
        scanned_pages=len(page_logs),
        detections=sum(item['detections'] for item in page_logs),
        mined=len(mined),
        overlap_rejected=overlap_rejected,
        duplicate_rejected=duplicate_rejected,
        below_score=below_score,
        gallery=gallery_file,
        seconds=time.perf_counter() - started,
        pages=page_logs,
    )
    write_json(output / 'mining_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--score-threshold', type=float, default=0.0)
    parser.add_argument('--max-face-iou', type=float, default=0.3)
    parser.add_argument('--max-per-page', type=int, default=50)
    parser.add_argument('--max-total', type=int, default=2000)
    parser.add_argument('--round', type=int, default=1, dest='mining_round')
    parser.add_argument('--gallery-size', type=int, default=12)
    parser.add_argument('--step', type=int)
    parser.add_argument('--scale-factor', type=float)
    parser.add_argument('--nms-threshold', type=float)
    parser.add_argument('--pre-nms-limit', type=int)
    parser.add_argument('--max-pages', type=int)
    parser.add_argument(
        '--page-selection', choices=('sequential', 'source-spread'),
        default='sequential',
    )
    parser.add_argument('--skip-mined-pages', action='store_true')
    args = parser.parse_args()
    try:
        summary = mine_hard_negatives(
            args.manifest, args.pages, args.model_dir, args.output,
            score_threshold=args.score_threshold, max_face_iou=args.max_face_iou,
            max_per_page=args.max_per_page, max_total=args.max_total,
            mining_round=args.mining_round, gallery_size=args.gallery_size,
            step=args.step, scale_factor=args.scale_factor,
            nms_threshold=args.nms_threshold,
            pre_nms_limit=args.pre_nms_limit, max_pages=args.max_pages,
            page_selection=args.page_selection,
            skip_mined_pages=args.skip_mined_pages,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f'ERROR: {error}\n')
    print(f'Mined {summary["mined"]} negatives from {summary["scanned_pages"]} train pages')


if __name__ == '__main__':
    main()
