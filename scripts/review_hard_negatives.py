"""Export hard-negative context sheets and apply explicit human decisions."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image, write_json


def _relative_path(path, parent):
    return Path(os.path.relpath(Path(path).resolve(), Path(parent).resolve())).as_posix()


def _fit(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _context(page, box):
    x1, y1, x2, y2 = box
    margin = max(x2 - x1, y2 - y1) * 2
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right = min(page.shape[1], x2 + margin)
    bottom = min(page.shape[0], y2 + margin)
    context = page[top:bottom, left:right].copy()
    cv2.rectangle(
        context, (x1 - left, y1 - top), (x2 - left - 1, y2 - top - 1),
        (0, 0, 255), max(1, round(max(context.shape[:2]) / 120)),
    )
    return context


def export_review_sheets(manifest, pages, output, items_per_sheet=12):
    manifest, pages, output = Path(manifest), Path(pages), Path(output)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    page_rows = json.loads(pages.read_text(encoding='utf-8'))
    if not rows or not isinstance(rows, list):
        raise ValueError('Hard-negative manifest must be a nonempty list')
    page_by_id = {row['page_id']: row for row in page_rows}
    if len(page_by_id) != len(page_rows):
        raise ValueError('pages.json contains duplicate page_id values')
    output.mkdir(parents=True, exist_ok=True)

    index_rows = []
    page_cache = {}
    tile_width, tile_height, columns = 360, 260, 3
    for offset in range(0, len(rows), items_per_sheet):
        batch = rows[offset:offset + items_per_sheet]
        canvas_rows = int(np.ceil(len(batch) / columns))
        canvas = np.full(
            (canvas_rows * tile_height, columns * tile_width, 3), 255,
            dtype=np.uint8,
        )
        sheet_number = offset // items_per_sheet + 1
        for local_index, row in enumerate(batch):
            review_index = offset + local_index + 1
            if row['page_id'] not in page_by_id:
                raise ValueError(f'Unknown page_id: {row["page_id"]}')
            page_row = page_by_id[row['page_id']]
            page_path = (pages.parent / page_row['image']).resolve()
            if page_path not in page_cache:
                page_cache[page_path] = read_image(page_path)
            page = page_cache[page_path]
            patch = read_image(manifest.parent / row['image'])
            context = _context(page, row['origin_bbox'])
            tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
            tile[28:248, :220] = _fit(context, 220, 220)
            tile[78:198, 230:350] = _fit(patch, 120, 120)
            cv2.putText(
                tile, f'#{review_index:03d} score={row["detector_score"]:.2f}',
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile, row['page_id'].rsplit(':', 1)[-1], (250, 220),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA,
            )
            y, x = divmod(local_index, columns)
            canvas[y * tile_height:(y + 1) * tile_height,
                   x * tile_width:(x + 1) * tile_width] = tile
            index_rows.append(dict(
                review_index=review_index,
                image=row['image'],
                page_id=row['page_id'],
                origin_bbox=row['origin_bbox'],
                detector_score=row['detector_score'],
                max_face_iou=row['max_face_iou'],
                sheet=f'review-sheet-{sheet_number:02d}.png',
            ))
        write_image(output / f'review-sheet-{sheet_number:02d}.png', canvas)
    write_json(output / 'review_index.json', index_rows)
    return index_rows


def apply_decisions(manifest, decisions, output, base_manifest=None):
    manifest, decisions, output = Path(manifest), Path(decisions), Path(output)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    decision_rows = json.loads(decisions.read_text(encoding='utf-8'))
    if not isinstance(decision_rows, list):
        raise ValueError('Decisions must be a JSON list')
    by_index = {}
    for decision in decision_rows:
        index = decision.get('review_index')
        status = decision.get('review_status')
        if index in by_index or status not in ('accepted', 'rejected'):
            raise ValueError('Decisions require unique indices and accepted/rejected status')
        by_index[index] = decision
    expected = set(range(1, len(rows) + 1))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(f'Decisions must cover every item; missing={missing[:5]} extra={extra[:5]}')

    reviewed, accepted, rejected = [], [], []
    for index, row in enumerate(rows, 1):
        decision = by_index[index]
        item = dict(row)
        item['image'] = _relative_path(manifest.parent / row['image'], output)
        item['review_status'] = decision['review_status']
        item['review_method'] = decision.get(
            'review_method', 'unspecified_visual_review'
        )
        item['review_note'] = decision.get('note', 'visual review')
        reviewed.append(item)
        (accepted if item['review_status'] == 'accepted' else rejected).append(item)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'reviewed_manifest.json', reviewed)
    write_json(output / 'accepted_manifest.json', accepted)
    write_json(output / 'rejected_manifest.json', rejected)
    accepted_augmented = None
    if base_manifest is not None:
        base_manifest = Path(base_manifest)
        base_rows = json.loads(base_manifest.read_text(encoding='utf-8'))
        accepted_augmented = []
        for row in base_rows:
            item = dict(row)
            item['image'] = _relative_path(base_manifest.parent / row['image'], output)
            if 'source_image' in item:
                item['source_image'] = _relative_path(
                    base_manifest.parent / row['source_image'], output
                )
            accepted_augmented.append(item)
        accepted_augmented.extend(accepted)
        write_json(output / 'accepted_augmented_manifest.json', accepted_augmented)
    summary = dict(
        reviewed=len(reviewed), accepted=len(accepted), rejected=len(rejected),
        source_manifest=str(manifest.resolve()), decisions=str(decisions.resolve()),
        accepted_augmented=(len(accepted_augmented)
                            if accepted_augmented is not None else None),
        review_methods=sorted({item['review_method'] for item in reviewed}),
        complete=True,
    )
    write_json(output / 'review_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--pages', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--items-per-sheet', type=int, default=12)
    parser.add_argument('--decisions')
    parser.add_argument('--base-manifest')
    args = parser.parse_args()
    if args.items_per_sheet < 1:
        parser.error('--items-per-sheet must be positive')
    index = export_review_sheets(
        args.manifest, args.pages, args.output, args.items_per_sheet
    )
    print(f'Exported {len(index)} candidates for visual review')
    if args.decisions:
        summary = apply_decisions(
            args.manifest, args.decisions, args.output, args.base_manifest
        )
        print(f'Applied decisions: {summary}')


if __name__ == '__main__':
    main()
