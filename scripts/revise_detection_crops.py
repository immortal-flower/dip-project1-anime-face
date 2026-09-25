"""A：基于已有版本扩展正例裁剪，保持样本身份、负例像素和固定划分。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
from src.data_io import read_image, write_image, write_json
from src.data_preparation import positive_crop_box, intersects, audit_rows


def revise(source, output, margin=.10):
    """所有修改写入新版本；pages 的原始真值框不会随正例扩边而改变。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new dataset version directory')
    if not 0 <= margin <= .5:
        raise ValueError('Positive margin must be in [0,0.5]')
    rows = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    pages = json.loads((source/'pages.json').read_text(encoding='utf-8'))
    lookup = {page['page_id']:page for page in pages}
    output.mkdir(parents=True)
    revised, decisions = [], []
    cached_path, original = None, None
    for old in rows:
        row = dict(old)
        path = (source/old['source_image']).resolve()
        row['source_image'] = os.path.relpath(path, output).replace('\\','/')
        destination = output/row['image']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row['label'] == 1:
            if path != cached_path:
                original, cached_path = read_image(path), path
            truth = list(old.get('source_gt_bbox', old['source_bbox']))
            others = [b for b in lookup[row['page_id']]['bboxes'] if b != truth]
            crop, effective = positive_crop_box(truth, original.shape[1], original.shape[0], margin, others)
            x1,y1,x2,y2 = crop
            patch = cv2.resize(original[y1:y2,x1:x2], (24,24), interpolation=cv2.INTER_AREA)
            write_image(destination, patch)
            row.update(source_gt_bbox=truth, source_bbox=crop,
                       positive_margin_requested=margin, positive_margin_effective=effective,
                       pixel_sha256=hashlib.sha256(patch.tobytes()).hexdigest(),
                       review_status='needs_review')
            # 旧裁剪的接受结论不能自动继承到不同像素的新裁剪。
            row.pop('sample_review_note', None)
            decisions.append(dict(image=row['image'], source_gt_bbox=truth, crop_bbox=crop,
                                  effective_margin=effective,
                                  preexisting_neighbor_overlap=any(intersects(truth,b) for b in others)))
        else:
            shutil.copy2(source/old['image'], destination)
        revised.append(row)
    audit_rows(revised)
    for page in pages:
        page['image'] = os.path.relpath((source/page['image']).resolve(), output).replace('\\','/')
    write_json(output/'manifest.json', revised)
    write_json(output/'pages.json', pages)
    shutil.copy2(source/'book_splits.json', output/'book_splits.json')
    counts = {s:dict(Counter('positive' if r['label']==1 else 'negative' for r in revised if r['split']==s))
              for s in ('train','val','test')}
    report = dict(parent_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                  positive_margin=margin, total=len(revised), split_counts=counts,
                  reduced_margin_count=sum(d['effective_margin']<margin for d in decisions),
                  preexisting_neighbor_overlap_count=sum(d['preexisting_neighbor_overlap'] for d in decisions),
                  original_ground_truth_unchanged=True, negatives_unchanged=True,
                  status='expanded crop candidate version; visual quality review pending')
    write_json(output/'crop_changes.json', decisions)
    write_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--positive-margin', type=float, default=.10)
    args = parser.parse_args()
    print(revise(args.source, args.output, args.positive_margin))


if __name__ == '__main__':
    main()
