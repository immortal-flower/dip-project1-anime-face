# A 的数据准备主入口：读漫画/XML → 分组划分 → 裁出 pos/neg → 保存图片与清单。
"""Build a reviewable Manga109 detector starter set, split by book/series first."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from src.data_io import read_image, write_image, write_json
from src.data_preparation import book_group, split_groups, stable_rng, sample_background, audit_rows


# 将 XML 人脸框转换为内部 xyxy 格式，保守包含最大坐标并限制在图像内。
def xml_box(node, width, height):
    # Conservatively interpret XML maxima as inclusive; internal boxes are exclusive.
    box = [max(0, int(node.get('xmin'))), max(0, int(node.get('ymin'))),
           min(width, int(node.get('xmax'))+1), min(height, int(node.get('ymax'))+1)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'Invalid XML box: {node.attrib}')
    return box


# 一次数据准备：先锁定来源划分，再采样、缩放、保存和输出溯源记录。
def prepare(root, output, seed=42, pages_per_book=6, negatives=6, positives=4):
    root, output = Path(root).resolve(), Path(output).resolve()
    xml_files = sorted((root/'annotations').glob('*.xml'))
    if not xml_files or not (root/'images').is_dir():
        raise ValueError('Use the dataset directory containing annotations/ and images/')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory to preserve previous data versions')
    output.mkdir(parents=True, exist_ok=True)
    # 先分组再裁图：同一本书及可识别同系列不会跨训练和测试。
    groups = split_groups([book_group(p.stem) for p in xml_files], seed)
    assignments = {p.stem: groups[book_group(p.stem)] for p in xml_files}
    write_json(output/'book_splits.json', dict(seed=seed, group_policy='series suffix volNN grouped; otherwise book',
                                              groups=groups, books=assignments))
    rows, full_pages, selected_xml = [], [], hashlib.sha256()
    seen, duplicate_skips = set(), 0
    contact = {-1: [], 1: []}
    for index, xml_path in enumerate(xml_files):
        book, split = xml_path.stem, assignments[xml_path.stem]
        selected_xml.update(xml_path.name.encode()+xml_path.read_bytes())
        # XML 已含人工制作的人脸位置；这里读取已有标注，不运行检测模型。
        pages = list(ET.parse(xml_path).getroot().findall('./pages/page'))
        # Use annotated interior pages for the first batch, not unlabelled covers.
        pages = [p for p in pages if p.findall('frame') and p.findall('face')]
        stable_rng(seed, book).shuffle(pages)
        for page in pages[:pages_per_book]:
            page_index = int(page.get('index'))
            image_path = root/'images'/book/f'{page_index:03d}.jpg'
            image = read_image(image_path)
            height, width = image.shape[:2]
            if (width, height) != (int(page.get('width')), int(page.get('height'))):
                raise ValueError(f'Image/annotation version or size mismatch: {image_path}')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # faces 收集本页全部人脸框，排除背景采样时不能只用选中的四张脸。
            faces = [xml_box(n, width, height) for n in page.findall('face')]
            page_key = f'manga109:{book}:{page_index:03d}'
            full_pages.append(dict(image=os.path.relpath(image_path, output).replace('\\','/'),
                                   source_id=f'manga109:{book_group(book)}', page_id=page_key, split=split,
                                   bboxes=faces, annotation_status='dataset_xml', width=width, height=height))
            rng = stable_rng(seed, page_key)
            # 正样本从已有标注中选取，先排除原始尺寸过小的人脸。
            pos = [box for box in faces if min(box[2]-box[0],box[3]-box[1]) >= 24]
            rng.shuffle(pos)
            # 负样本只得到裁剪坐标；真正裁剪和保存由下面的共同循环完成。
            neg = sample_background(gray, faces, rng, negatives)
            for label, boxes in [(1, pos[:positives]), (-1, neg)]:
                for k, box in enumerate(boxes):
                    x1,y1,x2,y2 = box
                    # Store small patches for the current loader, avoiding repeated full-page RAM use.
                    # 数组顺序为 [行y, 列x]，先裁出原图区域，再统一缩成检测窗口大小。
                    patch = cv2.resize(image[y1:y2,x1:x2], (24,24), interpolation=cv2.INTER_AREA)
                    digest = hashlib.sha256(patch.tobytes()).hexdigest()
                    if digest in seen:
                        duplicate_skips += 1
                        continue
                    seen.add(digest)
                    # label=1 表示人脸，label=-1 表示背景；文件名帮助人阅读，程序以清单标签为准。
                    suffix = 'pos' if label == 1 else 'neg'
                    relative = f'patches/{book}/{page_index:03d}_{suffix}_{k:02d}.png'
                    write_image(output/relative, patch)
                    # 一行清单对应一张小图：既记录小图位置，也保留原图坐标方便回查。
                    rows.append(dict(image=relative, source_id=f'manga109:{book_group(book)}',
                        page_id=page_key, split=split, label=label, bbox=[0,0,24,24],
                        source_image=os.path.relpath(image_path, output).replace('\\','/'), source_bbox=box,
                        pixel_sha256=digest, dataset='Manga109-v2026',
                        annotation_status='dataset_face_box' if label==1 else 'sampled_background_unreviewed',
                        review_status='needs_review'))
                    if len(contact[label]) < 48:
                        contact[label].append((patch, f'{book[:14]}:{page_index}:{k}'))
        if (index+1) % 10 == 0:
            print(f'Prepared {index+1}/{len(xml_files)} books, {len(rows)} patches', flush=True)
    audit_rows(rows)
    write_json(output/'manifest.json', rows)
    # 整页评估必须使用该页所有人脸框，而不能只用抽出来的正样本。
    write_json(output/'pages.json', full_pages)
    with (output/'review.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=['image','split','label','review_status','note'])
        writer.writeheader()
        writer.writerows({**{k:r[k] for k in ('image','split','label','review_status')},'note':''} for r in rows)
    for label, items in contact.items():
        if not items:
            continue
        tiles = []
        for patch, title in items:
            tile = np.full((108,128,3),255,np.uint8)
            tile[:96,16:112] = cv2.resize(patch,(96,96),interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile,title,(2,105),cv2.FONT_HERSHEY_SIMPLEX,.26,(0,0,0),1)
            tiles.append(tile)
        while len(tiles)%8:
            tiles.append(np.full((108,128,3),255,np.uint8))
        write_image(output/f'preview_{"positive" if label==1 else "negative"}.jpg',
                    np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]))
    report = dict(schema_version=1, seed=seed, dataset='Manga109-v2026',
        annotation_sha256=selected_xml.hexdigest(), books=len(xml_files), groups=len(groups),
        book_counts=dict(Counter(assignments.values())), pages=len(full_pages),
        pages_per_book=pages_per_book, requested_negatives_per_page=negatives,
        requested_positives_per_page=positives, negative_face_margin=.15, negative_min_gray_std=12,
        xml_maxima_policy='inclusive interpreted conservatively as max+1',
        split_counts={s:{'positive':sum(r['split']==s and r['label']==1 for r in rows),
                         'negative':sum(r['split']==s and r['label']==-1 for r in rows)} for s in ('train','val','test')},
        total_patches=len(rows), duplicate_patch_skips=duplicate_skips,
        status='starter set; human review pending; no landmarks',
        c_annotations='not included; preserve existing C splits when supplied')
    write_json(output/'report.json',report)
    print(report)


# 命令行入口：读取参数、调用主要处理函数，并将结果保存到指定位置。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pages-per-book', type=int, default=6)
    parser.add_argument('--negatives-per-page', type=int, default=6)
    parser.add_argument('--positives-per-page', type=int, default=4)
    args = parser.parse_args()
    if min(args.pages_per_book,args.negatives_per_page,args.positives_per_page) < 1:
        parser.error('Sample counts must be positive')
    prepare(args.root,args.output,args.seed,args.pages_per_book,args.negatives_per_page,args.positives_per_page)


if __name__ == '__main__':
    main()
