"""A：用感知指纹找近似重复候选，另存报告，不删除图片或改变划分。"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.data_io import read_image, write_json


def signature(image):
    """64位水平差分指纹与32×32灰度缩略图；指纹相似本身不是重复证明。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9,8), interpolation=cv2.INTER_AREA)
    bits = (small[:,1:] > small[:,:-1]).ravel()
    code = sum(int(value) << i for i,value in enumerate(bits))
    return code, cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)


def candidate_pairs(records, max_distance=4, max_mae=12):
    """五个不重叠位段建索引：最多4位不同必有一段相同，再精确过滤。

    已知 split 时只比较跨集合；未知 split（AnimeFace 尚未联合划分）比较全体。
    大小变化通过缩略图统一，宽高比差异超过15%会跳过。不能穷尽裁剪/旋转重复。
    """
    if not 0 <= max_distance <= 4:
        raise ValueError('This index supports Hamming distance 0..4')
    tables = [defaultdict(list) for _ in range(5)]
    pairs = []
    bands = [(0,13),(13,13),(26,13),(39,13),(52,12)]
    for i,current in enumerate(records):
        keys = [(current['code'] >> offset) & ((1<<width)-1) for offset,width in bands]
        candidates = set()
        for table,key in zip(tables,keys):
            candidates.update(table.get(key, []))
        for j in candidates:
            prior = records[j]
            if current.get('split') and current['split'] == prior.get('split'):
                continue
            if abs(np.log(current['aspect']/prior['aspect'])) > np.log(1.15):
                continue
            distance = bin(current['code'] ^ prior['code']).count('1')
            if distance > max_distance:
                continue
            mae = float(np.mean(np.abs(current['thumbnail'].astype(np.int16)-prior['thumbnail'].astype(np.int16))))
            if mae <= max_mae:
                pairs.append(dict(left=prior['image'], right=current['image'],
                                  left_split=prior.get('split'),right_split=current.get('split'),
                                  hamming=distance,gray_mae=mae,status='needs_visual_review'))
        for table,key in zip(tables,keys):
            table[key].append(i)
    return sorted(pairs,key=lambda pair:(pair['hamming'],pair['gray_mae'],pair['left'],pair['right']))


def run(items, root, output):
    """低纹理会让指纹退化；单列这些图片，不用它们宣布发现重复。"""
    root,output = Path(root),Path(output)
    if output.exists():
        raise FileExistsError('Use a new duplicate review directory')
    records,flat,errors,seen = [],[],[],{}
    exact_aliases = []
    for item in items:
        try:
            image = read_image(root/item['image'])
        except (ValueError,OSError) as error:
            errors.append(dict(image=item['image'],error=str(error))); continue
        digest = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        # 相同像素只建一个感知索引项；完全重复另外记录，不能静默漏掉跨集合重复。
        if digest in seen:
            exact_aliases.append(dict(left=seen[digest]['image'],right=item['image'],pixel_sha256=digest,
                                      left_split=seen[digest].get('split'),right_split=item.get('split')))
            continue
        seen[digest]=item
        code,thumbnail = signature(image)
        if float(thumbnail.std()) < 12:
            flat.append(item['image']); continue
        records.append(dict(image=item['image'],split=item.get('split'),code=code,
                            thumbnail=thumbnail,aspect=image.shape[1]/image.shape[0]))
        if len(records)%5000 == 0:
            print(f'Fingerprinted {len(records)} images',flush=True)
    pairs = candidate_pairs(records)
    write_json(output/'pairs.json',pairs)
    write_json(output/'flat_images.json',flat)
    write_json(output/'exact_aliases.json',exact_aliases)
    report = dict(input_count=len(items),indexed=len(records),low_texture_skipped=len(flat),
                  exact_aliases=len(exact_aliases),errors=errors,candidate_pairs=len(pairs),
                  max_hamming=4,max_gray_mae=12,max_aspect_ratio=1.15,
                  scope='cross-split if split known; all unknown-split images',
                  conclusion='candidate screening only; no automatic deduplication or split edits')
    write_json(output/'report.json',report)
    print(report,flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest')
    inputs.add_argument('--inventory',help='AnimeFace inventory.csv；完整相同图先选一个代表')
    parser.add_argument('--image-root',help='inventory中的image相对此目录')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.manifest:
        path=Path(args.manifest)
        items=json.loads(path.read_text(encoding='utf-8')); root=path.parent
    else:
        if not args.image_root:
            parser.error('--inventory requires --image-root')
        with Path(args.inventory).open(encoding='utf-8-sig',newline='') as stream:
            unique={}
            for row in csv.DictReader(stream):
                unique.setdefault(row['pixel_sha256'],row)
        items=list(unique.values()); root=Path(args.image_root)
    run(items,root,args.output)


if __name__ == '__main__':
    main()
