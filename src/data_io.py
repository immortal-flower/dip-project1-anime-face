# A：负责图像读取、清单校验和 24×24 样本裁剪；其他模块通过这里读取统一格式。
"""A: portable image IO and strict sample-manifest validation."""
import json
from pathlib import Path
import cv2
import numpy as np


# 用字节解码读取 BGR 图片，兼容中文路径；读不到有效图像时直接报错。
def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


# 根据文件后缀编码图片，再写入磁盘；自动创建父目录。
def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


# 以 UTF-8 保存清单或配置；禁止写入 JSON 不支持的 NaN/无穷大。
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


# 逐条检查清单和图片，把加载结果放入 _image；同一 source_id 不能跨集合。
def load_manifest(path):
    """JSON list; image paths relative to manifest; bbox uses exclusive x2/y2."""
    path = Path(path)
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON list')
    groups = {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or row['label'] not in (-1, 1):
            raise ValueError('Invalid split or label')
        group = row['source_id']
        if groups.setdefault(group, row['split']) != row['split']:
            raise ValueError(f'Data leakage across splits: {group}')
        image = read_image(path.parent / row['image'])
        h, w = image.shape[:2]
        x1, y1, x2, y2 = row['bbox']
        if not all(isinstance(v, int) for v in row['bbox']) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('bbox must contain in-bounds integer xyxy coordinates')
        if 'landmarks' in row:
            points = np.asarray(row['landmarks'], dtype=float)
            visible = np.asarray(row['visibility'])
            if points.shape != (28, 2) or not np.isfinite(points).all() or visible.shape != (28,) or not np.isin(visible, [0, 1]).all():
                raise ValueError('Expected 28 finite xy landmarks and 28 visibility flags')
            if 'landmark_weights' in row:
                weights = np.asarray(row['landmark_weights'], dtype=float)
                if (weights.shape != (28,) or not np.isfinite(weights).all()
                        or np.any((weights < 0) | (weights > 1))):
                    raise ValueError('Expected 28 landmark weights between 0 and 1')
        row = dict(row)
        row['_image'] = image
        yield row


# 从指定 split 收集正负样本，并统一为 24×24 灰度窗口。
def detection_samples(rows, split):
    patches, labels = [], []
    for row in rows:
        if row['split'] == split:
            x1, y1, x2, y2 = row['bbox']
            gray = cv2.cvtColor(row['_image'], cv2.COLOR_BGR2GRAY)
            patches.append(cv2.resize(gray[y1:y2, x1:x2], (24, 24)))
            labels.append(row['label'])
    if set(labels) != {-1, 1}:
        raise ValueError(f'{split} requires positive and negative detection samples')
    return patches, np.asarray(labels)
