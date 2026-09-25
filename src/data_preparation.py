# A：数据准备的规则函数。书本分组 → 固定划分 → 背景采样 → 跨集合检查。
"""A: deterministic group splits and conservative Manga109 background sampling."""
import hashlib
import random
import re
import numpy as np


# 去掉书名末尾的 volNN，让同系列多卷归入同一组。
def book_group(name):
    # Keep multiple volumes of the same titled series together.
    return re.sub(r'(?i)[_-]?vol[_-]?\d+$', '', name)


# 固定随机种子，约按 75/10/15 分配来源组；不是逐个裁剪图随机分配。
def split_groups(groups, seed=42):
    groups = sorted(set(groups))
    if len(groups) < 3:
        raise ValueError('At least three independent groups required')
    random.Random(seed).shuffle(groups)
    n_train = min(len(groups)-2, max(1, int(len(groups)*.75+.5)))
    n_val = min(len(groups)-n_train-1, max(1, int(len(groups)*.1+.5)))
    return {g: 'train' if i < n_train else 'val' if i < n_train+n_val else 'test'
            for i, g in enumerate(groups)}


# 由种子和稳定标识生成局部随机源，确保某一页的采样可以复现。
def stable_rng(seed, key):
    value = int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8], 'big')
    return random.Random(value)


# 判断两个右下边界不含的矩形是否有面积交集；只接触边缘不算重叠。
def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


# 把所有人脸框向四周扩大，给负样本采样留出安全距离。
def expanded_faces(faces, width, height, margin=.15):
    return [[max(0, int(np.floor(x1-(x2-x1)*margin))), max(0, int(np.floor(y1-(y2-y1)*margin))),
             min(width, int(np.ceil(x2+(x2-x1)*margin))), min(height, int(np.ceil(y2+(y2-y1)*margin)))]
            for x1, y1, x2, y2 in faces]


def positive_crop_box(face, width, height, margin=.10, other_faces=()):
    """正例四周扩边；有新邻脸进入时逐次减半，原标注框始终不变。

    margin 是相对于原框宽高的每侧比例，不是老师指定常数。
    原框已经重叠的邻脸单独留给质量复核，不能靠缩小框悄悄裁掉。
    """
    if not np.isfinite(margin) or not 0 <= margin <= .5:
        raise ValueError('Positive margin must be finite and in [0,0.5]')
    if not (0 <= face[0] < face[2] <= width and 0 <= face[1] < face[3] <= height):
        raise ValueError('Face must be an in-bounds xyxy box')
    new_neighbors = [box for box in other_faces if not intersects(face, box)]
    effective = margin
    for _ in range(8):
        crop = expanded_faces([face], width, height, effective)[0]
        if not any(intersects(crop, box) for box in new_neighbors):
            return crop, effective
        effective /= 2
    return list(face), 0.0


# 随机找有纹理的方框，排除人脸及周边，返回原图坐标下的背景框。
def sample_background(gray, faces, rng, count=6, margin=.15):
    height, width = gray.shape
    forbidden = expanded_faces(faces, width, height, margin)
    samples, seen = [], set()
    max_size = min(192, width, height)
    if max_size < 24:
        return samples
    for _ in range(count*300):
        size = rng.randint(24, max_size)
        x, y = rng.randint(0, width-size), rng.randint(0, height-size)
        box = [x, y, x+size, y+size]
        # 只要碰到扩展人脸框就舍弃；不能用低 IoU 当作背景的唯一依据。
        if tuple(box) in seen or any(intersects(box, face) for face in forbidden):
            continue
        # Exclude almost-flat gutters/backgrounds; retain textured scene negatives.
        patch = gray[y:y+size, x:x+size]
        # 标准差描述明暗变化，太低通常是大片白边或平涂；阈值是初始工程参数。
        if float(patch.std()) < 12:
            continue
        seen.add(tuple(box))
        samples.append(box)
        if len(samples) == count:
            break
    return samples


# 检查同一来源或相同像素是否跨 train/val/test；发现问题就拒绝通过。
def audit_rows(rows):
    """Fail rather than silently redistribute an existing annotated split."""
    sources, digests = {}, {}
    for row in rows:
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        for table, key in [(sources, row['source_id']), (digests, row.get('pixel_sha256'))]:
            if key is not None and table.setdefault(key, split) != split:
                raise ValueError(f'Cross-split leakage: {key}')
    return True
