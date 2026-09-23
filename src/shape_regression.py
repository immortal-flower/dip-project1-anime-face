"""C: normalized shapes, local pixel differences, masked ridge residuals."""
import numpy as np

from .landmark_schema import LANDMARK_COUNT


def shape_features(gray, shape, bbox, offsets):
    gray = np.asarray(gray)
    shape = np.asarray(shape, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    bbox = np.asarray(bbox, dtype=float)
    if gray.ndim != 2 or shape.shape != (LANDMARK_COUNT, 2):
        raise ValueError('Expected a grayscale image and a 28x2 shape')
    if offsets.ndim != 4 or offsets.shape[0] != LANDMARK_COUNT or offsets.shape[2:] != (2, 2):
        raise ValueError('offsets must have shape 28 x pairs x 2 x 2')
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError('Expected a finite positive-area xyxy bbox')
    origin, size = bbox[:2], bbox[2:]-bbox[:2]
    if np.any(size <= 0):
        raise ValueError('Expected a finite positive-area xyxy bbox')
    points = (shape[:, None, None, :] + offsets) * size + origin
    points = np.rint(points).astype(int)
    points[..., 0] = np.clip(points[..., 0], 0, gray.shape[1]-1)
    points[..., 1] = np.clip(points[..., 1], 0, gray.shape[0]-1)
    values = gray[points[..., 1], points[..., 0]].astype(float)
    return np.r_[(values[:, :, 0]-values[:, :, 1]).ravel()/255.0, 1.0]


def train_shape(images, boxes, points, visibility, seed=42, rounds=3, ridge=1.0,
                pairs_per_point=8):
    points, visibility = np.asarray(points), np.asarray(visibility)
    boxes = np.asarray(boxes, dtype=float)
    if rounds < 3 or rounds > 5 or ridge <= 0 or pairs_per_point < 2 or points.shape != (len(images), 28, 2) or visibility.shape != (len(images), 28):
        raise ValueError('Expected Nx28x2 points, Nx28 mask, 3-5 rounds and positive ridge')
    if len(images) == 0 or boxes.shape != (len(images), 4) or not np.isfinite(boxes).all():
        raise ValueError('Expected images and one finite xyxy bbox per image')
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Every bbox must have positive width and height')
    if not np.isfinite(points).all() or not np.isin(visibility, [0, 1]).all():
        raise ValueError('Landmarks and visibility must be finite/binary')
    visibility = visibility.astype(bool)
    if not visibility.any(axis=0).all():
        raise ValueError('Each landmark needs at least one visible training example')
    target = (points-boxes[:, None, :2]) / (boxes[:, None, 2:]-boxes[:, None, :2])
    mean = (target*visibility[:, :, None]).sum(axis=0) / visibility.sum(axis=0)[:, None]
    shapes = np.repeat(mean[None], len(images), axis=0)
    offsets = np.random.default_rng(seed).uniform(
        -0.12, 0.12, (LANDMARK_COUNT, pairs_per_point, 2, 2))
    weights = []
    for _ in range(rounds):
        x = np.stack([shape_features(im, shape, box, offsets) for im, shape, box in zip(images, shapes, boxes)])
        residual = (target-shapes).reshape(len(images), 56)
        w = np.zeros((x.shape[1], 56))
        for j in range(28):
            mask = visibility[:, j]
            a = x[mask]
            penalty = np.eye(a.shape[1])*ridge
            penalty[-1, -1] = 0
            w[:, 2*j:2*j+2] = np.linalg.solve(a.T@a+penalty, a.T@residual[mask, 2*j:2*j+2])
        shapes += (x@w).reshape(-1, 28, 2)
        weights.append(w)
    return dict(mean_shape=mean, offsets=offsets, weights=np.asarray(weights))


def predict_shape(model, gray, bbox):
    shape = model['mean_shape'].copy()
    for w in model['weights']:
        shape += (shape_features(gray, shape, bbox, model['offsets'])@w).reshape(28, 2)
    return shape*(np.asarray(bbox[2:])-bbox[:2])+bbox[:2]
