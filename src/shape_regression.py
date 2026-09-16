"""C: normalized shapes, local pixel differences, masked ridge residuals."""
import numpy as np


def shape_features(gray, shape, bbox, offsets):
    origin, size = np.asarray(bbox[:2]), np.asarray(bbox[2:])-bbox[:2]
    points = (shape[:, None, None, :] + offsets) * size + origin
    points = np.rint(points).astype(int)
    points[..., 0] = np.clip(points[..., 0], 0, gray.shape[1]-1)
    points[..., 1] = np.clip(points[..., 1], 0, gray.shape[0]-1)
    values = gray[points[..., 1], points[..., 0]].astype(float)
    return np.r_[(values[:, :, 0]-values[:, :, 1]).ravel()/255.0, 1.0]


def train_shape(images, boxes, points, visibility, seed=42, rounds=3, ridge=1.0):
    points, visibility = np.asarray(points), np.asarray(visibility, dtype=bool)
    boxes = np.asarray(boxes, dtype=float)
    if rounds < 3 or rounds > 5 or ridge <= 0 or points.shape != (len(images), 28, 2) or visibility.shape != (len(images), 28):
        raise ValueError('Expected Nx28x2 points, Nx28 mask, 3-5 rounds and positive ridge')
    if not visibility.any(axis=0).all():
        raise ValueError('Each landmark needs at least one visible training example')
    target = (points-boxes[:, None, :2]) / (boxes[:, None, 2:]-boxes[:, None, :2])
    mean = (target*visibility[:, :, None]).sum(axis=0) / visibility.sum(axis=0)[:, None]
    shapes = np.repeat(mean[None], len(images), axis=0)
    offsets = np.random.default_rng(seed).uniform(-0.12, 0.12, (28, 2, 2, 2))
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
