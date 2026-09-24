"""HOG + PCA + masked Ridge landmark regression.

This is an optional feature-engineered comparison to the local pixel-difference
shape regressor.  It predicts normalized coordinates from a normalized face
crop and supports per-point visibility masks.
"""
from __future__ import annotations

import cv2
import numpy as np

from .landmark_schema import FLIP_PAIRS, LANDMARK_COUNT


HOG_SIZE = 64


def _hog_descriptor():
    return cv2.HOGDescriptor((HOG_SIZE, HOG_SIZE), (16, 16), (8, 8), (8, 8), 9)


def hog_features(image, bbox):
    image = np.asarray(image)
    bbox = np.asarray(bbox, dtype=float)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected a nonempty grayscale or BGR image")
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("Expected one finite positive-area xyxy bbox")
    height, width = image.shape[:2]
    x1, y1 = np.floor(bbox[:2]).astype(int)
    x2, y2 = np.ceil(bbox[2:]).astype(int)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("The bbox must stay inside the image")
    crop = image[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return _hog_descriptor().compute(crop).reshape(-1).astype(np.float64)


def _flip_target(target, visibility):
    target, visibility = target.copy(), visibility.copy()
    target[:, 0] = 1.0 - target[:, 0]
    for left, right in FLIP_PAIRS:
        target[[left, right]] = target[[right, left]]
        visibility[[left, right]] = visibility[[right, left]]
    return target, visibility


def train_hog_landmark(images, boxes, points, visibility, pca_dim=48,
                       ridge=10.0, augment_flip=True):
    if pca_dim <= 0 or ridge <= 0:
        raise ValueError("pca_dim and ridge must be positive")
    boxes = np.asarray(boxes, dtype=float)
    points = np.asarray(points, dtype=float)
    visibility = np.asarray(visibility)
    count = len(images)
    if count == 0 or boxes.shape != (count, 4):
        raise ValueError("Expected images and one xyxy bbox per image")
    if points.shape != (count, LANDMARK_COUNT, 2):
        raise ValueError("Expected Nx28x2 landmark coordinates")
    if visibility.shape != (count, LANDMARK_COUNT) or not np.isin(visibility, [0, 1]).all():
        raise ValueError("Expected an Nx28 binary visibility mask")
    if not np.isfinite(boxes).all() or not np.isfinite(points).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Boxes and landmarks must be finite with positive box areas")
    visibility = visibility.astype(bool)
    target = (points - boxes[:, None, :2]) / (boxes[:, None, 2:] - boxes[:, None, :2])

    features, targets, masks = [], [], []
    for image, box, shape, visible in zip(images, boxes, target, visibility):
        features.append(hog_features(image, box))
        targets.append(shape)
        masks.append(visible)
        if augment_flip:
            x1, y1, x2, y2 = np.rint(box).astype(int)
            crop = image[y1:y2, x1:x2]
            flipped_shape, flipped_visible = _flip_target(shape, visible)
            features.append(hog_features(cv2.flip(crop, 1), [0, 0, crop.shape[1], crop.shape[0]]))
            targets.append(flipped_shape)
            masks.append(flipped_visible)
    features = np.asarray(features)
    targets = np.asarray(targets)
    masks = np.asarray(masks)
    if not masks.any(axis=0).all():
        raise ValueError("Each landmark needs at least one visible training example")

    feature_mean = features.mean(axis=0)
    centered = features - feature_mean
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    actual_dim = min(int(pca_dim), *right_vectors.shape)
    components = right_vectors[:actual_dim]
    reduced = centered @ components.T
    weights = np.zeros((LANDMARK_COUNT, actual_dim, 2))
    bias = np.zeros((LANDMARK_COUNT, 2))
    for index in range(LANDMARK_COUNT):
        x = reduced[masks[:, index]]
        y = targets[masks[:, index], index]
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        x_centered, y_centered = x - x_mean, y - y_mean
        weights[index] = np.linalg.solve(
            x_centered.T @ x_centered + ridge * np.eye(actual_dim),
            x_centered.T @ y_centered,
        )
        bias[index] = y_mean - x_mean @ weights[index]
    return dict(feature_mean=feature_mean, pca_components=components,
                weights=weights, bias=bias)


def predict_hog_landmark(model, image, bbox):
    feature = hog_features(image, bbox)
    reduced = (feature - model["feature_mean"]) @ model["pca_components"].T
    normalized = np.einsum("d,jdc->jc", reduced, model["weights"]) + model["bias"]
    bbox = np.asarray(bbox, dtype=float)
    return normalized * (bbox[2:] - bbox[:2]) + bbox[:2]
