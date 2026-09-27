"""Cascaded Fern/LBF landmark regression with visibility-weighted fitting.

Each fern evaluates shape-indexed pixel-difference tests around one current
landmark.  Its binary outcomes select one leaf, and the concatenated one-hot
leaf indicators form Local Binary Features (LBF).  A weighted ridge regressor
then predicts the 56-dimensional shape update at every cascade stage.
"""
from __future__ import annotations

import numpy as np

from .landmark_schema import FLIP_PAIRS, LANDMARK_COUNT


def _validate_training(images, boxes, points, landmark_weights, rounds,
                       ferns_per_point, fern_depth, ridge, learning_rate,
                       feature_radius):
    count = len(images)
    boxes = np.asarray(boxes, dtype=float)
    points = np.asarray(points, dtype=float)
    weights = np.asarray(landmark_weights, dtype=float)
    if count == 0 or boxes.shape != (count, 4):
        raise ValueError("Expected images and one xyxy bbox per image")
    if points.shape != (count, LANDMARK_COUNT, 2):
        raise ValueError("Expected Nx28x2 landmark coordinates")
    if weights.shape != (count, LANDMARK_COUNT):
        raise ValueError("Expected Nx28 landmark weights")
    if not (np.isfinite(boxes).all() and np.isfinite(points).all()
            and np.isfinite(weights).all()):
        raise ValueError("Training coordinates and weights must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Every bbox must have positive width and height")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("Landmark weights must be between 0 and 1")
    if not (weights.sum(axis=0) > 0).all():
        raise ValueError("Each landmark needs positive training weight")
    if rounds < 3 or rounds > 5:
        raise ValueError("Fern/LBF regression requires 3 to 5 cascade rounds")
    if ferns_per_point < 1 or fern_depth < 2 or fern_depth > 8:
        raise ValueError("Use at least one fern per point and depth 2 to 8")
    if ridge <= 0 or not 0 < learning_rate <= 1 or feature_radius <= 0:
        raise ValueError("ridge/radius must be positive; learning rate is (0,1]")
    checked = []
    for image in images:
        image = np.asarray(image)
        if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
            raise ValueError("Fern/LBF training expects nonempty uint8 grayscale images")
        checked.append(image)
    return checked, boxes, points, weights


def fern_responses(gray, shape, bbox, anchors, offsets):
    """Evaluate all shape-indexed pixel-pair tests for one image."""
    gray = np.asarray(gray)
    shape = np.asarray(shape, dtype=float)
    bbox = np.asarray(bbox, dtype=float)
    anchors = np.asarray(anchors, dtype=int)
    offsets = np.asarray(offsets, dtype=float)
    if gray.ndim != 2 or gray.dtype != np.uint8 or shape.shape != (LANDMARK_COUNT, 2):
        raise ValueError("Expected a uint8 grayscale image and normalized 28x2 shape")
    if bbox.shape != (4,) or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("Expected a positive-area xyxy bbox")
    if offsets.ndim != 4 or offsets.shape[0] != len(anchors) or offsets.shape[2:] != (2, 2):
        raise ValueError("offsets must be ferns x depth x pair x xy")
    if np.any((anchors < 0) | (anchors >= LANDMARK_COUNT)):
        raise ValueError("Fern anchors must reference the 28 landmarks")

    origin, size = bbox[:2], bbox[2:] - bbox[:2]
    positions = (shape[anchors, None, None, :] + offsets) * size + origin
    positions = np.rint(positions).astype(int)
    positions[..., 0] = np.clip(positions[..., 0], 0, gray.shape[1] - 1)
    positions[..., 1] = np.clip(positions[..., 1], 0, gray.shape[0] - 1)
    values = gray[positions[..., 1], positions[..., 0]].astype(np.float32)
    return (values[..., 0] - values[..., 1]) / 255.0


def _leaf_indices(responses, thresholds):
    responses = np.asarray(responses)
    thresholds = np.asarray(thresholds)
    if responses.shape != thresholds.shape or responses.ndim != 2:
        raise ValueError("Fern responses and thresholds must share fern x depth shape")
    bits = responses > thresholds
    powers = (1 << np.arange(bits.shape[1], dtype=np.int64))[None, :]
    return np.sum(bits.astype(np.int64) * powers, axis=1)


def _lbf_matrix(leaf_indices, leaf_count):
    leaf_indices = np.asarray(leaf_indices, dtype=np.int64)
    if leaf_indices.ndim != 2:
        raise ValueError("Expected samples x ferns leaf indices")
    count, fern_count = leaf_indices.shape
    if np.any((leaf_indices < 0) | (leaf_indices >= leaf_count)):
        raise ValueError("Leaf index outside fern range")
    matrix = np.zeros((count, fern_count * leaf_count), dtype=np.float32)
    columns = (np.arange(fern_count)[None, :] * leaf_count + leaf_indices)
    matrix[np.arange(count)[:, None], columns] = 1.0
    return matrix


def _weighted_ridge(features, targets, weights, ridge):
    """Fit one point's 2-D update with continuous sample weights.

    The dual form keeps fitting practical because LBF is sparse and normally
    has more leaf columns than training images.
    """
    active = weights > 0
    x = np.asarray(features[active], dtype=np.float64)
    y = np.asarray(targets[active], dtype=np.float64)
    root_weight = np.sqrt(np.asarray(weights[active], dtype=np.float64))[:, None]
    xw, yw = x * root_weight, y * root_weight
    gram = xw @ xw.T
    gram.flat[::len(gram) + 1] += ridge
    return xw.T @ np.linalg.solve(gram, yw)


def _add_horizontal_flips(images, boxes, points, weights):
    """Append face-crop flips with the official left/right point mapping."""
    augmented_images, augmented_boxes = list(images), list(boxes)
    augmented_points, augmented_weights = list(points), list(weights)
    normalized = ((points - boxes[:, None, :2]) /
                  (boxes[:, None, 2:] - boxes[:, None, :2]))
    for image, box, shape, point_weights in zip(images, boxes, normalized, weights):
        x1, y1 = np.floor(box[:2]).astype(int)
        x2, y2 = np.ceil(box[2:]).astype(int)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Cannot horizontally flip an empty face crop")
        flipped_shape = shape.copy()
        flipped_shape[:, 0] = 1.0 - flipped_shape[:, 0]
        flipped_weights = point_weights.copy()
        for left, right in FLIP_PAIRS:
            flipped_shape[[left, right]] = flipped_shape[[right, left]]
            flipped_weights[[left, right]] = flipped_weights[[right, left]]
        height, width = crop.shape
        augmented_images.append(np.ascontiguousarray(crop[:, ::-1]))
        augmented_boxes.append(np.asarray([0, 0, width, height], dtype=float))
        augmented_points.append(flipped_shape * [width, height])
        augmented_weights.append(flipped_weights)
    return (augmented_images, np.asarray(augmented_boxes, dtype=float),
            np.asarray(augmented_points, dtype=float),
            np.asarray(augmented_weights, dtype=float))


def train_lbf_landmark(images, boxes, points, visibility, seed=42, rounds=4,
                       ferns_per_point=4, fern_depth=4, ridge=10.0,
                       learning_rate=0.5, feature_radius=0.12,
                       landmark_weights=None, augment_flip=True):
    """Train a full cascaded Fern/LBF regressor.

    ``visibility`` supplies the required binary mask.  ``landmark_weights`` is
    optional and may contain continuous confidence values in [0, 1]; the two
    are multiplied so an invisible point always has zero loss weight.
    """
    visibility = np.asarray(visibility, dtype=float)
    confidence = (np.ones_like(visibility) if landmark_weights is None
                  else np.asarray(landmark_weights, dtype=float))
    if visibility.ndim != 2 or visibility.shape[1:] != (LANDMARK_COUNT,) \
            or not np.isin(visibility, [0, 1]).all():
        raise ValueError("visibility must be a binary Nx28 mask")
    if confidence.shape != visibility.shape or not np.isfinite(confidence).all() \
            or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("landmark_weights must be Nx28 values in [0,1]")
    combined_weights = visibility * confidence
    images, boxes, points, combined_weights = _validate_training(
        images, boxes, points, combined_weights, rounds, ferns_per_point,
        fern_depth, ridge, learning_rate, feature_radius)
    if not isinstance(augment_flip, (bool, np.bool_)):
        raise ValueError("augment_flip must be boolean")
    if augment_flip:
        images, boxes, points, combined_weights = _add_horizontal_flips(
            images, boxes, points, combined_weights)

    target = ((points - boxes[:, None, :2]) /
              (boxes[:, None, 2:] - boxes[:, None, :2]))
    mean_shape = ((target * combined_weights[:, :, None]).sum(axis=0) /
                  combined_weights.sum(axis=0)[:, None])
    shapes = np.repeat(mean_shape[None, :, :], len(images), axis=0)

    rng = np.random.default_rng(seed)
    anchors = np.repeat(np.arange(LANDMARK_COUNT, dtype=np.int16), ferns_per_point)
    fern_count = len(anchors)
    leaf_count = 1 << fern_depth
    all_offsets, all_thresholds, all_updates = [], [], []

    for _ in range(rounds):
        offsets = rng.uniform(-feature_radius, feature_radius,
                              (fern_count, fern_depth, 2, 2)).astype(np.float32)
        responses = np.stack([
            fern_responses(image, shape, box, anchors, offsets)
            for image, shape, box in zip(images, shapes, boxes)
        ])
        thresholds = np.median(responses, axis=0).astype(np.float32)
        leaf_indices = np.stack([
            _leaf_indices(response, thresholds) for response in responses
        ])
        features = _lbf_matrix(leaf_indices, leaf_count)
        residuals = target - shapes
        updates = np.zeros((fern_count * leaf_count, LANDMARK_COUNT * 2),
                           dtype=np.float32)
        for point_index in range(LANDMARK_COUNT):
            fitted = _weighted_ridge(features, residuals[:, point_index],
                                     combined_weights[:, point_index], ridge)
            updates[:, 2 * point_index:2 * point_index + 2] = (
                learning_rate * fitted).astype(np.float32)
        predicted = features @ updates
        shapes += predicted.reshape(-1, LANDMARK_COUNT, 2)
        shapes = np.clip(shapes, -0.25, 1.25)
        all_offsets.append(offsets)
        all_thresholds.append(thresholds)
        all_updates.append(updates.reshape(fern_count, leaf_count,
                                           LANDMARK_COUNT * 2))

    return {
        "mean_shape": mean_shape.astype(np.float32),
        "fern_anchors": anchors,
        "fern_offsets": np.asarray(all_offsets, dtype=np.float32),
        "fern_thresholds": np.asarray(all_thresholds, dtype=np.float32),
        "leaf_updates": np.asarray(all_updates, dtype=np.float32),
        "learning_rate": np.asarray(learning_rate, dtype=np.float32),
    }


def predict_lbf_landmark(model, gray, bbox):
    """Predict 28 original-image coordinates from one supplied face box."""
    required = ("mean_shape", "fern_anchors", "fern_offsets",
                "fern_thresholds", "leaf_updates")
    if any(key not in model for key in required):
        raise ValueError("Incomplete Fern/LBF landmark model")
    shape = np.asarray(model["mean_shape"], dtype=float).copy()
    anchors = np.asarray(model["fern_anchors"], dtype=int)
    offsets = np.asarray(model["fern_offsets"], dtype=float)
    thresholds = np.asarray(model["fern_thresholds"], dtype=float)
    updates = np.asarray(model["leaf_updates"], dtype=float)
    if offsets.shape[:2] != thresholds.shape[:2] or len(offsets) != len(updates):
        raise ValueError("Fern/LBF stage arrays do not agree")
    fern_count = len(anchors)
    for stage in range(len(offsets)):
        response = fern_responses(gray, shape, bbox, anchors, offsets[stage])
        leaves = _leaf_indices(response, thresholds[stage])
        if updates[stage].shape[0] != fern_count:
            raise ValueError("Fern/LBF update table has the wrong fern count")
        update = updates[stage, np.arange(fern_count), leaves].sum(axis=0)
        shape += update.reshape(LANDMARK_COUNT, 2)
        shape = np.clip(shape, -0.25, 1.25)
    bbox = np.asarray(bbox, dtype=float)
    return shape * (bbox[2:] - bbox[:2]) + bbox[:2]
