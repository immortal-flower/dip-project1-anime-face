import json
from pathlib import Path
import cv2
import numpy as np
from .sliding_window import scan_image
from .shape_regression import predict_shape
from .hog_landmark import predict_hog_landmark
from .lbf_landmark import predict_lbf_landmark
from .landmark_schema import validate_landmark_order


def _validate_image(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty BGR uint8 image')


def _refine_to_edges(image, bbox, points, config):
    """Move predictions slightly toward strong nearby lines with a spatial prior."""
    radius_ratio = float(config.get('radius_ratio', 0.05))
    strength = float(config.get('strength', 0.5))
    quantile = float(config.get('quantile', 0.8))
    if (radius_ratio <= 0 or not 0 <= strength <= 1
            or not 0 <= quantile < 1):
        raise ValueError('Invalid edge_refinement configuration')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    result = np.asarray(points, dtype=float).copy()
    bbox = np.asarray(bbox, dtype=float)
    radius = max(2, int(round(np.min(bbox[2:] - bbox[:2]) * radius_ratio)))
    sigma = max(1.0, radius * 0.65)
    height, width = gray.shape
    for index, (cx, cy) in enumerate(result):
        x1, x2 = max(0, int(np.floor(cx - radius))), min(width, int(np.ceil(cx + radius + 1)))
        y1, y2 = max(0, int(np.floor(cy - radius))), min(height, int(np.ceil(cy + radius + 1)))
        yy, xx = np.mgrid[y1:y2, x1:x2]
        spatial = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        scores = magnitude[y1:y2, x1:x2] * spatial
        threshold = np.quantile(scores, quantile)
        weights = np.where(scores >= threshold, scores, 0.0)
        total = weights.sum()
        if total > 1e-9:
            target = np.array([(weights * xx).sum() / total,
                               (weights * yy).sum() / total])
            result[index] += strength * (target - result[index])
    return result


class LandmarkRegressor:
    """Load the course landmark model and predict points for supplied boxes."""

    def __init__(self, model_path):
        path = Path(model_path)
        if path.is_file():
            archive_path, config_path = path, path.with_name('config.json')
        else:
            archive_path, config_path = path/'landmark.npz', path/'config.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        if self.config and not self.config.get('synthetic', False):
            validate_landmark_order(self.config.get('landmark_order'))
        with np.load(archive_path, allow_pickle=False) as archive:
            self.model = {key: archive[key] for key in archive.files}

    def predict(self, image, boxes):
        _validate_image(image)
        model_type = self.config.get('model_type', 'shape_regression')
        if model_type not in ('shape_regression', 'hog_ridge', 'hog_shape_ensemble',
                              'hog_shape_lbf_ensemble', 'lbf_fern'):
            raise ValueError(f'Unsupported landmark model_type: {model_type}')
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if model_type in ('shape_regression', 'hog_shape_ensemble',
                                  'hog_shape_lbf_ensemble', 'lbf_fern')
                else None)
        if model_type in ('hog_shape_ensemble', 'hog_shape_lbf_ensemble'):
            prefix = 'base_' if model_type == 'hog_shape_lbf_ensemble' else ''
            hog_model = {key[len(prefix) + 4:]: value for key, value in self.model.items()
                         if key.startswith(prefix + 'hog_')}
            shape_model = {key[len(prefix) + 6:]: value for key, value in self.model.items()
                           if key.startswith(prefix + 'shape_')}
            hog_weight = float(self.config.get('hog_weight', 0.5))
            residual_gain = float(self.config.get('residual_gain', 1.0))
            if (not 0.0 <= hog_weight <= 1.0 or not np.isfinite(residual_gain)
                    or residual_gain <= 0.0 or not hog_model or not shape_model):
                raise ValueError('Invalid HOG/shape ensemble model')
            if model_type == 'hog_shape_lbf_ensemble':
                lbf_model = {key[4:]: value for key, value in self.model.items()
                             if key.startswith('lbf_')}
                lbf_weight = float(self.config.get('lbf_weight', 0.0))
                if not 0.0 <= lbf_weight <= 1.0 or not lbf_model:
                    raise ValueError('Invalid HOG/shape/LBF ensemble model')
        results = []
        for value in boxes:
            source = dict(value) if isinstance(value, dict) else {'bbox': value}
            bbox = np.asarray(source['bbox'], dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
                raise ValueError('Each bbox must be finite positive-area xyxy')
            height, width = image.shape[:2]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                raise ValueError('Each bbox must stay inside the image')
            item = {key: val for key, val in source.items() if key != 'landmarks'}
            item['bbox'] = bbox.tolist()
            if model_type == 'hog_ridge':
                prediction = predict_hog_landmark(self.model, image, bbox)
            elif model_type == 'lbf_fern':
                prediction = predict_lbf_landmark(self.model, gray, bbox)
            elif model_type in ('hog_shape_ensemble', 'hog_shape_lbf_ensemble'):
                hog_prediction = predict_hog_landmark(hog_model, image, bbox)
                shape_prediction = predict_shape(shape_model, gray, bbox)
                prediction = (hog_weight * hog_prediction
                              + (1.0 - hog_weight) * shape_prediction)
                if residual_gain != 1.0:
                    mean_prediction = (shape_model['mean_shape']
                                       * (bbox[2:] - bbox[:2]) + bbox[:2])
                    prediction = (mean_prediction
                                  + residual_gain * (prediction - mean_prediction))
                if model_type == 'hog_shape_lbf_ensemble':
                    lbf_prediction = predict_lbf_landmark(lbf_model, gray, bbox)
                    prediction = ((1.0 - lbf_weight) * prediction
                                  + lbf_weight * lbf_prediction)
            else:
                prediction = predict_shape(self.model, gray, bbox)
            edge_config = self.config.get('edge_refinement')
            if edge_config and edge_config.get('enabled', True):
                prediction = _refine_to_edges(image, bbox, prediction, edge_config)
            item['landmarks'] = prediction.tolist()
            results.append(item)
        return results


class AnimeFaceDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        self.config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        self.model = json.loads((path/'detector.json').read_text(encoding='utf-8'))
        self.landmark_regressor = LandmarkRegressor(path)
        self.landmark = self.landmark_regressor.model
        self.last_scan_log = []

    def detect(self, image):
        _validate_image(image)
        results, self.last_scan_log = scan_image(image, self.model, self.config)
        return self.landmark_regressor.predict(image, results)
