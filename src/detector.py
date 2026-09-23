# C 的共同基础版：加载检测模型和关键点模型，把两步连接成统一 detect 接口。
import json
from pathlib import Path
import cv2
import numpy as np
from .sliding_window import scan_image
from .shape_regression import predict_shape
from .landmark_schema import validate_landmark_order


def _validate_image(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('Expected nonempty BGR uint8 image')


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            item['landmarks'] = predict_shape(self.model, gray, bbox).tolist()
            results.append(item)
        return results


class AnimeFaceDetector:
    # 加载 JSON 配置和不含 pickle 的模型数组，供后续多次推理复用。
    def __init__(self, model_path):
        path = Path(model_path)
        self.config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        self.model = json.loads((path/'detector.json').read_text(encoding='utf-8'))
        self.landmark_regressor = LandmarkRegressor(path)
        self.landmark = self.landmark_regressor.model
        self.last_scan_log = []

    # 输入 BGR 图，先检测框再回归 28 点；无脸返回空列表，函数内不弹窗。
    def detect(self, image):
        _validate_image(image)
        results, self.last_scan_log = scan_image(image, self.model, self.config)
        return self.landmark_regressor.predict(image, results)
