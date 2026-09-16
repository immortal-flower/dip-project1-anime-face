import json
from pathlib import Path
import cv2
import numpy as np
from .sliding_window import scan_image
from .shape_regression import predict_shape


class AnimeFaceDetector:
    def __init__(self, model_path):
        path = Path(model_path)
        self.config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        self.model = json.loads((path/'detector.json').read_text(encoding='utf-8'))
        with np.load(path/'landmark.npz', allow_pickle=False) as archive:
            self.landmark = {key: archive[key] for key in archive.files}
        self.last_scan_log = []

    def detect(self, image):
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError('Expected nonempty BGR uint8 image')
        results, self.last_scan_log = scan_image(image, self.model, self.config)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for item in results:
            item['landmarks'] = predict_shape(self.landmark, gray, item['bbox']).tolist()
        return results
