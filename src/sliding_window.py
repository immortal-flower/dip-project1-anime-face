import time
import cv2
from .channels11 import compute_11_channels
from .cascade import predict_window
from .pyramid import image_pyramid
from .grouping import nms


def scan_image(image, model, config):
    step = config['step']
    if not isinstance(step, int) or step < 1:
        raise ValueError('step must be a positive integer')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    predictions, logs = [], []
    for level, sx, sy in image_pyramid(gray, config['scale_factor']):
        start = time.perf_counter()
        channels = compute_11_channels(level)
        count, passed, candidates = 0, [0]*len(model['stages']), 0
        for y in range(0, level.shape[0]-23, step):
            for x in range(0, level.shape[1]-23, step):
                ok, score, n = predict_window(model, channels, x, y)
                count += 1
                for k in range(n):
                    passed[k] += 1
                if ok:
                    candidates += 1
                    predictions.append(dict(bbox=[x*sx, y*sy, (x+24)*sx, (y+24)*sy], score=score))
        logs.append(dict(width=level.shape[1], height=level.shape[0], windows=count,
                         stage_pass=passed, candidates=candidates, seconds=time.perf_counter()-start))
    return nms(predictions, config['nms_threshold']), logs
