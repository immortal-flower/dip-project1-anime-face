import cv2


def image_pyramid(gray, scale_factor=1.2):
    if scale_factor <= 1:
        raise ValueError('scale_factor must exceed 1')
    h, w = gray.shape
    factor, previous = 1.0, None
    while True:
        width, height = int(w/factor), int(h/factor)
        if min(width, height) < 24:
            break
        if (width, height) != previous:
            yield cv2.resize(gray, (width, height)), w/width, h/height
            previous = width, height
        factor *= scale_factor
