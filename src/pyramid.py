# B 的基础版：缩放整幅图像，使固定 24×24 窗口能搜索不同大小的人脸。
import cv2


# 生成缩放图及其回到原图的 x/y 比例，小于检测窗口时停止。
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
