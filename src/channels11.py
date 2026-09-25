# A：把二维灰度图变成 11 张特征图。先读 compute_11_channels，再对照通道拼图理解。
"""A: explicit integer feature channels, zero outside valid support."""
import numpy as np


# 输入 H×W 的 uint8 灰度图；返回按 C0～C10 顺序排列的同尺寸图像列表。
def compute_11_channels(gray):
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Expected a nonempty 2D uint8 grayscale image')
    # h 是高度、w 是宽度；所有通道保持原尺寸，无法计算的边缘保持 0。
    h, w = gray.shape
    ch = [gray.copy()] + [np.zeros_like(gray) for _ in range(10)]
    # Four-value average is floor((a+b+c+d+2)/4), not nested pair averages.
    # 计算两层平滑：target 为输出通道，source 为输入通道，gap 为取样间隔。
    # margin 是完整计算需要避开的右侧/下侧宽度；C2 要累计 C1 的边界。
    for target, source, gap, margin in [(1, 0, 1, 1), (2, 1, 2, 3)]:
        hh, ww = h - margin, w - margin
        if hh > 0 and ww > 0:
            # 先转 int32 再相加，避免 uint8 超过 255 后绕回造成错误。
            a = ch[source].astype(np.int32)
            ch[target][:hh, :ww] = (a[:hh, :ww] + a[gap:gap+hh, :ww]
                + a[:hh, gap:gap+ww] + a[gap:gap+hh, gap:gap+ww] + 2) // 4
    # C3～C6 在 C1 上跨 2 像素；C7～C10 在 C2 上跨 4 像素。
    for start, source, gap, margin in [(3, 1, 2, 3), (7, 2, 4, 7)]:
        hh, ww = h - margin, w - margin
        if hh <= 0 or ww <= 0:
            continue
        a = ch[source].astype(np.int32)
        # tl/tr/bl/br 分别是左上、右上、左下、右下位置对应的有效区域。
        tl, tr = a[:hh, :ww], a[:hh, gap:gap+ww]
        bl, br = a[gap:gap+hh, :ww], a[gap:gap+hh, gap:gap+ww]
        for k, delta in enumerate((tr-tl, bl-tl, br-tl, bl-tr)):
            # 差值 -255～255 映射到 0～255；零差值对应 127，并非黑色 0。
            ch[start+k][:hh, :ww] = (delta + 255) // 2
    return ch


def compute_11_channels_float(gray, quantize=True):
    """浮点对照实验，保持与整数版相同的取样位置和有效边界。

    quantize=True：每一步采用相同取整，应与整数结果逐像素相同。
    quantize=False：保留小数，观察取消中间取整后产生的差异。
    两种模式都只供对照实验；检测模型仍调用整数版。
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Expected a nonempty 2D uint8 grayscale image')
    height, width = gray.shape
    channels = [gray.astype(np.float32)] + [np.zeros(gray.shape, np.float32) for _ in range(10)]
    for target, source, gap, margin in [(1,0,1,1), (2,1,2,3)]:
        h, w = height-margin, width-margin
        if h <= 0 or w <= 0:
            continue
        plane = channels[source]
        total = (plane[:h,:w]+plane[gap:gap+h,:w]
                 +plane[:h,gap:gap+w]+plane[gap:gap+h,gap:gap+w])
        # +0.5 后向下取整，与整数版的 (sum+2)//4 一致。
        channels[target][:h,:w] = np.floor(total/4+.5) if quantize else total/4
    for start, source, gap, margin in [(3,1,2,3),(7,2,4,7)]:
        h,w = height-margin,width-margin
        if h <= 0 or w <= 0:
            continue
        plane = channels[source]
        tl,tr = plane[:h,:w],plane[:h,gap:gap+w]
        bl,br = plane[gap:gap+h,:w],plane[gap:gap+h,gap:gap+w]
        for direction,delta in enumerate((tr-tl,bl-tl,br-tl,bl-tr)):
            value = (delta+255)/2
            channels[start+direction][:h,:w] = np.floor(value) if quantize else value
    return channels
