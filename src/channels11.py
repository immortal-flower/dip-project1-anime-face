"""A: explicit integer feature channels, zero outside valid support."""
import numpy as np


def compute_11_channels(gray):
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Expected a nonempty 2D uint8 grayscale image')
    h, w = gray.shape
    ch = [gray.copy()] + [np.zeros_like(gray) for _ in range(10)]
    # Four-value average is floor((a+b+c+d+2)/4), not nested pair averages.
    for target, source, gap, margin in [(1, 0, 1, 1), (2, 1, 2, 3)]:
        hh, ww = h - margin, w - margin
        if hh > 0 and ww > 0:
            a = ch[source].astype(np.int32)
            ch[target][:hh, :ww] = (a[:hh, :ww] + a[gap:gap+hh, :ww]
                + a[:hh, gap:gap+ww] + a[gap:gap+hh, gap:gap+ww] + 2) // 4
    for start, source, gap, margin in [(3, 1, 2, 3), (7, 2, 4, 7)]:
        hh, ww = h - margin, w - margin
        if hh <= 0 or ww <= 0:
            continue
        a = ch[source].astype(np.int32)
        tl, tr = a[:hh, :ww], a[:hh, gap:gap+ww]
        bl, br = a[gap:gap+hh, :ww], a[gap:gap+hh, gap:gap+ww]
        for k, delta in enumerate((tr-tl, bl-tl, br-tl, bl-tr)):
            ch[start+k][:hh, :ww] = (delta + 255) // 2
    return ch
