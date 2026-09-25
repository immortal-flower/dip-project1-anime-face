# A 的通道观察入口：读一张图片 → 灰度化 → 计算 11 通道 → 拼图保存。
import argparse
import cv2
import numpy as np
from src.channels11 import compute_11_channels
from src.data_io import read_image, write_image


def channel_grid(gray, tile_size=200, original=None):
    """按固定 0～255 灰度显示 12 格图，保持长宽比，不分别拉伸对比度。"""
    tiles = []
    for label,plane in [('original',gray if original is None else original)] + [(f'C{i}',ch) for i,ch in enumerate(compute_11_channels(gray))]:
        scale = tile_size/max(plane.shape[:2])
        height,width = max(1,round(plane.shape[0]*scale)),max(1,round(plane.shape[1]*scale))
        resized = cv2.resize(plane,(width,height))
        tile = np.full((tile_size+28,tile_size,3),255,np.uint8)
        y,x = 28+(tile_size-height)//2,(tile_size-width)//2
        tile[y:y+height,x:x+width] = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR) if resized.ndim==2 else resized
        cv2.putText(tile,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,12,4)])


# 命令行入口：读取参数、调用主要处理函数，并将结果保存到指定位置。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = read_image(args.image)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    write_image(args.output, channel_grid(gray, original=original))


if __name__ == '__main__':
    main()
