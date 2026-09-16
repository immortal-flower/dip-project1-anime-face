import argparse
import cv2
import numpy as np
from src.channels11 import compute_11_channels
from src.data_io import read_image, write_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    gray = cv2.cvtColor(read_image(args.image), cv2.COLOR_BGR2GRAY)
    tiles = []
    for label, plane in [('original', gray)] + [(f'C{i}', ch) for i, ch in enumerate(compute_11_channels(gray))]:
        tile = cv2.cvtColor(cv2.resize(plane, (200, 200)), cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        tiles.append(tile)
    write_image(args.output, np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, 12, 4)]))


if __name__ == '__main__':
    main()
