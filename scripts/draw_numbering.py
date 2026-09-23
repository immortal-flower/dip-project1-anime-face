"""Generate a schematic, not a training image."""
import cv2
import numpy as np
from src.data_io import write_image


def main():
    points = [(105,220),(125,300),(240,370),(355,300),(375,220),
              (135,150),(175,135),(215,145),(265,145),(305,135),(345,150),
              (135,220),(165,205),(205,215),(155,240),(185,240),(215,225),
              (265,215),(305,205),(345,220),(265,225),(295,240),(325,240),
              (240,270),(200,320),(240,315),(280,320),(240,340)]
    image = np.full((420, 480, 3), 250, np.uint8)
    for group in ([0,1,2,3,4], [5,6,7], [8,9,10],
                  [11,12,13,16,15,14,11], [17,18,19,22,21,20,17],
                  [24,25,26,27,24]):
        cv2.polylines(image, [np.array([points[i] for i in group])], False, (150,150,150), 2)
    for i, (x,y) in enumerate(points):
        cv2.circle(image, (x,y), 3, (20,120,220), -1)
        cv2.putText(image, str(i), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, .42, (30,30,30), 1)
    cv2.putText(image, 'hysts28-v1 | schematic only', (35,35), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1)
    write_image('docs/landmark_numbering.png', image)


if __name__ == '__main__':
    main()
