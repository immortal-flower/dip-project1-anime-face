"""Local OpenCV annotator: draw boxes, click 28 points, explicitly review pages."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.data_io import read_image, write_json
from src.landmark_schema import LANDMARK_ORDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, help='Directory containing images')
    p.add_argument('--output', required=True, help='New or existing page manifest')
    a = p.parse_args()
    root, output = Path(a.images).resolve(), Path(a.output).resolve()
    import os
    pages = json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
    known = {str((output.parent/r['image']).resolve()) for r in pages}
    files = sorted(f for f in root.rglob('*') if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
    for path in files:
        if str(path) in known:
            continue
        image = read_image(path)
        scale = min(1., 1000/image.shape[1], 750/image.shape[0])
        display = cv2.resize(image, None, fx=scale, fy=scale)
        faces = []
        while True:
            canvas = display.copy()
            for face in faces:
                b = np.rint(np.asarray(face['bbox'])*scale).astype(int)
                cv2.rectangle(canvas, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 1)
            print(f'{path.name}: select face rectangle, ENTER confirms; ESC ends boxes.')
            x, y, w, h = cv2.selectROI('Select face - ENTER / ESC', canvas, False)
            cv2.destroyAllWindows()
            if w == 0 or h == 0:
                break
            box = [int(x/scale), int(y/scale), min(image.shape[1], int(np.ceil((x+w)/scale))),
                   min(image.shape[0], int(np.ceil((y+h)/scale)))]
            points, visible = [], []
            window = '28 points: click; right-click invisible; U undo; ENTER finish; ESC discard'
            def click(event, px, py, flags, param):
                if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and len(points) < 28:
                    points.append([px/scale, py/scale])
                    visible.append(int(event == cv2.EVENT_LBUTTONDOWN))
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, click)
            saved = False
            while True:
                canvas = display.copy()
                for j, point in enumerate(points):
                    q = tuple(np.rint(np.asarray(point)*scale).astype(int))
                    cv2.circle(canvas, q, 3, (0, 255, 0) if visible[j] else (0, 0, 255), -1)
                    cv2.putText(canvas, str(j), q, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1)
                cv2.putText(canvas, f'Next index: {len(points)} / 28', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(30) & 255
                if key in (ord('u'), ord('U')) and points:
                    points.pop(); visible.pop()
                if key == 13 and len(points) == 28:
                    saved = True
                    break
                if key == 27:
                    break
            cv2.destroyAllWindows()
            if saved:
                faces.append(dict(bbox=box, landmarks=points, visibility=visible,
                                  landmark_order=LANDMARK_ORDER))
        answer = input('All faces and points checked? Type YES to save; otherwise skip this image: ')
        if answer != 'YES':
            continue
        pages.append(dict(image=os.path.relpath(path, output.parent), source_id=path.relative_to(root).as_posix(),
                          complete=True, reviewed=True, landmark_order=LANDMARK_ORDER, faces=faces))
        write_json(output, pages)
    print(f'Saved {len(pages)} checked pages to {output}')


if __name__ == '__main__':
    main()
