"""Interactively review and correct teacher-generated 28-point labels.

Left-drag moves the nearest point; right-click toggles its visibility.  Enter
accepts the current sample, P goes back, U undoes, S saves, and Q saves/exits.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_json
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order, validate_landmarks


WINDOW = "hysts28-v1 correction"


def nearest_landmark(points, x, y, maximum_distance):
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points - [x, y], axis=1)
    index = int(np.argmin(distances))
    return index if distances[index] <= maximum_distance else None


def validate_correction_row(row):
    validate_landmark_order(row.get("landmark_order"))
    validate_landmarks(row["landmarks"], row["visibility"])
    bbox = np.asarray(row["bbox"], dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("Expected one finite positive-area xyxy bbox")


def _portable_image_path(image_path: Path, output_parent: Path) -> str:
    try:
        return os.path.relpath(image_path, output_parent)
    except ValueError:
        return str(image_path)


def correct(input_path, output_path, start_index=0, include_reviewed=False):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a nonempty landmark manifest")
    rows = copy.deepcopy(rows)
    image_paths = []
    for row in rows:
        if "landmark_order" not in row and row.get("teacher_model") == "anime-face-detector/hrnetv2":
            row["landmark_order"] = LANDMARK_ORDER
        validate_correction_row(row)
        image_path = (input_path.parent / row["image"]).resolve()
        image_paths.append(image_path)
        row["image"] = _portable_image_path(image_path, output_path.parent)

    def save():
        write_json(output_path, rows)
        reviewed = sum(r.get("reviewed_landmarks") is True for r in rows)
        print(f"saved {output_path}; reviewed {reviewed}/{len(rows)}", flush=True)

    candidates = [i for i, row in enumerate(rows)
                  if include_reviewed or row.get("reviewed_landmarks") is not True]
    candidates = [i for i in candidates if i >= start_index]
    if not candidates:
        save()
        return

    position = 0
    while 0 <= position < len(candidates):
        row_index = candidates[position]
        row = rows[row_index]
        image = read_image(image_paths[row_index])
        height, width = image.shape[:2]
        scale = min(8.0, 1000 / width, 750 / height)
        scale = max(1.0, scale)
        points = np.asarray(row["landmarks"], dtype=float)
        visible = np.asarray(row["visibility"], dtype=int)
        history = []
        state = {"dragging": None}

        def mouse(event, px, py, flags, _):
            x, y = px / scale, py / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                index = nearest_landmark(points, x, y, 14 / scale)
                if index is not None:
                    history.append((points.copy(), visible.copy()))
                    state["dragging"] = index
            elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
                points[state["dragging"]] = [np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)]
            elif event == cv2.EVENT_LBUTTONUP:
                state["dragging"] = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                index = nearest_landmark(points, x, y, 14 / scale)
                if index is not None:
                    history.append((points.copy(), visible.copy()))
                    visible[index] = 1 - visible[index]

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, mouse)
        move = 0
        while True:
            canvas = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_NEAREST)
            for index, point in enumerate(points):
                center = tuple(np.rint(point * scale).astype(int))
                color = (0, 220, 0) if visible[index] else (0, 80, 255)
                cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(index), (center[0] + 4, center[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
            status = (f"{row_index + 1}/{len(rows)}  left-drag move | right visibility | "
                      "ENTER next | P prev | U undo | S save | Q quit")
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 25), (30, 30, 30), -1)
            cv2.putText(canvas, status, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 255
            if key in (ord("u"), ord("U")) and history:
                previous_points, previous_visibility = history.pop()
                points[:] = previous_points
                visible[:] = previous_visibility
            elif key in (ord("s"), ord("S")):
                row["landmarks"], row["visibility"] = points.tolist(), visible.tolist()
                save()
            elif key in (13, ord("n"), ord("N")):
                row["landmarks"], row["visibility"] = points.tolist(), visible.tolist()
                row["landmark_order"] = LANDMARK_ORDER
                row["annotation_status"] = "human_reviewed"
                row["reviewed_landmarks"] = True
                save()
                move = 1
                break
            elif key in (ord("p"), ord("P")):
                row["landmarks"], row["visibility"] = points.tolist(), visible.tolist()
                move = -1
                break
            elif key in (ord("q"), ord("Q"), 27):
                row["landmarks"], row["visibility"] = points.tolist(), visible.tolist()
                save()
                cv2.destroyAllWindows()
                return
        position = max(0, position + move)
    cv2.destroyAllWindows()
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--include-reviewed", action="store_true")
    args = parser.parse_args()
    correct(args.input, args.output, args.start_index, args.include_reviewed)


if __name__ == "__main__":
    main()
