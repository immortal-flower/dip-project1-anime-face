"""Predict 28 landmarks inside Manga109 ground-truth face boxes."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image, write_json
from src.detector import LandmarkRegressor


REGION_COLORS = [
    ((0, 5), (255, 120, 30)),
    ((5, 11), (0, 165, 255)),
    ((11, 23), (30, 200, 80)),
    ((23, 24), (40, 40, 230)),
    ((24, 28), (190, 80, 180)),
]


def page_faces(annotation: Path, page_index: int):
    root = ET.parse(annotation).getroot()
    page = next((node for node in root.find("pages").findall("page")
                 if int(node.attrib["index"]) == page_index), None)
    if page is None:
        raise ValueError(f"Page {page_index} is absent from {annotation}")
    return [[float(node.attrib[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            for node in page.findall("face")]


def draw(image, faces):
    canvas = image.copy()
    radius = max(2, round(min(image.shape[:2]) / 220))
    for index, face in enumerate(faces, 1):
        box = np.rint(face["bbox"]).astype(int)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (255, 100, 0), 2)
        cv2.putText(canvas, f"GT face {index}", (box[0], max(18, box[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)
        points = np.asarray(face["landmarks"], dtype=float)
        for (start, stop), color in REGION_COLORS:
            for x, y in points[start:stop]:
                cv2.circle(canvas, tuple(np.rint([x, y]).astype(int)), radius,
                           color, -1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manga-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pages", nargs="+", required=True,
                        help="Entries formatted as Book:page, for example Akuhamu:031")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root, output = Path(args.manga_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    regressor = LandmarkRegressor(args.model_dir)
    summary = []
    for value in args.pages:
        book, page_text = value.rsplit(":", 1)
        page_index = int(page_text)
        image_path = root / "images" / book / f"{page_index:03d}.jpg"
        annotation = root / "annotations.v2020.12.18" / f"{book}.xml"
        image = read_image(image_path)
        boxes = page_faces(annotation, page_index)
        height, width = image.shape[:2]
        boxes = [[max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
                 for x1, y1, x2, y2 in boxes]
        boxes = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
        faces = regressor.predict(image, boxes)
        stem = f"{book}_{page_index:03d}"
        write_image(output / f"{stem}.jpg", draw(image, faces))
        write_json(output / f"{stem}.json", {
            "source": str(image_path),
            "bbox_source": "Manga109 annotations.v2020.12.18 face",
            "landmark_model": str(Path(args.model_dir).resolve()),
            "landmark_config": regressor.config,
            "faces": faces,
        })
        summary.append({"book": book, "page": page_index, "faces": len(faces),
                        "image": f"{stem}.jpg", "json": f"{stem}.json"})
        print(f"{book}:{page_index:03d}: {len(faces)} annotated face(s)", flush=True)
    write_json(output / "summary.json", {"pages": len(summary), "results": summary})


if __name__ == "__main__":
    main()
