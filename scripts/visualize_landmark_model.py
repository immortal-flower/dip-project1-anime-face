"""Render teacher pseudo labels and shape-regressor predictions for review."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from src.data_io import load_manifest, write_image
from src.detector import LandmarkRegressor
from src.landmark_schema import validate_landmark_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    regressor = LandmarkRegressor(args.model)
    rows = [row for row in load_manifest(args.manifest)
            if row["split"] == args.split and "landmarks" in row]
    for row in rows:
        validate_landmark_order(row.get("landmark_order"))
    if args.scale <= 0:
        parser.error("--scale must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows[:args.limit]):
        image = row["_image"].copy()
        predicted = np.asarray(regressor.predict(image, [row["bbox"]])[0]["landmarks"])
        teacher = np.asarray(row["landmarks"])
        visible = np.asarray(row["visibility"], dtype=bool)
        image = cv2.resize(image, None, fx=args.scale, fy=args.scale,
                           interpolation=cv2.INTER_NEAREST)
        for point in teacher[visible]:
            cv2.circle(image, tuple(np.rint(point * args.scale).astype(int)), 2,
                       (0, 220, 0), -1,
                       cv2.LINE_AA)
        for point in predicted:
            cv2.circle(image, tuple(np.rint(point * args.scale).astype(int)), 2,
                       (0, 0, 255), -1,
                       cv2.LINE_AA)
        cv2.putText(image, "green=teacher  red=regressor", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        write_image(output / f"{index:03d}.jpg", image)
    print(f"Saved {min(args.limit, len(rows))} previews to {output}")


if __name__ == "__main__":
    main()
