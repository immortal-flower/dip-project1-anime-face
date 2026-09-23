"""Evaluate the course landmark regressor independently with ground-truth boxes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.data_io import load_manifest, write_image, write_json
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order
from src.shape_regression import predict_shape


def evaluate(manifest, model, output, split="test", normalization="interocular",
             allow_unreviewed=False, preview_count=24):
    rows = [row for row in load_manifest(manifest)
            if row["split"] == split and row["label"] == 1 and "landmarks" in row]
    if not rows:
        raise ValueError(f"No landmark rows in split {split!r}")
    for row in rows:
        validate_landmark_order(row.get("landmark_order"))
    if split == "test" and not allow_unreviewed:
        unreviewed = [row["source_id"] for row in rows if row.get("reviewed_landmarks") is not True]
        if unreviewed:
            raise ValueError(f"Test landmarks require human review; first unreviewed: {unreviewed[0]}")

    model_path = Path(model)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError("Model directory must contain config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_landmark_order(config.get("landmark_order"))
        model_path = model_path / "landmark.npz"
    with np.load(model_path, allow_pickle=False) as archive:
        landmark_model = {name: archive[name] for name in archive.files}

    output = Path(output)
    records, values, methods = [], [], Counter()
    fallback = "bbox_diagonal" if normalization == "interocular" else None
    for index, row in enumerate(rows):
        gray = cv2.cvtColor(row["_image"], cv2.COLOR_BGR2GRAY)
        prediction = predict_shape(landmark_model, gray, row["bbox"])
        details = nme_details(prediction, row["landmarks"], row["visibility"],
                              row["bbox"], normalization, fallback)
        if details["nme"] is not None:
            values.append(details["nme"])
            methods[details["normalization"]] += 1
        records.append({"image": row["image"], "source_id": row["source_id"],
                        "nme": details["nme"],
                        "normalization": details["normalization"],
                        "normalizer": details["normalizer"]})
        if index < preview_count:
            canvas = cv2.resize(row["_image"], None, fx=4, fy=4,
                                interpolation=cv2.INTER_NEAREST)
            truth = np.asarray(row["landmarks"])
            for point in truth:
                cv2.circle(canvas, tuple(np.rint(point * 4).astype(int)), 2,
                           (0, 220, 0), -1, cv2.LINE_AA)
            for point in prediction:
                cv2.circle(canvas, tuple(np.rint(point * 4).astype(int)), 2,
                           (0, 0, 255), -1, cv2.LINE_AA)
            write_image(output / "images" / f"{index:04d}.jpg", canvas)

    result = {
        "schema_version": 1,
        "split": split,
        "landmark_order": LANDMARK_ORDER,
        "sample_count": len(rows),
        "evaluated_count": len(values),
        "human_ground_truth": all(row.get("reviewed_landmarks") is True for row in rows),
        "requested_normalization": normalization,
        "normalization_counts": dict(methods),
        "mean_nme": float(np.mean(values)) if values else None,
        "median_nme": float(np.median(values)) if values else None,
        "records": records,
    }
    write_json(output / "metrics.json", result)
    print({key: value for key, value in result.items() if key != "records"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--normalization", choices=("interocular", "bbox_diagonal"),
                        default="interocular")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--preview-count", type=int, default=24)
    args = parser.parse_args()
    evaluate(args.manifest, args.model, args.output, args.split,
             args.normalization, args.allow_unreviewed, args.preview_count)


if __name__ == "__main__":
    main()
