"""Train and export a validation-selected HOG/shape landmark ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data_io import load_manifest, write_json
from src.detector import LandmarkRegressor
from src.hog_landmark import train_hog_landmark
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order
from src.shape_regression import train_shape


def train(manifest, output, hog_weight=0.5, pca_dim=128, hog_ridge=10.0,
          shape_rounds=5, shape_ridge=100.0, pairs_per_point=24):
    if not 0.0 <= hog_weight <= 1.0:
        raise ValueError("hog_weight must be between 0 and 1")
    rows = list(load_manifest(manifest))
    train_rows = [row for row in rows if row["split"] == "train" and row["label"] == 1
                  and "landmarks" in row and any(row["visibility"])]
    if not train_rows:
        raise ValueError("No visible positive training landmarks")
    for row in [row for row in rows if "landmarks" in row]:
        validate_landmark_order(row.get("landmark_order"))

    images = [row["_image"] for row in train_rows]
    boxes = [row["bbox"] for row in train_rows]
    points = [row["landmarks"] for row in train_rows]
    visibility = [row["visibility"] for row in train_rows]
    hog = train_hog_landmark(images, boxes, points, visibility, pca_dim=pca_dim,
                             ridge=hog_ridge, augment_flip=True)
    shape = train_shape([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images],
                        boxes, points, visibility, seed=42, rounds=shape_rounds,
                        ridge=shape_ridge, pairs_per_point=pairs_per_point)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = {f"hog_{key}": value for key, value in hog.items()}
    archive.update({f"shape_{key}": value for key, value in shape.items()})
    np.savez_compressed(output / "landmark.npz", **archive)
    training_labels = ("human_reviewed" if all(row.get("reviewed_landmarks") is True
                                                 for row in train_rows)
                       else "contains_unreviewed_prelabels")
    config = {
        "schema_version": 1, "model_type": "hog_shape_ensemble",
        "landmark_count": 28, "landmark_order": LANDMARK_ORDER,
        "hog_weight": hog_weight, "pca_dim": pca_dim, "hog_ridge": hog_ridge,
        "shape_rounds": shape_rounds, "shape_ridge": shape_ridge,
        "pairs_per_point": pairs_per_point, "coordinate_convention": "xyxy-exclusive",
        "nme_normalization": "interocular; bbox diagonal fallback",
        "training_labels": training_labels,
    }
    write_json(output / "config.json", config)
    regressor = LandmarkRegressor(output)
    metrics = {**config, "manifest": str(Path(manifest).resolve()),
               "training_samples": len(train_rows)}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split and "landmarks" in row]
        values = []
        for row in selected:
            prediction = regressor.predict(row["_image"], [row["bbox"]])[0]["landmarks"]
            details = nme_details(prediction, row["landmarks"], row["visibility"], row["bbox"])
            if details["nme"] is not None:
                values.append(details["nme"])
        metrics[f"{split}_nme"] = float(np.mean(values)) if values else None
        metrics[f"{split}_evaluated_samples"] = len(values)
        metrics[f"{split}_human_ground_truth"] = all(
            row.get("reviewed_landmarks") is True for row in selected)
    write_json(output / "metrics.json", metrics)
    print(f"Ensemble landmark model saved to {output}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hog-weight", type=float, default=0.5)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--hog-ridge", type=float, default=10.0)
    parser.add_argument("--shape-rounds", type=int, default=5, choices=(3, 4, 5))
    parser.add_argument("--shape-ridge", type=float, default=100.0)
    parser.add_argument("--pairs-per-point", type=int, default=24)
    args = parser.parse_args()
    train(args.manifest, args.output, args.hog_weight, args.pca_dim, args.hog_ridge,
          args.shape_rounds, args.shape_ridge, args.pairs_per_point)


if __name__ == "__main__":
    main()
