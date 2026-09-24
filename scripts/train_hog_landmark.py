"""Train the optional HOG + PCA + Ridge 28-point comparison model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data_io import load_manifest, write_json
from src.hog_landmark import predict_hog_landmark, train_hog_landmark
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order


def train(manifest, output, pca_dim=48, ridge=10.0, augment_flip=True):
    rows = list(load_manifest(manifest))
    train_rows = [row for row in rows if row["split"] == "train" and row["label"] == 1
                  and "landmarks" in row and any(row["visibility"])]
    if not train_rows:
        raise ValueError("No positive training rows with landmarks")
    for row in [row for row in rows if "landmarks" in row]:
        validate_landmark_order(row.get("landmark_order"))
    model = train_hog_landmark(
        [row["_image"] for row in train_rows],
        [row["bbox"] for row in train_rows],
        [row["landmarks"] for row in train_rows],
        [row["visibility"] for row in train_rows],
        pca_dim=pca_dim, ridge=ridge, augment_flip=augment_flip,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "landmark.npz", **model)
    training_labels = ("human_reviewed" if all(row.get("reviewed_landmarks") is True
                                                 for row in train_rows)
                       else "contains_unreviewed_prelabels")
    metrics = {
        "schema_version": 1,
        "manifest": str(Path(manifest).resolve()),
        "model_type": "hog_ridge",
        "training_samples": len(train_rows),
        "pca_dim": int(model["pca_components"].shape[0]),
        "ridge": ridge,
        "horizontal_flip_augmentation": augment_flip,
        "landmark_order": LANDMARK_ORDER,
        "training_labels": training_labels,
    }
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split and row["label"] == 1
                    and "landmarks" in row]
        values, counts = [], {}
        for row in selected:
            prediction = predict_hog_landmark(model, row["_image"], row["bbox"])
            details = nme_details(prediction, row["landmarks"], row["visibility"], row["bbox"])
            if details["nme"] is not None:
                values.append(details["nme"])
                method = details["normalization"]
                counts[method] = counts.get(method, 0) + 1
        metrics[f"{split}_nme"] = float(np.mean(values)) if values else None
        metrics[f"{split}_samples"] = len(selected)
        metrics[f"{split}_normalization_counts"] = counts
        metrics[f"{split}_human_ground_truth"] = all(
            row.get("reviewed_landmarks") is True for row in selected)
    metrics["ground_truth_evaluation"] = metrics.get("test_human_ground_truth", False)
    metrics["warning"] = (None if metrics["ground_truth_evaluation"] else
        "At least one evaluated label is not human-reviewed; NME is not final accuracy.")
    write_json(output / "metrics.json", metrics)
    write_json(output / "config.json", {
        "schema_version": 1,
        "model_type": "hog_ridge",
        "landmark_count": 28,
        "landmark_order": LANDMARK_ORDER,
        "pca_dim": int(model["pca_components"].shape[0]),
        "ridge": ridge,
        "horizontal_flip_augmentation": augment_flip,
        "coordinate_convention": "xyxy-exclusive",
        "nme_normalization": "interocular; bbox diagonal fallback",
        "training_labels": training_labels,
    })
    print(f"HOG landmark model saved to {output}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pca-dim", type=int, default=48)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()
    train(args.manifest, args.output, args.pca_dim, args.ridge, not args.no_flip)


if __name__ == "__main__":
    main()
