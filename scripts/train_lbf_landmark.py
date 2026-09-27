"""Train the advanced cascaded Fern/LBF 28-point landmark model."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data_io import load_manifest, write_json
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order
from src.lbf_landmark import predict_lbf_landmark, train_lbf_landmark


def train(manifest, output, seed=42, rounds=4, ferns_per_point=4,
          fern_depth=4, ridge=10.0, learning_rate=0.5,
          feature_radius=0.12, evaluate_test=False, augment_flip=True):
    rows = list(load_manifest(manifest))
    train_rows = [row for row in rows if row["split"] == "train"
                  and row["label"] == 1 and "landmarks" in row
                  and any(row["visibility"])]
    if not train_rows:
        raise ValueError("No positive training rows with visible landmarks")
    for row in [value for value in rows if "landmarks" in value]:
        validate_landmark_order(row.get("landmark_order"))

    images = [cv2.cvtColor(row["_image"], cv2.COLOR_BGR2GRAY)
              for row in train_rows]
    visibility = [row["visibility"] for row in train_rows]
    confidence = [row.get("landmark_weights", [1.0] * 28)
                  for row in train_rows]
    model = train_lbf_landmark(
        images, [row["bbox"] for row in train_rows],
        [row["landmarks"] for row in train_rows], visibility,
        landmark_weights=confidence, seed=seed, rounds=rounds,
        ferns_per_point=ferns_per_point, fern_depth=fern_depth, ridge=ridge,
        learning_rate=learning_rate, feature_radius=feature_radius,
        augment_flip=augment_flip)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "landmark.npz", **model)
    training_labels = ("human_reviewed" if all(
        row.get("reviewed_landmarks") is True for row in train_rows)
        else "contains_unreviewed_prelabels")
    config = {
        "schema_version": 1, "model_type": "lbf_fern",
        "landmark_count": 28, "landmark_order": LANDMARK_ORDER,
        "seed": seed, "rounds": rounds,
        "ferns_per_point": ferns_per_point, "fern_depth": fern_depth,
        "ridge": ridge, "learning_rate": learning_rate,
        "feature_radius": feature_radius,
        "augment_flip": augment_flip,
        "loss": "visibility_x_landmark_weighted_ridge",
        "coordinate_convention": "xyxy-exclusive",
        "nme_normalization": "interocular; bbox diagonal fallback",
        "training_labels": training_labels,
    }
    write_json(output / "config.json", config)

    metrics = {**config, "manifest": str(Path(manifest).resolve()),
               "training_samples": len(train_rows)}
    evaluated_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    for split in evaluated_splits:
        selected = [row for row in rows if row["split"] == split
                    and row["label"] == 1 and "landmarks" in row]
        values, normalization_counts = [], {}
        for row in selected:
            gray = cv2.cvtColor(row["_image"], cv2.COLOR_BGR2GRAY)
            prediction = predict_lbf_landmark(model, gray, row["bbox"])
            details = nme_details(prediction, row["landmarks"],
                                  row["visibility"], row["bbox"])
            if details["nme"] is not None:
                values.append(details["nme"])
                method = details["normalization"]
                normalization_counts[method] = normalization_counts.get(method, 0) + 1
        metrics[f"{split}_nme"] = float(np.mean(values)) if values else None
        metrics[f"{split}_median_nme"] = float(np.median(values)) if values else None
        metrics[f"{split}_samples"] = len(selected)
        metrics[f"{split}_normalization_counts"] = normalization_counts
        metrics[f"{split}_human_ground_truth"] = all(
            row.get("reviewed_landmarks") is True for row in selected)
    metrics["ground_truth_evaluation"] = metrics.get("test_human_ground_truth", False)
    metrics["test_reserved_during_selection"] = not evaluate_test
    write_json(output / "metrics.json", metrics)
    print(f"Fern/LBF landmark model saved to {output}")
    print({key: value for key, value in metrics.items()
           if key.endswith("_nme") or key == "training_samples"})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4, choices=(3, 4, 5))
    parser.add_argument("--ferns-per-point", type=int, default=4)
    parser.add_argument("--fern-depth", type=int, default=4, choices=range(2, 9))
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--feature-radius", type=float, default=0.12)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate test only after validation selection")
    parser.add_argument("--no-flip-augmentation", action="store_true")
    args = parser.parse_args()
    train(args.manifest, args.output, args.seed, args.rounds,
          args.ferns_per_point, args.fern_depth, args.ridge,
          args.learning_rate, args.feature_radius, args.evaluate_test,
          not args.no_flip_augmentation)


if __name__ == "__main__":
    main()
