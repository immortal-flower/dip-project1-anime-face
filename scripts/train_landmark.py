"""Train the course shape regressor from positive 28-point annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data_io import load_manifest, write_json
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order
from src.shape_regression import predict_shape, train_shape


def train(manifest: str, output: str, rounds: int, ridge: float, seed: int,
          pairs_per_point: int) -> None:
    rows = list(load_manifest(manifest))
    train_rows = [r for r in rows if r["split"] == "train" and r["label"] == 1
                  and "landmarks" in r and any(r["visibility"])]
    if not train_rows:
        raise ValueError("No positive training rows with landmarks")
    for row in [r for r in rows if "landmarks" in r]:
        validate_landmark_order(row.get("landmark_order"))
    images = [cv2.cvtColor(r["_image"], cv2.COLOR_BGR2GRAY) for r in train_rows]
    model = train_shape(images, [r["bbox"] for r in train_rows],
                        [r["landmarks"] for r in train_rows],
                        [r["visibility"] for r in train_rows],
                        seed=seed, rounds=rounds, ridge=ridge,
                        pairs_per_point=pairs_per_point)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "landmark.npz", **model)
    training_labels = ("human_reviewed" if all(r.get("reviewed_landmarks") is True
                                                for r in train_rows)
                       else "contains_unreviewed_prelabels")
    metrics = {"schema_version": 1, "manifest": str(Path(manifest).resolve()),
               "training_samples": len(train_rows), "rounds": rounds,
               "ridge": ridge, "seed": seed,
               "pairs_per_point": pairs_per_point,
               "landmark_order": LANDMARK_ORDER,
               "training_labels": training_labels}
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split and r["label"] == 1
                    and "landmarks" in r]
        if not selected:
            continue
        values, normalization_counts = [], {}
        for row in selected:
            gray = cv2.cvtColor(row["_image"], cv2.COLOR_BGR2GRAY)
            predicted = predict_shape(model, gray, row["bbox"])
            details = nme_details(predicted, np.asarray(row["landmarks"]),
                                  np.asarray(row["visibility"]), row["bbox"])
            if details["nme"] is not None:
                values.append(details["nme"])
                method = details["normalization"]
                normalization_counts[method] = normalization_counts.get(method, 0) + 1
        metrics[f"{split}_nme"] = float(np.mean(values)) if values else None
        metrics[f"{split}_samples"] = len(selected)
        metrics[f"{split}_normalization_counts"] = normalization_counts
        metrics[f"{split}_human_ground_truth"] = all(
            r.get("reviewed_landmarks") is True for r in selected)
    metrics["ground_truth_evaluation"] = metrics.get("test_human_ground_truth", False)
    metrics["warning"] = (None if metrics["ground_truth_evaluation"] else
        "At least one evaluated label is not human-reviewed; NME is not final accuracy.")
    write_json(output_path / "metrics.json", metrics)
    write_json(output_path / "config.json", {
        "schema_version": 1,
        "landmark_count": 28,
        "rounds": rounds,
        "ridge": ridge,
        "seed": seed,
        "pairs_per_point": pairs_per_point,
        "coordinate_convention": "xyxy-exclusive",
        "landmark_order": LANDMARK_ORDER,
        "nme_normalization": "interocular; bbox diagonal fallback",
        "training_labels": training_labels,
    })
    print(f"Landmark model saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3, choices=(3, 4, 5))
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-point", type=int, default=2)
    args = parser.parse_args()
    train(args.manifest, args.output, args.rounds, args.ridge, args.seed,
          args.pairs_per_point)


if __name__ == "__main__":
    main()
