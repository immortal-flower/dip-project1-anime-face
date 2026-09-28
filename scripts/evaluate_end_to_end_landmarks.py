"""Evaluate B face detections followed by C 28-point regression."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.data_io import load_manifest, write_image, write_json
from src.detection_metrics import iou
from src.detector import AnimeFaceDetector
from src.landmark_metrics import nme_details
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order


def evaluate(manifest, detector_model, landmark_model, output, split="test",
             iou_threshold=0.5, preview_count=24, allow_unreviewed=False):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    rows = [row for row in load_manifest(manifest)
            if row["split"] == split and row["label"] == 1
            and "landmarks" in row]
    if not rows:
        raise ValueError(f"No positive landmark rows in split {split!r}")
    for row in rows:
        validate_landmark_order(row.get("landmark_order"))
    if split == "test" and not allow_unreviewed:
        unreviewed = [row["source_id"] for row in rows
                      if row.get("reviewed_landmarks") is not True]
        if unreviewed:
            raise ValueError(f"Test landmarks require human review: {unreviewed[0]}")

    detector = AnimeFaceDetector(detector_model, landmark_model)
    output = Path(output)
    records, nmes, normalization_counts = [], [], Counter()
    true_positive = false_positive = false_negative = 0
    for index, row in enumerate(rows):
        detections = detector.detect(row["_image"])
        overlaps = [iou(item["bbox"], row["bbox"]) for item in detections]
        best_index = int(np.argmax(overlaps)) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        matched = best_index is not None and best_iou >= iou_threshold
        false_positive += len(detections) - int(matched)
        if matched:
            true_positive += 1
            selected = detections[best_index]
            details = nme_details(selected["landmarks"], row["landmarks"],
                                  row["visibility"], row["bbox"])
            if details["nme"] is not None:
                nmes.append(details["nme"])
                normalization_counts[details["normalization"]] += 1
        else:
            false_negative += 1
            selected, details = None, None
        records.append({
            "image": row["image"], "source_id": row["source_id"],
            "detections": len(detections), "matched": matched,
            "best_iou": float(best_iou),
            "nme": details["nme"] if details else None,
            "normalization": details["normalization"] if details else None,
        })

        if index < preview_count:
            canvas = cv2.resize(row["_image"], None, fx=4, fy=4,
                                interpolation=cv2.INTER_NEAREST)
            truth = np.asarray(row["landmarks"])
            visible = np.asarray(row["visibility"], dtype=bool)
            for point in truth[visible]:
                cv2.circle(canvas, tuple(np.rint(point * 4).astype(int)), 2,
                           (0, 220, 0), -1, cv2.LINE_AA)
            for item in detections:
                box = np.rint(np.asarray(item["bbox"]) * 4).astype(int)
                color = (0, 255, 255) if item is selected else (255, 160, 0)
                cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:] - 1), color, 1)
            if selected:
                for point in selected["landmarks"]:
                    cv2.circle(canvas, tuple(np.rint(np.asarray(point) * 4).astype(int)),
                               2, (0, 0, 255), -1, cv2.LINE_AA)
            write_image(output / "images" / f"{index:04d}.jpg", canvas)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    result = {
        "schema_version": 1, "split": split,
        "landmark_order": LANDMARK_ORDER, "iou_threshold": iou_threshold,
        "sample_count": len(rows), "tp": true_positive,
        "fp": false_positive, "fn": false_negative,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
              if precision + recall else 0.0,
        "matched_landmark_count": len(nmes),
        "matched_mean_nme": float(np.mean(nmes)) if nmes else None,
        "matched_median_nme": float(np.median(nmes)) if nmes else None,
        "normalization_counts": dict(normalization_counts),
        "human_ground_truth": all(row.get("reviewed_landmarks") is True for row in rows),
        "records": records,
    }
    write_json(output / "metrics.json", result)
    print({key: value for key, value in result.items() if key != "records"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--landmark-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    evaluate(args.manifest, args.detector_model, args.landmark_model, args.output,
             args.split, args.iou_threshold, args.preview_count,
             args.allow_unreviewed)


if __name__ == "__main__":
    main()
