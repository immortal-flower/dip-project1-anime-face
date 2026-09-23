"""Generate reviewable 28-point pseudo labels for cropped AnimeFace images.

The output is deliberately marked as unreviewed model output.  It is useful for
testing and bootstrapping the landmark pipeline, but it is not ground truth.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image, write_json
from src.landmark_schema import LANDMARK_ORDER


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _select_images(root: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No images found under {root}")
    random.Random(seed).shuffle(files)
    return files[: min(limit, len(files))]


def _split(index: int, total: int) -> str:
    # Deterministic 75/10/15 split after the seeded shuffle.
    ratio = index / max(total, 1)
    if ratio < 0.75:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def _draw_preview(image: np.ndarray, points: np.ndarray, scores: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    radius = max(1, round(min(image.shape[:2]) / 90))
    for index, ((x, y), score) in enumerate(zip(points, scores), start=1):
        color = (0, 220, 0) if score >= 0.3 else (0, 165, 255)
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (center[0] + 2, center[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    return canvas


def generate(args: argparse.Namespace) -> None:
    source = Path(args.detector_source).resolve()
    if not (source / "anime_face_detector").is_dir():
        raise ValueError(f"anime-face-detector package not found under {source}")
    sys.path.insert(0, str(source))
    from anime_face_detector.detector import LandmarkDetector

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see the GPU")

    output = Path(args.output).resolve()
    preview_dir = output / "previews"
    output.mkdir(parents=True, exist_ok=True)
    files = _select_images(Path(args.images).resolve(), args.limit, args.seed)
    detector = LandmarkDetector(
        landmark_checkpoint_path=Path(args.landmark_model).resolve(),
        face_detector_name=None,
        device=args.device,
        flip_test=not args.no_flip_test,
        box_scale_factor=1.0,
    )

    rows = []
    confidence_values = []
    started = time.perf_counter()
    for index, image_path in enumerate(files):
        image = read_image(image_path)
        height, width = image.shape[:2]
        box = np.array([0, 0, width - 1, height - 1, 1.0], dtype=np.float32)
        result = detector(image, boxes=[box])[0]
        keypoints = np.asarray(result["keypoints"], dtype=float)
        if keypoints.shape != (28, 3) or not np.isfinite(keypoints).all():
            raise ValueError(f"Unexpected landmark output for {image_path}: {keypoints.shape}")
        points, scores = keypoints[:, :2], keypoints[:, 2]
        confidence_values.extend(scores.tolist())
        rows.append({
            "image": str(image_path),
            "source_id": f"animeface:{image_path.stem}",
            "split": _split(index, len(files)),
            "label": 1,
            "bbox": [0, 0, int(width), int(height)],
            "landmarks": points.tolist(),
            "visibility": [int(score >= args.visibility_threshold) for score in scores],
            "landmark_scores": scores.tolist(),
            "landmark_order": LANDMARK_ORDER,
            "annotation_status": "model_prelabel_unreviewed",
            "teacher_model": "anime-face-detector/hrnetv2",
        })
        if index < args.preview_count:
            write_image(preview_dir / f"{index:03d}_{image_path.stem}.jpg",
                        _draw_preview(image, points, scores))
        if (index + 1) % 25 == 0 or index + 1 == len(files):
            print(f"prelabelled {index + 1}/{len(files)}", flush=True)

    elapsed = time.perf_counter() - started
    write_json(output / "manifest.json", rows)
    write_json(output / "metadata.json", {
        "schema_version": 1,
        "annotation_status": "model_prelabel_unreviewed",
        "landmark_order": LANDMARK_ORDER,
        "ground_truth": False,
        "images_root": str(Path(args.images).resolve()),
        "teacher_model": str(Path(args.landmark_model).resolve()),
        "device": args.device,
        "flip_test": not args.no_flip_test,
        "seed": args.seed,
        "sample_count": len(rows),
        "split_counts": {name: sum(r["split"] == name for r in rows)
                         for name in ("train", "val", "test")},
        "visibility_threshold": args.visibility_threshold,
        "mean_landmark_score": float(np.mean(confidence_values)),
        "minimum_landmark_score": float(np.min(confidence_values)),
        "elapsed_seconds": elapsed,
        "images_per_second": len(rows) / elapsed,
        "warning": "Teacher-generated pseudo labels; manually review before reporting NME.",
    })
    print(json.dumps({"output": str(output), "samples": len(rows),
                      "seconds": round(elapsed, 2)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--detector-source", required=True)
    parser.add_argument("--landmark-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visibility-threshold", type=float, default=0.3)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--no-flip-test", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.preview_count < 0:
        parser.error("--limit must be positive and --preview-count cannot be negative")
    generate(args)


if __name__ == "__main__":
    main()
