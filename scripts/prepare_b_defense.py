"""Build B defense tables, figures, case metadata, and frozen-number evidence.

This script never trains or scans a detector.  It only reads already frozen
results plus the new validation-only step/scale runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.data_io import read_image, write_image, write_json
from src.detection_metrics import iou


ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data/processed/manga109_detection_v2_margin10/pages.json"
V2_RESULT = ROOT / "results/optimization-v2/manga109-test-final-108.json"
OLD_RESULT = ROOT / "results/b-final/test-full-108pages.json"
V2_MODEL = ROOT / "results/optimization-v2/final-model"
OLD_MODEL = ROOT / "results/b-final/model"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left = (width - resized.shape[1]) // 2
    top = (height - resized.shape[0]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def aggregate_scan(result):
    stages = []
    raw_candidates = after_nms = after_threshold = windows = 0
    for page in result["pages"]:
        windows += int(page["windows"])
        after_threshold += int(page["detections"])
        for level in page["scan"]:
            for index, item in enumerate(level.get("stages", [])):
                while len(stages) <= index:
                    stages.append(dict(stage=len(stages), evaluated=0, passed=0,
                                       rejected=0, seconds=0.0))
                for key in ("evaluated", "passed", "rejected"):
                    stages[index][key] += int(item[key])
                stages[index]["seconds"] += float(item["seconds"])
            summary = level.get("scan_summary")
            if summary:
                raw_candidates += int(summary["raw_candidates"])
                after_nms += int(summary["detections_after_nms"])
    return dict(total_windows=windows, stage_totals=stages,
                raw_candidates=raw_candidates, detections_after_nms=after_nms,
                detections_after_score_filter=after_threshold)


def summarize_step_scale(output):
    directory = output / "step-scale"
    specifications = [
        ("step", 1, 1.2, "step1-scale1.2.json"),
        ("step", 2, 1.2, "step2-scale1.2.json"),
        ("scale", 2, 1.1, "step2-scale1.1.json"),
        ("scale", 2, 1.2, "step2-scale1.2.json"),
        ("scale", 2, 1.3, "step2-scale1.3.json"),
    ]
    unique = {}
    rows = []
    for comparison, step, scale, filename in specifications:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Step/scale run is missing: {path}")
        result = read_json(path)
        if result["split"] != "val":
            raise ValueError(f"Defense experiment must be val-only: {path}")
        if result["selected_page_ids"] != [
            "manga109:HanzaiKousyouninMinegishiEitarou:061"
        ]:
            raise ValueError(f"Unexpected validation page selection: {path}")
        aggregate = aggregate_scan(result)
        metrics = result["metrics"]
        row = dict(
            comparison=comparison, step=step, scale_factor=scale,
            page_ids=result["selected_page_ids"], page_count=metrics["page_count"],
            windows=aggregate["total_windows"],
            stage_totals=aggregate["stage_totals"],
            raw_candidates=aggregate["raw_candidates"],
            detections_after_nms=aggregate["detections_after_nms"],
            detections_after_score_filter=aggregate["detections_after_score_filter"],
            tp=metrics["tp"], fp=metrics["fp"], fn=metrics["fn"],
            precision=metrics["precision"], recall=metrics["recall"],
            f1=metrics["f1"], seconds=result["seconds"],
            seconds_per_page=result["seconds"] / metrics["page_count"],
            result_file=str(path.relative_to(ROOT)).replace("\\", "/"),
        )
        rows.append(row)
        unique[filename] = row

    baseline = unique["step2-scale1.2.json"]
    dense = unique["step1-scale1.2.json"]
    conclusions = [
        (f"At scale 1.2, step 1 used {dense['windows']/baseline['windows']:.2f}x "
         f"the windows and {dense['seconds']/baseline['seconds']:.2f}x the time of step 2."),
        "This is one fixed validation page, so the measured differences are local evidence only.",
    ]
    summary = dict(
        schema_version=1, split="val", model="V2 frozen detector",
        selection_rule="existing deterministic source-spread order; max_pages=1",
        page_ids=rows[0]["page_ids"], random_seed=42,
        unique_run_count=len(unique), comparison_row_count=len(rows),
        frozen_settings=read_json(V2_MODEL / "config.json"),
        rows=rows, conclusions=conclusions,
    )
    write_json(directory / "summary.json", summary)
    lines = [
        "# V2 validation-only step / scale comparison", "",
        "The same frozen validation page and V2 parameters were used in every run. "
        "The `(2, 1.2)` run appears twice because it is the shared baseline for both requested comparisons.",
        "", "| comparison | step | scale | windows | TP | FP | FN | Precision | Recall | F1 | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['step']} | {row['scale_factor']:.1f} | "
            f"{row['windows']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['seconds']:.2f} |"
        )
    lines += ["", "## Scope-safe conclusion", ""] + [f"- {item}" for item in conclusions]
    (directory / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


HARD_NEGATIVES = [
    dict(manifest="results/hard-negatives/round-1-preview/review/reviewed_manifest.json",
         index=3, category="dialogue_text", round=1,
         explanation="对白框中的字符形成眼睛或眉毛般的局部黑白梯度。"),
    dict(manifest="results/hard-negatives/round-2-diverse/review/reviewed_manifest.json",
         index=2, category="high_contrast_hair", round=2,
         explanation="成束高对比斜线与动漫刘海、眼周边缘的局部结构相似。"),
    dict(manifest="results/hard-negatives/round-1-preview/review/reviewed_manifest.json",
         index=1, category="clothing_texture", round=1,
         explanation="衣服条纹与轮廓在小窗口内形成近似五官的明暗组合。"),
    dict(manifest="results/hard-negatives/round-2-diverse/review/reviewed_manifest.json",
         index=4, category="dense_manga_lines", round=2,
         explanation="密集竖线和斜边产生大量强梯度，触发局部像素差弱分类器。"),
    dict(manifest="results/optimization-v2/hnm-train/review/reviewed_manifest.json",
         index=11, category="eye_like_local_structure", round=3,
         explanation="衣领扣件的双圆形结构在低分辨率下呈现类似双眼的布局。"),
    dict(manifest="results/hard-negatives/round-2-diverse/review/reviewed_manifest.json",
         index=9, category="building_object_outline", round=2,
         explanation="屋顶和物体交界形成封闭轮廓及中心暗区，外观近似脸部布局。"),
]


def context_crop(page, box):
    x1, y1, x2, y2 = map(int, box)
    margin = 2 * max(x2 - x1, y2 - y1)
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right, bottom = min(page.shape[1], x2 + margin), min(page.shape[0], y2 + margin)
    result = page[top:bottom, left:right].copy()
    cv2.rectangle(result, (x1 - left, y1 - top),
                  (x2 - left - 1, y2 - top - 1), (0, 0, 255), 2)
    return result


def build_hard_negative_examples(output):
    directory = output / "hard-negatives"
    directory.mkdir(parents=True, exist_ok=True)
    pages = {row["page_id"]: row for row in read_json(PAGES)}
    metadata, tiles = [], []
    for number, spec in enumerate(HARD_NEGATIVES, 1):
        manifest_path = ROOT / spec["manifest"]
        row = read_json(manifest_path)[spec["index"] - 1]
        if row.get("review_status") != "accepted":
            raise ValueError(f"Hard negative #{number} was not accepted")
        page_row = pages[row["page_id"]]
        page = read_image((PAGES.parent / page_row["image"]).resolve())
        crop = read_image((manifest_path.parent / row["image"]).resolve())
        context = context_crop(page, row["origin_bbox"])
        crop_name = f"hn_{number:02d}_crop.png"
        context_name = f"hn_{number:02d}_context.png"
        write_image(directory / crop_name, crop)
        write_image(directory / context_name, context)

        tile = np.full((340, 560, 3), 255, dtype=np.uint8)
        tile[42:292, 0:380] = fit(context, 380, 250)
        tile[82:242, 390:550] = fit(crop, 160, 160)
        cv2.putText(tile, f"HN{number:02d} {spec['category']}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(tile, f"round={spec['round']} score={row['detector_score']:.2f}",
                    (8, 320), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 20, 20), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
        metadata.append(dict(
            id=f"hn_{number:02d}", category=spec["category"],
            crop=crop_name, context=context_name, page_id=row["page_id"],
            source_id=row["source_id"], origin_bbox=row["origin_bbox"],
            detector_score=row.get("detector_score"), hard_negative_round=spec["round"],
            review_decision=row["review_status"], review_method=row.get("review_method"),
            review_note=row.get("review_note"), explanation=spec["explanation"],
            source_manifest=spec["manifest"], source_index=spec["index"],
        ))
    sheet = np.full((680, 1680, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, 3)
        sheet[row * 340:(row + 1) * 340, column * 560:(column + 1) * 560] = tile
    write_image(directory / "hard_negatives_sheet.png", sheet)
    write_json(directory / "metadata.json", metadata)
    lines = ["# Six reviewed hard negatives", "",
             "All six entries have `review_status=accepted`; rejected face-like candidates are excluded.", ""]
    for item in metadata:
        lines += [f"## {item['id']}: {item['category']}", "",
                  f"- Page: `{item['page_id']}`", f"- Score: {item['detector_score']:.4f}",
                  f"- Round: {item['hard_negative_round']}",
                  f"- Explanation: {item['explanation']}", ""]
    (directory / "notes.md").write_text("\n".join(lines), encoding="utf-8")
    return metadata


DETECTION_CASES = [
    ("success", "manga109:UltraEleven:036",
     "V2 frozen test page with the highest TP count; selected match has the highest IoU."),
    ("success", "manga109:DollGun:025",
     "V2 frozen test page with the second-highest TP count; selected match has the highest IoU."),
    ("fp", "manga109:YoumaKourin:099",
     "V2 frozen test page with the largest FP count; selected FP has the highest score."),
    ("fp", "manga109:Joouari:001",
     "High-FP page with zero TP; selected FP has the highest score."),
    ("fn", "manga109:MoeruOnisan_vol01:068",
     "V2 frozen test page with the largest FN count; selected FN is the smallest unmatched GT."),
    ("fn", "manga109:TetsuSan:028",
     "Another high-FN page from a different source; selected FN is the smallest unmatched GT."),
]


def match_detections(predictions, truths, threshold=.5):
    unmatched = set(range(len(truths)))
    matches = {}
    for pred_index in sorted(range(len(predictions)),
                             key=lambda index: predictions[index]["score"], reverse=True):
        candidates = [(iou(predictions[pred_index]["bbox"], truths[index]), index)
                      for index in unmatched]
        if candidates:
            overlap, truth_index = max(candidates)
            if overlap >= threshold:
                unmatched.remove(truth_index)
                matches[pred_index] = (truth_index, overlap)
    return matches, unmatched


def annotate_case(image, truths, predictions, case_type, selected_pred, selected_truth,
                  title):
    canvas = image.copy()
    matches, unmatched_truths = match_detections(predictions, truths)
    for index, box in enumerate(truths):
        color = (0, 140, 255) if index in unmatched_truths else (0, 200, 0)
        thickness = 4 if index == selected_truth else 2
        x1, y1, x2, y2 = map(lambda value: int(round(value)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      (255, 0, 255) if index == selected_truth else color, thickness)
    for index, prediction in enumerate(predictions):
        color = (255, 180, 0) if index in matches else (0, 0, 255)
        thickness = 4 if index == selected_pred else 2
        x1, y1, x2, y2 = map(lambda value: int(round(value)), prediction["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      (255, 0, 255) if index == selected_pred else color, thickness)
        cv2.putText(canvas, f"{prediction['score']:.1f}", (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 46), (255, 255, 255), -1)
    cv2.putText(canvas, title, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, .68,
                (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def build_detection_cases(output):
    directory = output / "detection-cases"
    directory.mkdir(parents=True, exist_ok=True)
    pages = {row["page_id"]: row for row in read_json(PAGES)}
    result = read_json(V2_RESULT)
    metadata, thumbnails = [], []
    counters = {"success": 0, "fp": 0, "fn": 0}
    for case_type, page_id, reason in DETECTION_CASES:
        counters[case_type] += 1
        number = counters[case_type]
        page = pages[page_id]
        truths = page["bboxes"]
        predictions = result["predictions"][page_id]
        matches, unmatched_truths = match_detections(predictions, truths)
        unmatched_predictions = [index for index in range(len(predictions)) if index not in matches]
        selected_pred = selected_truth = metadata_truth = None
        selected_iou = 0.0
        if case_type == "success":
            selected_pred, (selected_truth, selected_iou) = max(
                matches.items(), key=lambda item: item[1][1])
            metadata_truth = selected_truth
        elif case_type == "fp":
            selected_pred = max(unmatched_predictions,
                                key=lambda index: predictions[index]["score"])
            if truths:
                selected_iou, metadata_truth = max(
                    (iou(predictions[selected_pred]["bbox"], box), index)
                    for index, box in enumerate(truths))
        else:
            selected_truth = min(
                unmatched_truths,
                key=lambda index: ((truths[index][2] - truths[index][0]) *
                                   (truths[index][3] - truths[index][1]), index),
            )
            metadata_truth = selected_truth
        metrics = result["metrics"]["per_page"][page_id]
        image = read_image((PAGES.parent / page["image"]).resolve())
        title = f"{case_type.upper()}  {page_id}  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
        annotated = annotate_case(image, truths, predictions, case_type,
                                  selected_pred, selected_truth, title)
        filename = f"{case_type}_{number:02d}.png"
        write_image(directory / filename, annotated)
        thumbnails.append(fit(annotated, 760, 560))
        selected_detection = predictions[selected_pred] if selected_pred is not None else None
        metadata.append(dict(
            case_type=case_type, file=filename, page_id=page_id,
            source_id=page["source_id"], split="test", frozen_result=str(
                V2_RESULT.relative_to(ROOT)).replace("\\", "/"),
            page_metrics=metrics,
            selected_detection_score=(selected_detection.get("score")
                                      if selected_detection else None),
            gt_bbox=(truths[metadata_truth] if metadata_truth is not None else None),
            predicted_bbox=(selected_detection["bbox"] if selected_detection else None),
            iou=float(selected_iou), selection_reason=reason,
            display_note="GT green; matched prediction cyan; FP red; FN orange; selected item magenta.",
        ))
    sheet = np.full((1680, 1520, 3), 255, dtype=np.uint8)
    for index, thumb in enumerate(thumbnails):
        row, column = divmod(index, 2)
        sheet[row * 560:(row + 1) * 560, column * 760:(column + 1) * 760] = thumb
    write_image(directory / "detection_cases_sheet.png", sheet)
    write_json(directory / "metadata.json", metadata)
    return metadata


def summarize_frozen_result(result_path, model_dir):
    result = read_json(result_path)
    detector = read_json(model_dir / "detector.json")
    aggregate = aggregate_scan(result)
    stages = aggregate["stage_totals"]
    return dict(
        source_result=str(result_path.relative_to(ROOT)).replace("\\", "/"),
        split=result["split"], page_count=result["metrics"]["page_count"],
        metrics={key: result["metrics"][key]
                 for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
        total_windows=aggregate["total_windows"], total_seconds=result["seconds"],
        seconds_per_page=result["seconds"] / result["metrics"]["page_count"],
        scan_config=result["scan_config"], raw_candidates=aggregate["raw_candidates"],
        detections_after_nms=aggregate["detections_after_nms"],
        detections_after_score_filter=aggregate["detections_after_score_filter"],
        stage_count=len(detector["stages"]),
        weak_tree_counts=[len(stage["trees"]) for stage in detector["stages"]],
        stage_totals=stages,
        stage0_early_rejection_rate=(stages[0]["rejected"] / stages[0]["evaluated"]),
        evaluation_input_detector_sha256=result.get("detector_sha256"),
        evaluation_input_config_sha256=result.get("config_sha256"),
        packaged_detector_sha256=sha256(model_dir / "detector.json"),
        packaged_config_sha256=sha256(model_dir / "config.json"),
    )


def integration_summary():
    try:
        raw = subprocess.check_output(
            ["git", "show", "origin/integration/b-c:docs/c_results/BC_END_TO_END_SUMMARY.json"],
            cwd=ROOT, text=True, encoding="utf-8",
        )
        return json.loads(raw)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_frozen_numbers(output, step_scale, hard_negatives, detection_cases):
    old = summarize_frozen_result(OLD_RESULT, OLD_MODEL)
    v2 = summarize_frozen_result(V2_RESULT, V2_MODEL)
    v2_config = read_json(V2_MODEL / "config.json")
    v2.update(
        threshold=v2_config["score_threshold"],
        nms_method=v2_config["nms_method"],
        min_box_support=v2_config["min_box_support"],
        box_calibration=v2_config["box_calibration"],
        hard_negative_policy=v2_config["enhanced_selection"],
        model_zip="deliverables/B_ENHANCED_MODEL.zip",
        model_zip_sha256=sha256(ROOT / "deliverables/B_ENHANCED_MODEL.zip"),
    )
    r1_mining = read_json(ROOT / "results/hard-negatives/round-1-preview/mining_summary.json")
    r1_review = read_json(ROOT / "results/hard-negatives/round-1-preview/review/review_summary.json")
    r2_mining = read_json(ROOT / "results/hard-negatives/round-2-diverse/mining_summary.json")
    r2_review = read_json(ROOT / "results/hard-negatives/round-2-diverse/review/review_summary.json")
    train_review = read_json(ROOT / "results/optimization-v2/hnm-train/review/review_summary.json")
    val_review = read_json(ROOT / "results/optimization-v2/hnm-val/review/review_summary.json")
    capacity = read_json(ROOT / "results/hard-negatives/round-2-diverse/capacity-512-vs-1024.json")
    weighted = read_json(ROOT / "results/hard-negatives/round-2-diverse/c1024-hnm-comparison.json")
    fpr_512 = capacity["independent_evaluation"]["validation_negative"]["original"][-1]["cumulative_pass_rate"]
    fpr_1024 = capacity["independent_evaluation"]["validation_negative"]["retrained"][-1]["cumulative_pass_rate"]
    fpr_weighted = weighted["independent_evaluation"]["validation_negative"]["retrained"][-1]["cumulative_pass_rate"]
    hard_negative_numbers = dict(
        round1=dict(mined=r1_mining["mined"], accepted=r1_review["accepted"],
                    rejected=r1_review["rejected"]),
        round2=dict(mined=r2_mining["mined"], accepted=r2_review["accepted"],
                    rejected=r2_review["rejected"]),
        first_two_rounds_accepted=r1_review["accepted"] + r2_review["accepted"],
        v2_new_train=dict(mined=train_review["reviewed"], accepted=train_review["accepted"],
                          rejected=train_review["rejected"]),
        v2_total_train_accepted=(r1_review["accepted"] + r2_review["accepted"] +
                                 train_review["accepted"]),
        v2_validation=dict(mined=val_review["reviewed"], accepted=val_review["accepted"],
                           rejected=val_review["rejected"], use="selection/calibration only"),
        test_hard_negatives=0,
        validation_fpr=dict(candidate512=fpr_512, candidate1024=fpr_1024,
                            candidate1024_hn_weight3=fpr_weighted),
        conclusion=("Hard-negative retraining was not monotonically beneficial: the 1024-feature "
                    "plain baseline had lower validation FPR than the weighted HN model, so the "
                    "plain 1024 model remained the old frozen detector. V2 used frozen early stages "
                    "and appended late stages under a separate recall-focused full-page protocol."),
    )
    protocol1 = integration_summary()
    protocol2_old = read_json(ROOT / "results/optimization-v2/c38-baseline-local-boxes.json")
    protocol2_v2 = read_json(ROOT / "results/optimization-v2/c38-final.json")
    bc = dict(
        protocol1=dict(
            name="integration/b-c old B frozen detector + C final 50/50 ensemble",
            evidence="origin/integration/b-c:docs/c_results/BC_END_TO_END_SUMMARY.json",
            bbox_protocol="AnimeFace cropped image; original bbox is the full crop [0,0,width,height]",
            summary=protocol1,
            nme_definition="interocular distance when both eyes are sufficiently visible; bbox diagonal fallback",
        ),
        protocol2=dict(
            name="feature/detection user-approved local face boxes",
            bbox_protocol="38 user-approved local face boxes; different from Protocol 1 full-image boxes",
            old_b=dict(detection_metrics=protocol2_old["detection_metrics"],
                       landmark_evaluable=protocol2_old["landmark_evaluable"],
                       mean_nme=protocol2_old["mean_nme"],
                       end_to_end_success_rate=protocol2_old["end_to_end_success_rate"]),
            v2_b=dict(detection_metrics=protocol2_v2["detection_metrics"],
                      landmark_evaluable=protocol2_v2["landmark_evaluable"],
                      mean_nme=protocol2_v2["mean_nme"],
                      end_to_end_success_rate=protocol2_v2["end_to_end_success_rate"]),
            nme_definition="mean visible-point error / sqrt(local box area)",
            landmark_model_warning=("The landmark.npz packaged by B is not C's final ensemble; these NME "
                                    "values are not directly comparable with Protocol 1 interocular NME."),
        ),
    )
    numbers = dict(
        schema_version=1, frozen_at="2026-09-29", model_input=[24, 24], channels=11,
        candidate_features=1024, training_seed=42,
        data_leakage_rule="train fits; val selects/reports small comparison; exposed test only reports frozen outputs",
        old_b_manga109_test=old, v2_b_manga109_test=v2,
        hard_negatives=hard_negative_numbers, animeface_protocols=bc,
        step_scale_validation=step_scale,
        selected_hard_negative_examples=hard_negatives,
        selected_detection_cases=detection_cases,
        never_mix=[
            "old 3-stage B detector and V2 5-stage detector",
            "Manga109 full-page detection and AnimeFace cropped-face detection",
            "AnimeFace full-image bbox and user-approved local-face bbox",
            "C interocular/bbox-diagonal-fallback NME and B helper sqrt(box-area) NME",
            "C final ensemble landmark model and B package landmark.npz",
            "automatic IoU metrics and any future human-reviewed metrics",
        ],
    )
    frozen = output / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    write_json(frozen / "DEFENSE_NUMBERS.json", numbers)
    write_json(output / "DEFENSE_NUMBERS.json", numbers)
    return numbers


def percent(value):
    return f"{100 * value:.2f}%"


def write_defense_doc(numbers):
    old = numbers["old_b_manga109_test"]
    v2 = numbers["v2_b_manga109_test"]
    hn = numbers["hard_negatives"]
    protocol1 = numbers["animeface_protocols"]["protocol1"]["summary"]
    protocol2 = numbers["animeface_protocols"]["protocol2"]
    step_rows = numbers["step_scale_validation"]["rows"]
    lines = [
        "# B 模块答辩前数字冻结与证据索引", "",
        "> 冻结日期：2026-09-29。Manga109 的 108 页和 AnimeFace 的 38 张 test 均已暴露，本文只汇总冻结结果；新增 step/scale 实验只使用 1 张预先固定的 validation 页。除 `integration/b-c` 的正式 C 汇总外，所有数字均可追溯到 `results/defense-prep/frozen/DEFENSE_NUMBERS.json` 及其中列出的原始 result。", "",
        "## 答辩推荐使用数字", "",
        "1. 检测输入窗口：24×24。",
        "2. 通道数：11。",
        "3. 候选像素差特征：1024。",
        f"4. 原冻结模型：3 个 Stage，弱树数 {old['weak_tree_counts']}。",
        f"5. V2：5 个 Stage，弱树数 {v2['weak_tree_counts']}；前三级完全冻结。",
        f"6. Stage 0 在 108 页 test 上早拒绝 {percent(old['stage0_early_rejection_rate'])} 的窗口。",
        f"7. 原模型 Manga109-108：P/R/F1={percent(old['metrics']['precision'])}/{percent(old['metrics']['recall'])}/{percent(old['metrics']['f1'])}。",
        f"8. V2 Manga109-108：P/R/F1={percent(v2['metrics']['precision'])}/{percent(v2['metrics']['recall'])}/{percent(v2['metrics']['f1'])}。",
        f"9. 原→V2 Recall：{percent(old['metrics']['recall'])}→{percent(v2['metrics']['recall'])}，约 {v2['metrics']['recall']/old['metrics']['recall']:.2f} 倍。",
        f"10. 前两轮 hard negatives：接受 {hn['first_two_rounds_accepted']} 个。",
        f"11. V2 新增 train hard negatives：接受 {hn['v2_new_train']['accepted']} 个；累计 train 接受 {hn['v2_total_train_accepted']} 个。",
        f"12. V2 val hard negatives：{hn['v2_validation']['accepted']} 个，只用于选择/校准；test 为 0。",
        f"13. C 最终 ensemble 在 oracle bbox 下 mean NME={protocol1['oracle_bbox']['mean_nme']:.4f}（C 的 interocular/必要时 bbox diagonal 协议）。",
        f"14. 旧 B + C 正式端到端成功：{protocol1['detected_bbox']['end_to_end_success_count']}/38={percent(protocol1['detected_bbox']['end_to_end_success_rate'])}。",
        "", "这些 Manga109 TP/FP/FN 是冻结预测与真值的 IoU≥0.5 自动一对一匹配，尚未完成逐页用户人工批准；答辩时应称为“冻结自动统计”。", "",
        "## 绝对不能混用的数字", "",
    ]
    lines += [f"- {item}。" for item in numbers["never_mix"]]
    lines += [
        "", "## 1. 课程要求的 step / scale 小型实验", "",
        "固定 validation 页：`manga109:HanzaiKousyouninMinegishiEitarou:061`。选择规则是现有 `source-spread` 顺序的第一张 val 页，先冻结后运行；没有依据效果挑页。模型、阈值、weighted NMS、min support=2、框校准和 IoU 都保持 V2 不变。", "",
        "| 对照 | step | scale | windows | TP | FP | FN | Precision | Recall | F1 | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in step_rows:
        lines.append(
            f"| {row['comparison']} | {row['step']} | {row['scale_factor']:.1f} | {row['windows']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['seconds']:.2f} |"
        )
    step1, step2 = step_rows[0], step_rows[1]
    lines += [
        "", f"在 scale=1.2 的这一页上，step 1 的窗口数是 step 2 的 {step1['windows']/step2['windows']:.2f} 倍，耗时是 {step1['seconds']/step2['seconds']:.2f} 倍。scale=1.1/1.2/1.3 的召回变化只反映这一固定页，不能外推为总体性能结论。每级 evaluated/passed/rejected、raw candidates 和 NMS 后数量保存在 `results/defense-prep/step-scale/summary.json`。", "",
        "## 2. 六个困难负样本图例", "",
        "| 编号 | 类型 | page_id | round | score | 为什么容易误检 |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in numbers["selected_hard_negative_examples"]:
        lines.append(
            f"| {item['id']} | {item['category']} | `{item['page_id']}` | "
            f"{item['hard_negative_round']} | {item['detector_score']:.2f} | {item['explanation']} |"
        )
    lines += [
        "", "六项的源清单均为 `review_status=accepted`；被复核剔除的疑似完整脸没有进入图例。拼图：`results/defense-prep/hard-negatives/hard_negatives_sheet.png`。", "",
        "## 3. 冻结检测案例", "",
        "| 类别 | page_id | TP/FP/FN | 选择规则 |",
        "|---|---|---:|---|",
    ]
    for item in numbers["selected_detection_cases"]:
        metric = item["page_metrics"]
        lines.append(
            f"| {item['case_type']} | `{item['page_id']}` | {metric['tp']}/{metric['fp']}/{metric['fn']} | {item['selection_reason']} |"
        )
    lines += [
        "", "案例全部来自已经冻结的 V2 Manga109 108 页 test JSON，只用于解释，不据此修改阈值、NMS、step、scale 或模型。成功页仍同时展示 FP/FN，避免只报最好看的局部。", "",
        "## 4. 冻结数字 A：原 B 模型，Manga109 108 test", "",
        f"- TP/FP/FN={old['metrics']['tp']}/{old['metrics']['fp']}/{old['metrics']['fn']}；P/R/F1={old['metrics']['precision']:.7f}/{old['metrics']['recall']:.7f}/{old['metrics']['f1']:.7f}。",
        f"- 窗口 {old['total_windows']}；总耗时 {old['total_seconds']:.2f}s；每页 {old['seconds_per_page']:.2f}s。",
        f"- step={old['scan_config']['step']}，scale={old['scan_config']['scale_factor']}，threshold={old['scan_config']['score_threshold']}，NMS IoU={old['scan_config']['nms_threshold']}。",
        f"- raw candidates={old['raw_candidates']}，NMS 后={old['detections_after_nms']}，阈值后={old['detections_after_score_filter']}。",
        f"- Stage 0/1/2 evaluated/passed/rejected：" + "; ".join(
            f"{s['evaluated']}/{s['passed']}/{s['rejected']}" for s in old['stage_totals']) + "。",
        f"- 评价输入 detector/config SHA256：`{old['evaluation_input_detector_sha256']}` / `{old['evaluation_input_config_sha256']}`。交付包 config 后续写入部署阈值，因此包内 config SHA256 为 `{old['packaged_config_sha256']}`，两者不可假设相同。", "",
        "## 5. 冻结数字 B：V2，Manga109 108 test", "",
        f"- TP/FP/FN={v2['metrics']['tp']}/{v2['metrics']['fp']}/{v2['metrics']['fn']}；P/R/F1={v2['metrics']['precision']:.7f}/{v2['metrics']['recall']:.7f}/{v2['metrics']['f1']:.7f}。",
        f"- 5 stages；threshold={v2['threshold']}；weighted NMS；min support={v2['min_box_support']}。",
        f"- box calibration：`{json.dumps(v2['box_calibration'], ensure_ascii=False)}`。",
        f"- 模型 ZIP SHA256：`{v2['model_zip_sha256']}`。",
        "- 数据边界：train hard negatives 拟合；val 只选择阈值/后处理/框校准；test hard negatives=0，冻结后只评价一次。", "",
        "## 6. 冻结数字 C：hard-negative 真实正负结果", "",
        f"- Round 1：mined/accepted/rejected={hn['round1']['mined']}/{hn['round1']['accepted']}/{hn['round1']['rejected']}。",
        f"- Round 2：mined/accepted/rejected={hn['round2']['mined']}/{hn['round2']['accepted']}/{hn['round2']['rejected']}。",
        f"- validation negative cumulative FPR：512 baseline={hn['validation_fpr']['candidate512']:.4f}；1024 baseline={hn['validation_fpr']['candidate1024']:.4f}；1024 + HN weight 3={hn['validation_fpr']['candidate1024_hn_weight3']:.4f}。",
        "- 困难负样本并非每轮都改善 validation；权重 3 模型仍差于普通 1024 模型，所以旧冻结模型保留普通 1024 版本。该负结果不得删除。", "",
        "## 7. 冻结数字 D：B+C / AnimeFace 两种协议", "",
        "### Protocol 1：正式 integration/b-c", "",
        f"- 旧 B detector + C 最终 50/50 ensemble；原 bbox 是 AnimeFace 裁剪整图。TP/FP/FN={protocol1['detected_bbox']['tp']}/{protocol1['detected_bbox']['fp']}/{protocol1['detected_bbox']['fn']}，P/R/F1={protocol1['detected_bbox']['precision']:.4f}/{protocol1['detected_bbox']['recall']:.4f}/{protocol1['detected_bbox']['f1']:.4f}。",
        f"- oracle mean NME={protocol1['oracle_bbox']['mean_nme']:.4f}；matched detected mean NME={protocol1['detected_bbox']['matched_mean_nme']:.4f}；端到端成功={protocol1['detected_bbox']['end_to_end_success_count']}/38={percent(protocol1['detected_bbox']['end_to_end_success_rate'])}。",
        "- NME 优先使用双眼中心距，眼点不足时回退 bbox diagonal。", "",
        "### Protocol 2：feature/detection 的用户批准局部脸框", "",
        f"- 原 B：TP/FP/FN={protocol2['old_b']['detection_metrics']['tp']}/{protocol2['old_b']['detection_metrics']['fp']}/{protocol2['old_b']['detection_metrics']['fn']}，P/R/F1={protocol2['old_b']['detection_metrics']['precision']:.4f}/{protocol2['old_b']['detection_metrics']['recall']:.4f}/{protocol2['old_b']['detection_metrics']['f1']:.4f}。",
        f"- V2：TP/FP/FN={protocol2['v2_b']['detection_metrics']['tp']}/{protocol2['v2_b']['detection_metrics']['fp']}/{protocol2['v2_b']['detection_metrics']['fn']}，P/R/F1={protocol2['v2_b']['detection_metrics']['precision']:.4f}/{protocol2['v2_b']['detection_metrics']['recall']:.4f}/{protocol2['v2_b']['detection_metrics']['f1']:.4f}。",
        "- `scripts/evaluate_c_end_to_end.py` 使用 mean visible error / sqrt(local box area)，且 B 包的 `landmark.npz` 不是 C 最终 ensemble；所以 Protocol 2 的 NME 不能与 Protocol 1 横向比较。", "",
        "## 8. 复现与证据", "",
        "- step/scale 命令：`results/defense-prep/step-scale/commands.txt`。",
        "- 完整本地冻结数字：`results/defense-prep/frozen/DEFENSE_NUMBERS.json`。",
        "- 六个 hard negatives：`results/defense-prep/hard-negatives/metadata.json`。",
        "- 六个检测案例：`results/defense-prep/detection-cases/metadata.json`。",
        "- 汇总脚本：`python -m scripts.prepare_b_defense`。", "",
        "## 9. 剩余风险", "",
        "- step/scale 只有 1 张固定 validation 页，仅满足小型验收和复杂度趋势展示，不能声称总体提升。",
        "- Manga109 108 页和两套 C-38 的 TP/FP/FN 目前是自动 IoU 匹配；用户逐页人工复核尚未完成。",
        "- V2 提升 Manga109 recall，但在局部框 C-38 上退化，说明仍有明显跨域和框语义问题。",
        "- test 已暴露，不能继续据其结果调整任何参数。", "",
    ]
    (ROOT / "docs/B_DEFENSE_PREP.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/defense-prep")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    step_scale = summarize_step_scale(output)
    hard_negatives = build_hard_negative_examples(output)
    detection_cases = build_detection_cases(output)
    numbers = build_frozen_numbers(output, step_scale, hard_negatives, detection_cases)
    write_defense_doc(numbers)
    print(json.dumps({
        "step_scale_rows": len(step_scale["rows"]),
        "hard_negative_examples": len(hard_negatives),
        "detection_cases": len(detection_cases),
        "old_metrics": numbers["old_b_manga109_test"]["metrics"],
        "v2_metrics": numbers["v2_b_manga109_test"]["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
