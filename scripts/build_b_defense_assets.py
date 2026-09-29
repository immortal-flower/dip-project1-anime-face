"""Build traceable, PPT-ready figures from frozen B-module artifacts.

This script is deliberately read-only with respect to models and predictions:
it does not train, scan, tune, or evaluate a detector.  It renders figures from
already frozen JSON/report artifacts and copies representative annotated cases
created by ``scripts.prepare_b_defense``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from scripts.prepare_b_defense import (
    HARD_NEGATIVES,
    OLD_MODEL,
    OLD_RESULT,
    ROOT,
    V2_MODEL,
    V2_RESULT,
    aggregate_scan,
    build_detection_cases,
    build_hard_negative_examples,
    read_json,
    summarize_frozen_result,
)


OUT = ROOT / "results/defense-assets"
PREP = ROOT / "results/defense-prep"
OLD_COLOR = "#4C78A8"
V2_COLOR = "#F58518"
ACCENT = "#2A9D8F"
RED = "#E45756"
GRAY = "#6B7280"
LIGHT = "#EEF2F7"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 25,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, relative):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def pct(value):
    return f"{100 * value:.2f}%"


def label_bars(ax, bars, formatter=lambda value: f"{value:g}", pad=4):
    for bar in bars:
        ax.annotate(formatter(bar.get_height()),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, pad), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13, fontweight="bold")


def ensure_case_assets():
    hn_dir = PREP / "hard-negatives"
    case_dir = PREP / "detection-cases"
    if not (hn_dir / "metadata.json").is_file():
        build_hard_negative_examples(PREP)
    if not (case_dir / "metadata.json").is_file():
        build_detection_cases(PREP)
    return read_json(hn_dir / "metadata.json"), read_json(case_dir / "metadata.json")


def plot_metric_comparison(old, v2):
    labels = ["Precision", "Recall", "F1"]
    old_values = [old["metrics"][key] for key in ("precision", "recall", "f1")]
    v2_values = [v2["metrics"][key] for key in ("precision", "recall", "f1")]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(16, 9))
    width = .34
    b1 = ax.bar(x - width / 2, old_values, width, color=OLD_COLOR,
                label="Original frozen (3-stage)")
    b2 = ax.bar(x + width / 2, v2_values, width, color=V2_COLOR,
                label="Enhanced v2 (5-stage)")
    label_bars(ax, b1, pct)
    label_bars(ax, b2, pct)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, .20)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
    ax.set_ylabel("Micro metric (IoU >= 0.5)")
    ax.set_title("B Detector: Manga109 108-page Test")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, loc="upper left")
    ax.text(.5, -.16,
            "v2 mainly improves Recall and F1, at the cost of more False Positives.",
            transform=ax.transAxes, ha="center", fontsize=16, color="#374151")
    save(fig, "05_final_metrics/manga109_old_vs_v2.png")


def plot_counts(old, v2):
    labels = ["TP", "FP", "FN"]
    old_values = [old["metrics"][key] for key in ("tp", "fp", "fn")]
    v2_values = [v2["metrics"][key] for key in ("tp", "fp", "fn")]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(16, 9))
    width = .34
    b1 = ax.bar(x - width / 2, old_values, width, color=OLD_COLOR,
                label="Original frozen")
    b2 = ax.bar(x + width / 2, v2_values, width, color=V2_COLOR,
                label="Enhanced v2")
    label_bars(ax, b1, lambda value: f"{int(value):,}")
    label_bars(ax, b2, lambda value: f"{int(value):,}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 3000)
    ax.set_ylabel("Detection count")
    ax.set_title("Detection Counts on the Frozen Manga109 Test")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False)
    save(fig, "05_final_metrics/tp_fp_fn_comparison.png")


def plot_funnel(old):
    stages = old["stage_totals"]
    values = [old["total_windows"]] + [row["passed"] for row in stages]
    labels = ["All windows", "After Stage 0", "After Stage 1", "After Stage 2"]
    max_width = values[0]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-.82, .82)
    ax.set_ylim(-.25, 4.25)
    ax.axis("off")
    colors = ["#DCE8F5", "#AFCBE6", "#74A9D8"]
    for index in range(3):
        top = .5 * values[index] / max_width
        bottom = .5 * values[index + 1] / max_width
        y_top, y_bottom = 3.7 - index * 1.12, 2.75 - index * 1.12
        ax.add_patch(Polygon([(-top, y_top), (top, y_top),
                              (bottom, y_bottom), (-bottom, y_bottom)],
                             closed=True, facecolor=colors[index], edgecolor="white"))
        stage = stages[index]
        pass_rate = stage["passed"] / stage["evaluated"]
        ax.text(0, (y_top + y_bottom) / 2,
                f"Stage {index}\nEvaluated {stage['evaluated']:,}  |  "
                f"Passed {stage['passed']:,}  |  Rejected {stage['rejected']:,}\n"
                f"Conditional pass {pass_rate:.1%}",
                ha="center", va="center", fontsize=13, color="#102A43",
                fontweight="bold" if index == 0 else "normal")
    ax.text(0, 4.08, f"{labels[0]}: {values[0]:,}", ha="center", fontsize=17)
    ax.text(0, .03, f"Surviving windows: {values[-1]:,}", ha="center", fontsize=17)
    reject = stages[0]["rejected"] / stages[0]["evaluated"]
    ax.text(.79, 3.18, f"Stage 0 rejects\n{reject:.1%}", ha="right", va="center",
            fontsize=18, color=RED, fontweight="bold")
    ax.set_title("Cascade Early Rejection on Manga109 Test", pad=15)
    save(fig, "02_cascade/cascade_funnel.png")


def plot_stage_workload(old):
    stages = old["stage_totals"]
    labels = [f"Stage {row['stage']}" for row in stages]
    evaluated = np.array([row["evaluated"] for row in stages]) / 1e6
    seconds = [row["seconds"] for row in stages]
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(labels, evaluated, color=[OLD_COLOR, "#7BA6CE", "#AFCBE6"], width=.55)
    for index, (bar, value) in enumerate(zip(bars, evaluated)):
        ax.annotate(f"{value:.2f}M",
                    (bar.get_x() + bar.get_width() / 2, value - .28),
                    ha="center", va="top", fontsize=13, fontweight="bold",
                    color="white" if index == 0 else "#1F2937")
    ax.set_ylim(0, 15)
    ax.set_ylabel("Evaluated windows (millions)")
    ax.grid(axis="y", alpha=.2)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(labels, seconds, color=RED, marker="o", markersize=10, linewidth=3,
             label="Stage scoring time")
    for index, value in enumerate(seconds):
        ax2.annotate(f"{value:.1f} s", (index, value), xytext=(0, 12),
                     textcoords="offset points", ha="center", color=RED,
                     fontweight="bold")
    ax2.set_ylim(0, 650)
    ax2.set_ylabel("Stage scoring time (seconds)", color=RED)
    ax.set_title("Cascade Stage Workload (Original Frozen Model)")
    ax.text(.5, -.14, "Timing covers stage scoring only; feature extraction is excluded.",
            transform=ax.transAxes, ha="center", fontsize=15, color="#374151")
    save(fig, "02_cascade/stage_workload.png")


def plot_hnm_flow(hnm):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    cards = [
        (1.0, 5.1, 3.2, 2.3, "Round 1", f"Pages  {hnm['round1']['scanned_pages']}\nMined  120\nAccepted  109\nRejected  11", OLD_COLOR),
        (4.8, 5.1, 3.2, 2.3, "Round 2", f"Pages  {hnm['round2']['scanned_pages']}\nMined  120\nAccepted  106\nRejected  14", "#6B9AC4"),
        (8.6, 5.1, 3.2, 2.3, "V2 train", f"Accepted  287\nEarly rounds  215\nTotal train  502", V2_COLOR),
        (12.4, 5.1, 2.6, 2.3, "V2 val", "Accepted  88\nSelection only\nNo fitting", ACCENT),
    ]
    for x, y, w, h, title, body, color in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.12,rounding_size=.14",
                                    facecolor=color, edgecolor="none", alpha=.95))
        ax.text(x + w / 2, y + h - .45, title, ha="center", color="white",
                fontsize=18, fontweight="bold")
        ax.text(x + w / 2, y + .82, body, ha="center", va="center",
                color="white", fontsize=14, linespacing=1.45)
    for x1, x2 in [(4.2, 4.8), (8.0, 8.6), (11.8, 12.4)]:
        ax.add_patch(FancyArrowPatch((x1, 6.25), (x2, 6.25), arrowstyle="-|>",
                                    mutation_scale=22, color=GRAY, linewidth=2))
    ax.add_patch(FancyBboxPatch((4.6, 1.25), 6.8, 1.6,
                                boxstyle="round,pad=.12,rounding_size=.12",
                                facecolor="#FDECEC", edgecolor=RED, linewidth=2))
    ax.text(8, 2.2, "TEST HARD NEGATIVES = 0", ha="center", va="center",
            fontsize=23, fontweight="bold", color=RED)
    ax.text(8, 1.65, "Frozen test pages were never used for mining or tuning.",
            ha="center", fontsize=14, color="#6B1D1D")
    ax.set_title("Hard Negative Mining: Data Flow and Split Boundary")
    save(fig, "03_hard_negatives/hnm_flow.png")


def plot_hnm_validation():
    labels = ["512\nbaseline", "512 + HN\nweight 1", "1024\nbaseline", "1024 + HN\nweight 3"]
    values = [.4111, .4750, .3333, .3694]
    colors = [OLD_COLOR, "#9DB9D3", V2_COLOR, "#F8B36C"]
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(labels, values, color=colors, width=.62)
    label_bars(ax, bars, pct)
    ax.set_ylim(0, .55)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.0f}%")
    ax.set_ylabel("Cumulative validation FPR after Stage 2")
    ax.set_title("Hard Negative Mining: Actual Validation Outcome")
    ax.grid(axis="y", alpha=.22)
    ax.text(.5, -.14, "Some hard-negative configurations did not improve validation FPR.",
            transform=ax.transAxes, ha="center", fontsize=16, color="#374151")
    save(fig, "03_hard_negatives/hnm_validation_comparison.png")


def plot_hard_negative_grids(metadata):
    source = PREP / "hard-negatives"
    labels = ["Dialogue text", "Hair texture", "Clothing texture",
              "Dense manga lines", "Eye-like local", "Object outline"]
    for kind, field, filename in [
        ("crop", "crop", "hard_negative_grid.png"),
        ("context", "context", "hard_negative_context_grid.png"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, item, label in zip(axes.flat, metadata, labels):
            image = plt.imread(source / item[field])
            ax.imshow(image)
            ax.set_title(f"{label}  |  score {item['detector_score']:.2f}", fontsize=16)
            ax.axis("off")
        fig.suptitle("Reviewed Accepted Hard Negatives" +
                     (" — Context" if kind == "context" else " — Crops"),
                     fontsize=25, y=.98)
        fig.tight_layout(rect=(0, 0, 1, .95))
        save(fig, f"03_hard_negatives/{filename}")


def plot_detection_grid(metadata):
    source = PREP / "detection-cases"
    target = OUT / "04_detection_examples"
    target.mkdir(parents=True, exist_ok=True)
    order = [
        next(item for item in metadata if item["case_type"] == kind and item["file"].endswith(f"{index:02d}.png"))
        for index in (1, 2) for kind in ("success", "fp", "fn")
    ]
    for item in metadata:
        shutil.copy2(source / item["file"], target / item["file"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titles = ["Success / TP", "False Positive", "False Negative"]
    for index, (ax, item) in enumerate(zip(axes.flat, order)):
        ax.imshow(plt.imread(source / item["file"]))
        metric = item["page_metrics"]
        page_label = ":".join(item["page_id"].split(":")[-2:])
        heading = f"{titles[index % 3]}\n" if index < 3 else ""
        ax.set_title(f"{heading}{page_label} | {metric['tp']}/{metric['fp']}/{metric['fn']}",
                     fontsize=12, pad=5)
        ax.axis("off")
    fig.suptitle("Representative V2 Detection Cases — Frozen Manga109 Test", fontsize=24, y=.99)
    fig.text(.5, .01, "Green: matched GT | Cyan: matched prediction | Red: FP | Orange: FN | Magenta: selected item",
             ha="center", fontsize=13, color="#374151")
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.075,
                        wspace=.055, hspace=.18)
    save(fig, "04_detection_examples/detection_cases_grid.png")
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")


def plot_scan_ablation(summary):
    rows = summary["ablations"]
    labels = [f"step {r['scan_config']['step']}\nscale {r['scan_config']['scale_factor']}" for r in rows]
    f1 = [r["metrics"]["f1"] for r in rows]
    seconds = [r["seconds"] for r in rows]
    windows = [r["total_windows"] / 1000 for r in rows]
    colors = ["#AFCBE6", OLD_COLOR, "#F8B36C", V2_COLOR]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    bars = ax1.bar(labels, f1, color=colors)
    label_bars(ax1, bars, pct)
    ax1.set_ylim(0, .026)
    ax1.yaxis.set_major_formatter(lambda value, _pos: f"{value * 100:.1f}%")
    ax1.set_ylabel("F1 (before score threshold)")
    ax1.set_title("Accuracy")
    ax1.grid(axis="y", alpha=.2)
    x = np.arange(4)
    b2 = ax2.bar(x - .18, windows, .36, color="#7BA6CE", label="Windows (thousands)")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Windows (thousands)")
    ax2.set_ylim(0, 820)
    label_bars(ax2, b2, lambda value: f"{value:.0f}k")
    axr = ax2.twinx()
    axr.spines["right"].set_visible(True)
    axr.plot(x, seconds, color=RED, marker="o", linewidth=3, markersize=9)
    for index, value in enumerate(seconds):
        axr.annotate(f"{value:.1f}s", (index, value), xytext=(0, 10),
                     textcoords="offset points", ha="center", color=RED, fontweight="bold")
    axr.set_ylabel("Runtime (seconds)", color=RED)
    axr.set_ylim(0, 115)
    ax2.set_title("Computation")
    fig.suptitle("Existing Coarse Scan Ablation (step = 6 / 12)", fontsize=25, y=.98)
    fig.text(.5, .015, "Validation: 6 fixed source-spread pages. Denser scanning costs much more; gains are not linear.",
             ha="center", fontsize=14, color="#374151")
    fig.tight_layout(rect=(0, .04, 1, .94))
    save(fig, "06_ablation/existing_scan_ablation.png")


def plot_evolution():
    nodes = [
        ("Baseline 3-Stage", "Complete pipeline"),
        ("Hard Negative Mining", "Real false alarms"),
        ("1024 Features", "More discriminative capacity"),
        ("Late Stages 3–4", "Filter difficult windows"),
        ("Weighted NMS", "Merge overlapping windows"),
        ("Box Calibration", "Improve bbox geometry"),
        ("Enhanced v2", "Frozen 5-stage detector"),
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    positions = [(1, 5.4), (4.8, 5.4), (8.6, 5.4), (12.4, 5.4),
                 (3.0, 2.0), (7.0, 2.0), (11.0, 2.0)]
    for index, ((title, subtitle), (x, y)) in enumerate(zip(nodes, positions)):
        color = OLD_COLOR if index == 0 else (V2_COLOR if index == len(nodes) - 1 else "#E8EEF5")
        text_color = "white" if index in (0, len(nodes) - 1) else "#1F2937"
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.45,
                                    boxstyle="round,pad=.12,rounding_size=.15",
                                    facecolor=color, edgecolor="#CAD5E2", linewidth=1.5))
        ax.text(x + 1.35, y + .92, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color=text_color)
        ax.text(x + 1.35, y + .43, subtitle, ha="center", va="center",
                fontsize=11, color=text_color)
    arrow_pairs = list(zip(positions[:3], positions[1:4])) + [
        (positions[3], positions[4]), (positions[4], positions[5]), (positions[5], positions[6])]
    for start, end in arrow_pairs:
        sx, sy = start[0] + 2.7, start[1] + .72
        ex, ey = end[0], end[1] + .72
        if start == positions[3]:
            sx, sy, ex, ey = start[0] + 1.35, start[1], end[0] + 1.35, end[1] + 1.45
        ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>",
                                    mutation_scale=18, color=GRAY, linewidth=2,
                                    connectionstyle="arc3,rad=.05"))
    ax.set_title("B Detector Evolution: Baseline to Enhanced v2")
    save(fig, "01_model_overview/model_evolution.png")


def plot_key_card(old, v2, hnm):
    cards = [
        ("Window", "24 × 24"), ("Channels", "11"), ("Candidate features", "1024"),
        ("Original cascade", "3 stages"), ("Enhanced v2", "5 stages"),
        ("Stage 0 rejection", pct(old["stage0_early_rejection_rate"])),
        ("Original Recall / F1", f"{pct(old['metrics']['recall'])} / {pct(old['metrics']['f1'])}"),
        ("v2 Recall / F1", f"{pct(v2['metrics']['recall'])} / {pct(v2['metrics']['f1'])}"),
        ("Training hard negatives", f"215 early  |  {hnm['total_train_accepted']} total"),
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    for index, (title, value) in enumerate(cards):
        row, col = divmod(index, 3)
        x, y = .7 + col * 5.15, 5.95 - row * 2.25
        color = V2_COLOR if title.startswith("v2") or title == "Enhanced v2" else OLD_COLOR
        ax.add_patch(FancyBboxPatch((x, y), 4.6, 1.65,
                                    boxstyle="round,pad=.12,rounding_size=.15",
                                    facecolor="white", edgecolor=color, linewidth=2.5))
        ax.text(x + .25, y + 1.17, title, fontsize=13, color=GRAY, va="center")
        ax.text(x + .25, y + .55, value, fontsize=22, color=color,
                fontweight="bold", va="center")
    ax.set_title("B Detector — Key Numbers", pad=20)
    save(fig, "05_final_metrics/key_numbers_card.png")


def read_hnm_numbers():
    base = ROOT / "results"
    specs = {
        "round1": base / "hard-negatives/round-1-preview",
        "round2": base / "hard-negatives/round-2-diverse",
        "v2_train": base / "optimization-v2/hnm-train",
        "v2_val": base / "optimization-v2/hnm-val",
    }
    result = {}
    for name, directory in specs.items():
        mining = read_json(directory / "mining_summary.json")
        review = read_json(directory / "review/review_summary.json")
        result[name] = {
            "scanned_pages": mining["scanned_pages"],
            "mined": mining["mined"],
            "accepted": review["accepted"],
            "rejected": review["rejected"],
            "source_files": [
                str((directory / "mining_summary.json").relative_to(ROOT)).replace("\\", "/"),
                str((directory / "review/review_summary.json").relative_to(ROOT)).replace("\\", "/"),
            ],
        }
    result["first_two_rounds_accepted"] = result["round1"]["accepted"] + result["round2"]["accepted"]
    result["total_train_accepted"] = result["first_two_rounds_accepted"] + result["v2_train"]["accepted"]
    result["test_hard_negatives"] = 0
    result["validation_fpr"] = {
        "source": "docs/B_RETRAIN_COMPARISON.md",
        "source_type": "document",
        "candidate512_baseline": .4111,
        "candidate512_hn_weight1": .4750,
        "candidate1024_baseline": .3333,
        "candidate1024_hn_weight3": .3694,
    }
    return result


def git_identity():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                     text=True, encoding="utf-8").strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True, encoding="utf-8").strip()
    return branch, commit


def write_metadata(old, v2, hnm, hn_examples, detection_examples, summary):
    branch, commit = git_identity()
    metadata = {
        "schema_version": 1,
        "generated_from_frozen_artifacts": True,
        "repo": {"branch": branch, "commit": commit},
        "evaluation_protocol": {
            "dataset": "Manga109", "test_pages": 108, "iou_threshold": .5,
            "matching": "score-ordered one-to-one automatic matching",
            "test_used_for_tuning": False,
        },
        "original_detector": old,
        "enhanced_v2": v2,
        "cascade_stage_stats": {
            "model": "original frozen 3-stage detector",
            "split": "test", "source": old["source_result"],
            "stages": old["stage_totals"],
            "stage0_rejection_rate": old["stage0_early_rejection_rate"],
            "timing_scope": "stage scoring only; feature extraction excluded",
        },
        "hard_negative_mining": hnm,
        "selected_hard_negative_examples": hn_examples,
        "selected_detection_examples": detection_examples,
        "scan_ablation": {
            "source": "docs/b_results/B_FINAL_SUMMARY.json",
            "split": "val", "selection": "6 fixed source-spread pages",
            "rows": summary["ablations"],
            "warning": "This is the existing step=6/12 coarse ablation, not step=1/2.",
        },
        "asset_policy": {
            "old_color": OLD_COLOR, "v2_color": V2_COLOR,
            "new_training_run": False, "full_test_rerun": False,
            "test_parameter_changes": False,
        },
    }
    path = OUT / "metadata/DEFENSE_METRICS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "03_hard_negatives/metadata.json").write_text(
        json.dumps(hn_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def write_docs(metadata):
    old = metadata["original_detector"]
    v2 = metadata["enhanced_v2"]
    hnm = metadata["hard_negative_mining"]
    index = f"""# B 模块答辩素材索引

所有素材由 `python -m scripts.build_b_defense_assets` 从冻结工件生成。主指标均为 Manga109 108-page test、IoU≥0.5 自动一对一匹配；没有重训、test 调参或 108 页重跑。

## Manga109 old vs v2

文件：`05_final_metrics/manga109_old_vs_v2.png`  
建议放在 PPT：主讲第 1 页  
它回答的问题：V2 的整体检测指标发生了什么变化？  
一句话结论：Recall {pct(old['metrics']['recall'])}→{pct(v2['metrics']['recall'])}，F1 {pct(old['metrics']['f1'])}→{pct(v2['metrics']['f1'])}，但 FP 增加。  
数据来源：`{old['source_result']}`、`{v2['source_result']}`；Manga109 test；old 3-stage 与 v2 5-stage。  
注意事项：不可称为“全面提升”。

## Cascade early rejection

文件：`02_cascade/cascade_funnel.png`  
建议放在 PPT：主讲第 2 页  
它回答的问题：Cascade 为什么能减少后级工作量？  
一句话结论：原冻结模型 Stage 0 拒绝 {pct(old['stage0_early_rejection_rate'])} 的滑窗。  
数据来源：`{old['source_result']}`；Manga109 test；old 3-stage。  
注意事项：漏斗是 old frozen model，不是 v2。

## Hard-negative examples

文件：`03_hard_negatives/hard_negative_grid.png`、`hard_negative_context_grid.png`  
建议放在 PPT：主讲第 3 页  
它回答的问题：传统特征最容易把什么背景误认为人脸？  
一句话结论：文字、头发、衣纹、密集线条、眼状局部和物体轮廓均会触发误报。  
数据来源：Round 1/2 与 V2 train 的 reviewed manifests；train；accepted only。  
注意事项：上下文红框仅用于定位原始 crop。

## Detection cases

文件：`04_detection_examples/detection_cases_grid.png` 及 6 张单页图  
建议放在 PPT：主讲第 4 页  
它回答的问题：V2 的 TP、FP、FN 在真实页面上是什么样？  
一句话结论：提升 Recall 后仍存在高对比误报及小脸/尺度相关漏检。  
数据来源：`{v2['source_result']}`；Manga109 test；v2。  
注意事项：代表性解释案例，不代表总体平均；成功页也保留 FP/FN。

## Key numbers

文件：`05_final_metrics/key_numbers_card.png`  
建议放在 PPT：主讲总结页  
它回答的问题：B 模块最关键的结构和实验数字是什么？  
一句话结论：24×24、11 通道、1024 特征、3→5 stages、Stage 0 早拒绝约 69%。  
数据来源：冻结模型配置、108 页结果、HNM review summaries。  
注意事项：不含 C 的 NME。

## TP / FP / FN counts

文件：`05_final_metrics/tp_fp_fn_comparison.png`  
建议放在 PPT：Backup  
它回答的问题：Recall 的提升付出了什么代价？  
一句话结论：TP 69→247，同时 FP 960→2647。  
数据来源：old/v2 Manga109 test JSON。  
注意事项：线性 Y 轴，未截断。

## HNM flow

文件：`03_hard_negatives/hnm_flow.png`  
建议放在 PPT：Backup / HNM 讲解页  
它回答的问题：困难负样本如何保持数据边界？  
一句话结论：train accepted={hnm['total_train_accepted']}，val accepted={hnm['v2_val']['accepted']} 只用于选择，test=0。  
数据来源：各轮 mining/review summary。  
注意事项：validation HN 未用于拟合。

## HNM validation result

文件：`03_hard_negatives/hnm_validation_comparison.png`  
建议放在 PPT：Backup  
它回答的问题：HNM 是否单调改善泛化？  
一句话结论：部分 hard-negative 配置未改善 validation FPR。  
数据来源：`docs/B_RETRAIN_COMPARISON.md`（source=document）；validation region set。  
注意事项：不得删除负结果。

## Model evolution

文件：`01_model_overview/model_evolution.png`  
建议放在 PPT：主讲方法页或 Backup  
它回答的问题：V2 由哪些真实工程步骤构成？  
一句话结论：HNM、容量扩展、后级 Stage、weighted NMS 和框校准共同组成 V2。  
数据来源：`docs/B_OPTIMIZATION_V2_REPORT.md` 与冻结 config。  
注意事项：这是算法路线，不是因果消融。

## Existing scan ablation

文件：`06_ablation/existing_scan_ablation.png`  
建议放在 PPT：Backup  
它回答的问题：扫描更密是否线性换来更高指标？  
一句话结论：窗口数和耗时显著增加，F1 增益并非线性。  
数据来源：`docs/b_results/B_FINAL_SUMMARY.json`；6 fixed validation pages；old candidate model。  
注意事项：这是 step=6/12，不是 step=1/2。

## Stage workload

文件：`02_cascade/stage_workload.png`  
建议放在 PPT：Backup  
它回答的问题：每一级实际处理多少窗口、耗时多少？  
一句话结论：早拒绝使后级 evaluated windows 从 13.51M 降到 4.20M/3.64M。  
数据来源：old frozen Manga109 test stage logs。  
注意事项：stage scoring time 不包含 feature extraction。
"""
    (OUT / "PPT_ASSET_INDEX.md").write_text(index, encoding="utf-8")

    doc = f"""# B 模块答辩素材包

## 冻结身份

- Branch：`{metadata['repo']['branch']}`
- HEAD：`{metadata['repo']['commit']}`
- 输出：`results/defense-assets/`
- 生成命令：`python -m scripts.build_b_defense_assets`

本素材包只读取冻结 JSON、review manifest、模型配置和已有报告；未重训、未修改阈值/NMS/校准参数、未重新运行 108 页 test，也未使用 test 做选择。

## 核心结果

| Model | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Original frozen | {old['metrics']['tp']} / {old['metrics']['fp']} / {old['metrics']['fn']} | {pct(old['metrics']['precision'])} | {pct(old['metrics']['recall'])} | {pct(old['metrics']['f1'])} |
| Enhanced v2 | {v2['metrics']['tp']} / {v2['metrics']['fp']} / {v2['metrics']['fn']} | {pct(v2['metrics']['precision'])} | {pct(v2['metrics']['recall'])} | {pct(v2['metrics']['f1'])} |

协议：Manga109、冻结 108-page test、IoU≥0.5、按 score 的自动一对一匹配。V2 的 Recall 与 F1 较高，但 FP 也从 {old['metrics']['fp']} 增至 {v2['metrics']['fp']}，不能描述为无代价或全面提升。

原 3-stage 模型的 Stage 0 evaluated/passed/rejected 为 {old['stage_totals'][0]['evaluated']:,}/{old['stage_totals'][0]['passed']:,}/{old['stage_totals'][0]['rejected']:,}，即早拒绝 {pct(old['stage0_early_rejection_rate'])}。Stage timing 仅为 scoring，不包含 feature extraction。

## HNM 边界

- Round 1：mined/accepted/rejected={hnm['round1']['mined']}/{hnm['round1']['accepted']}/{hnm['round1']['rejected']}，扫描 {hnm['round1']['scanned_pages']} 页。
- Round 2：mined/accepted/rejected={hnm['round2']['mined']}/{hnm['round2']['accepted']}/{hnm['round2']['rejected']}，扫描 {hnm['round2']['scanned_pages']} 页。
- 前两轮 accepted={hnm['first_two_rounds_accepted']}；V2 新 train accepted={hnm['v2_train']['accepted']}；累计 train accepted={hnm['total_train_accepted']}。
- V2 validation accepted={hnm['v2_val']['accepted']}，只用于选择/校准；test hard negatives=0。
- 验证负结果也被保留：512 baseline/HN FPR={pct(.4111)}/{pct(.4750)}；1024 baseline/HN weight-3={pct(.3333)}/{pct(.3694)}。

## 图例与案例

六个 accepted HN 类型：dialogue text、high-contrast hair、clothing texture、dense manga lines、eye-like local structure、building/object outline。源页、坐标、score、review decision 和解释均在 `results/defense-assets/03_hard_negatives/metadata.json`。

检测案例来自冻结 V2 逐页 artifact：Success=`UltraEleven:036`、`DollGun:025`；FP=`YoumaKourin:099`、`Joouari:001`；FN=`MoeruOnisan_vol01:068`、`TetsuSan:028`。选择规则及逐页 TP/FP/FN 在 `04_detection_examples/metadata.json`。

## 追溯入口

- 汇总机器可读数据：`results/defense-assets/metadata/DEFENSE_METRICS.json`
- 每张图的 PPT 用途、来源与注意事项：`results/defense-assets/PPT_ASSET_INDEX.md`
- 原检测结果：`{old['source_result']}`、`{v2['source_result']}`
- 正式旧消融：`docs/b_results/B_FINAL_SUMMARY.json`
- HNM 报告数字：`docs/B_RETRAIN_COMPARISON.md`（原始 JSON 不方便统一抽取的条目标注为 `source_type=document`）

## 不应混用

本素材包没有生成 B+C 主图。`integration/b-c` 的 old B + C final ensemble 与本分支 local-face-box 协议不同；B-side NME normalization 也不等于 C 正式 interocular NME，不能放在同一比较图中。
"""
    (ROOT / "docs/B_DEFENSE_ASSETS.md").write_text(doc, encoding="utf-8")


def main():
    setup_style()
    for directory in ("01_model_overview", "02_cascade", "03_hard_negatives",
                      "04_detection_examples", "05_final_metrics", "06_ablation", "metadata"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    hn_examples, detection_examples = ensure_case_assets()
    old = summarize_frozen_result(OLD_RESULT, OLD_MODEL)
    v2 = summarize_frozen_result(V2_RESULT, V2_MODEL)
    hnm = read_hnm_numbers()
    summary = read_json(ROOT / "docs/b_results/B_FINAL_SUMMARY.json")

    plot_metric_comparison(old, v2)
    plot_counts(old, v2)
    plot_funnel(old)
    plot_stage_workload(old)
    plot_hnm_flow(hnm)
    plot_hnm_validation()
    plot_hard_negative_grids(hn_examples)
    plot_detection_grid(detection_examples)
    plot_scan_ablation(summary)
    plot_evolution()
    plot_key_card(old, v2, hnm)
    metadata = write_metadata(old, v2, hnm, hn_examples, detection_examples, summary)
    write_docs(metadata)
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "png_count": len(list(OUT.rglob("*.png"))),
        "branch": metadata["repo"]["branch"],
        "commit": metadata["repo"]["commit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
