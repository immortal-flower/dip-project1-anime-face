# B Module Final Test Result

## 1. Purpose

This document records the final evaluation of the frozen traditional-feature
anime face detector on the untouched Manga109 test split.

The test split was not used for model selection, threshold tuning, bounding-box
calibration, hard-negative mining decisions, or scan-parameter optimization.

## 2. Frozen detector configuration

- Window size: 24 x 24
- Channels: 11 handcrafted channels
- Candidate features: 1024 point-pair difference features
- Cascade stages: 5
- Weak trees per stage: [19, 20, 19, 13, 12]
- Sliding-window step: 2
- Image-pyramid scale factor: 1.2
- Score threshold: 22.993154433101203
- Pre-NMS limit: 500
- NMS method: hard NMS
- NMS IoU threshold: 0.3
- Evaluation IoU threshold: 0.5

Bounding-box calibration:

- dx = +0.03
- dy = +0.03
- scale_x = 0.95
- scale_y = 0.85

## 3. Final test protocol

- Split: test
- Number of pages: 108
- Execution: page-level multiprocessing
- Parallel workers: 6
- Logical CPUs: 12
- Windows per page: 1,491,184
- Total scanned windows: 161,047,872
- Wall-clock runtime: approximately 4.80 hours

The parallel implementation changes only page scheduling. Each page uses the
same frozen detector and inference parameters as the sequential evaluator.

## 4. Final test metrics

| Metric | Result |
| --- | ---: |
| TP | 150 |
| FP | 977 |
| FN | 1279 |
| Ground-truth faces | 1429 |
| Final detections | 1127 |
| Precision | 13.31% |
| Recall | 10.50% |
| F1 | 11.74% |

## 5. Validation vs. final test

| Split | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (60 pages) | 73 | 635 | 506 | 10.31% | 12.61% | 11.34% |
| Final test (108 pages) | 150 | 977 | 1279 | 13.31% | 10.50% | 11.74% |

The final-test F1 is close to the validation F1 (+0.40 percentage points).
Precision increases on the test set, while recall decreases.

No detector parameter was changed after observing the final-test result.

## 6. Typical qualitative cases

Automatically selected report examples:

### Success case 1

- Page: `manga109:YoumaKourin:024`
- TP = 8
- FP = 11
- FN = 10
- F1 = 43.24%

### Success case 2

- Page: `manga109:HinagikuKenzan:032`
- TP = 5
- FP = 11
- FN = 4
- F1 = 40.00%

### FP-heavy failure

- Page: `manga109:YoumaKourin:102`
- TP = 1
- FP = 42
- FN = 9
- F1 = 3.77%

### FN-heavy failure

- Page: `manga109:MoeruOnisan_vol01:068`
- TP = 2
- FP = 12
- FN = 38
- F1 = 7.41%

Generated figures are stored under:

`results-final-test/report-assets/typical-figures/`

Important files:

- `final_test_typical_cases_montage.png`
- individual success / FP-heavy / FN-heavy figures
- `typical_cases_summary.json`
- `INDEX.md`

## 7. Raw evidence

Final evaluation outputs:

- `results-final-test/test-final-parallel.json`
- `results-final-test/test-final-parallel.log`

The JSON result contains the frozen configuration hashes, page list,
per-page predictions, per-page metrics, scan logs, and aggregate test metrics.

## 8. Final conclusion

The final frozen B-module detector achieves:

**Precision 13.31%, Recall 10.50%, F1 11.74%**

on the complete 108-page untouched Manga109 test split.

The detector retains the required traditional handcrafted 11-channel feature
pipeline, AdaBoost cascade, image pyramid, sliding-window scanning, and hard NMS.
