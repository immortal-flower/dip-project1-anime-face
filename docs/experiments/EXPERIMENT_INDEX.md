# B-module Experiment Index

## 1. Baseline candidate-pool scaling

The original detector used 1024 point-pair pixel-difference candidates.

Anchored candidate-pool experiments preserved the original first 1024
features exactly.

- 1024 candidates:
  - validation positive: 249 / 252
  - validation negative: 120 / 360
  - recall: 98.81%
  - FPR: 33.33%

- 2048 anchored candidates:
  - validation positive: 249 / 252
  - validation negative: 152 / 360
  - FPR: 42.22%

- 4096 anchored candidates:
  - validation positive: 249 / 252
  - validation negative: 161 / 360
  - FPR: 44.72%

Conclusion: simply enlarging the random point-pair pool did not improve
validation performance.

## 2. Late cascade stages

Starting from the 3-stage 1024-feature detector:

- Stage 3:
  - positive: 249 / 252
  - negative: 108 / 360
  - FPR: 30.00%

- Stage 4:
  - positive: 249 / 252
  - negative: 105 / 360
  - FPR: 29.17%

Final cascade tree counts:

[19, 20, 19, 13, 12]

## 3. Sliding-window / pyramid ablation

Fixed 6-page validation subset.

Best tested configuration:

- step = 2
- scale factor = 1.2
- Precision = 3.73%
- Recall = 33.90%
- F1 = 6.72%

This configuration provided the best F1 / runtime trade-off among the
tested step {1,2} and scale factor {1.1,1.2,1.3} combinations.

## 4. Hard-negative mining

Round 1:

- mined candidates: 50
- manually accepted hard negatives: 36
- rejected: 14

A new Stage 5 trained with target recall 1.0 could not reject validation
negatives:

- validation input: 249 positive / 105 negative
- validation output: 249 positive / 105 negative

Conclusion: this HNM round did not improve the detector under the strict
recall constraint.

## 5. Positive crop alignment

Training positives analyzed: 1880.

- 97.61% used the full 10% positive margin.
- only 2.39% required reduced margin.
- median face width inside the final 24x24 patch: about 19.83 px.
- median face height: about 19.84 px.
- more than 99% of samples were essentially centered.

Conclusion: positive crop alignment is highly consistent and is unlikely
to be the main detector bottleneck.

## 6. Feature usage audit

For the 5-stage baseline:

- candidate pool: 1024
- internal-node feature uses: 249
- unique features used: 170
- pool utilization: 16.60%

Pixel-pair distance:

- <2 px: 3.61%
- 2-4 px: 12.45%
- 4-8 px: 32.53%
- 8-12 px: 24.10%
- >=12 px: 27.31%

Median distance: 8.06 px.

All 11 channels were used.

## 7. Region-mean pixel-difference experiment

Added 264 balanced 3x3 region-mean differences:

- 24 region features per channel
- all 11 channels covered
- total feature pool: 1288

From-scratch mixed 3-stage detector:

- baseline: 249 positive / 120 negative
- region-enhanced: 249 positive / 132 negative

Late-stage mixed experiment:

- point-only Stage 3: 249 positive / 108 negative
- mixed Stage 3: 249 positive / 109 negative

Region features were actively selected by AdaBoost, but did not improve
validation rejection performance.

Final detector therefore retains point-pair pixel-difference features.

## 8. NMS experiment

Hard NMS was retained.

On the fixed 6-page validation subset:

- hard NMS:
  - TP = 8
  - FP = 67
  - FN = 51
  - F1 = 11.94%

- weighted NMS, support 2:
  - TP = 7
  - FP = 64
  - FN = 52
  - F1 = 10.77%

- weighted NMS, support 1:
  - TP = 7
  - FP = 68
  - FN = 52
  - F1 = 10.45%

## 9. Localization / bbox calibration

Uncalibrated operating point:

- TP = 69
- FP = 639
- FN = 510
- Precision = 9.75%
- Recall = 11.92%
- F1 = 10.72%

Selected bbox calibration:

- dx = 0.03
- dy = 0.03
- scale_x = 0.95
- scale_y = 0.85

Calibrated operating point:

- TP = 73
- FP = 635
- FN = 506
- Precision = 10.31%
- Recall = 12.61%
- F1 = 11.34%

This operating point simultaneously reduced both false positives and
false negatives.

## 10. Final frozen detector

- 11 channels
- 1024 point-pair pixel-difference candidates
- 5-stage cascade
- trees: [19, 20, 19, 13, 12]
- step = 2
- scale factor = 1.2
- hard NMS threshold = 0.3
- score threshold = 22.993154433101203
- bbox calibration = (dx=0.03, dy=0.03, sx=0.95, sy=0.85)

Validation:

- TP = 73
- FP = 635
- FN = 506
- Precision = 10.31%
- Recall = 12.61%
- F1 = 11.34%

Final test is run only after all parameters are frozen.
