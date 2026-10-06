# B Detector Final Optimization

## Final detector

- 11 feature channels
- 24x24 sliding window
- 1024 point-pair pixel-difference candidates
- 5-stage AdaBoost cascade
- weak trees per stage: [19, 20, 19, 13, 12]

## Frozen inference configuration

- pyramid scale factor: 1.2
- sliding step: 2
- hard NMS threshold: 0.3
- score threshold: 22.993154433101203
- bbox calibration:
  - dx = 0.03
  - dy = 0.03
  - scale_x = 0.95
  - scale_y = 0.85
- evaluation IoU threshold: 0.5

## Validation result

60-page validation:

- TP = 73
- FP = 635
- FN = 506
- Precision = 10.31%
- Recall = 12.61%
- F1 = 11.34%

Before bbox calibration, using the same scan configuration:

- TP = 69
- FP = 639
- FN = 510
- Precision = 9.75%
- Recall = 11.92%
- F1 = 10.72%

The selected calibrated operating point simultaneously reduced FP and FN.

## Region pixel-difference ablation

A balanced 3x3 region-mean pixel-difference extension was tested over all
11 channels.

From-scratch 3-stage validation:

- point-pair baseline: 249 positives / 120 negatives
- region-enhanced: 249 positives / 132 negatives

Late-stage validation:

- point-only Stage 3: 249 positives / 108 negatives
- mixed Stage 3: 249 positives / 109 negatives

Region features were actively selected by AdaBoost, but did not improve
validation rejection performance. The final detector therefore retains
the original point-pair pixel-difference representation.

The test split was not used for model or parameter selection.
