# ER missed-positive probability diagnostic

Read-only evaluation of the prior binary pilot's eight validation crops. Three predetermined illustrative thresholds (0.5, 0.3, 0.1); no checkpoint change, production threshold selection or retraining.

| Threshold | Mean Dice | Mean precision | Mean recall | Mean FP pixels/crop |
|---|---:|---:|---:|---:|
| 0.5 | 0.1558 | 0.2440 | 0.1164 | 5268 |
| 0.3 | 0.1760 | 0.2377 | 0.1421 | 6849 |
| 0.1 | 0.2082 | 0.2289 | 0.1951 | 10173 |

Mean per-crop ROC AUC is 0.6249. Lowering to0.1 increases recall by7.87 percentage points but still misses approximately80.5% of pseudo-positive pixels on average. False positives nearly double and precision worsens. Visual probability maps show many target networks remain low probability, while some off-target coarse structures remain high probability. Thus this is not adequately explained by the0.5 cutoff alone. No production change recommended.

Caveats: comparison to algorithm-derived contours using the newly confirmed background semantics; single repeatedly viewed validation specimen; correlated crops. These are exploratory agreement metrics, not independently verified biological accuracy.

Full image: ../outputs/er_binary_probability_20261030/probability_threshold_review.png
Data: ../outputs/er_binary_probability_20261030/diagnostic.json
Next: matched fresh-initialization ER versus Fluo input control, fixed same labels and specimen split.
