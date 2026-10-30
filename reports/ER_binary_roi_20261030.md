# ER binary ROI pilot — 2026-10-30

## Decision
Reject for production. Broad positive sheets decreased, but validation ER recall collapsed. Preserve every prior model and label. No further training started.

## Label semantics and design
User confirmed yellow is positive ER and gray/blue is background. Inside the original 40 selected 320×320 crops only, label 1 remains positive; 0/255 become background. Original data and masks are unchanged. The artificial 64-pixel supervision windows were not used to redefine background. 32 training crops from four specimens and eight validation crops from a separate specimen retain the existing split.

Fixed 500 updates, batch four, Adam 0.0003, seed 20261029, existing calibrated Fluo-only U-Net, frozen BatchNorm, dropout off, threshold 0.5. No validation checkpoint selection or threshold search. Training took 23.13 seconds on Secure Cloud RTX A6000; environment setup and transfer are additional billed time.

## Results on the same new binary validation labels
| Metric | Before | After |
|---|---:|---:|
| Mean Dice | 0.3229 | 0.1558 |
| Mean IoU | 0.1953 | 0.0868 |
| Mean precision | 0.2000 | 0.2440 |
| Mean recall | 0.9055 | 0.1164 |
| Positive area fraction | 0.7002 | 0.0706 |

Training Dice after the update is 0.7528. This large training–validation gap indicates poor transfer in this pilot, without proving a single causal explanation. Appearance/domain variation, fitting to a small crop set and cross-channel information limitations remain hypotheses. These scores cannot be compared directly with older partial-label Dice because the scoring region changed. Positive contours remain algorithm-derived; one repeatedly inspected validation specimen is not an independent generalization test.

## Artifacts and verification
Local results: `outputs/er_binary_roi_20261030/`. `all8_binary_before_after.png` columns are Fluo / binary target / before / after. Archive SHA256, checkpoint SHA256 and Keras ZIP integrity passed; all 48 prediction NPZ files are finite with expected 320×320 shapes and binary labels. Before/after validation inputs and targets match.
Checkpoint SHA256: `a4c2c403b6b9333c58bb26d0dd4bdf4c93a4b22264fbfa6b0b6efebf9c2c8efc`.

## Cleanup and cost
Pod `[pod-id-redacted]` terminated after verified local download. Console confirms GPUs 0, vCPUs 0, storage 0 GB, endpoints 0 and spend $0.00/hour. Displayed balance $4.59 → $4.53, approximately $0.06; rounded balance difference, not an itemized bill. Cleanup screenshot saved locally.

## Next informative experiment
Before further Fluo-only tuning, consider an ER-input positive control with the same new binary labels and specimen split. This would help distinguish label/training limitations from cross-channel transfer limitations. It has not been started. Do not deploy this candidate or automatically tune thresholds against these validation images.
