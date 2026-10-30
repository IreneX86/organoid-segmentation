# Fluo-only ER partial-label pilot — locked before validation preparation

Preparation completed on 2026-10-29: **1,520 training patches + 80 validation
patches**, no skipped streams, no test pixels opened. Every saved patch passed
SHA256, shape, label-code and class-count verification. Source crop equality
was checked during generation. Local U-Net forward / masked loss / checkpoint
round-trip checks passed without optimizer updates. Training has not started.
Operational state: `outputs/er_partial_student_dataset_20261029/readiness.json`.

Subsequent authorized GPU run completed:6 epochs, best1, validation partial-label
Dice0.030353. Candidate rejected; results safely local and Pod terminated.
See [pilot report](../reports/ER_partial_student_pilot_20261029.md).

## Question

Can old ER-positive regions teach a Fluo-only student without treating missed
intracellular ER as background? This pilot does not promise detection of all ER.
Keep original TIFFs, old masks and failed student checkpoints unchanged.

## Fixed dataset and model

- Six training stacks / validation `cell_20260904_23963_NZ_40`.
- Test `cell_20260814_23300_NZ_100` is not opened or packaged.
- Every training/validation plane supplies one positive-rich and one
  exterior-negative-rich 384×384 grid crop where that class exists.
- Negatives follow the existing fixed dual-channel p20 / sigma2 / edge-connected /
  erosion4 / positive-distance>12 rule. No validation-driven rule revision.
- Label 1 = old ER positive; 0 = weak exterior negative; 255 = ignore.
- Fluo is normalized using its whole plane's p1/p99.8 limits before cropping.
  Only raw Fluo crops, limits and target arrays are packaged, no ER intensities.
- U-Net base filters16; Adam LR0.001; batch4 (two crops from each stream);
  48 steps/epoch; 20 maximum epochs; EarlyStopping patience5 on validation loss.
- Sample training organoids uniformly, then slices. Paired flips and rot90 only.
  Whole-organoid grouping prevents overlapping XY or adjacent Z leaking across splits.
- Masked BCE + masked Dice: ignored pixels contribute to neither term.
- Validation visits every prepared validation crop, no augmentation. Checkpoint
  selected only by validation loss. Threshold fixed at 0.5; no threshold search.

## Interpretation and safeguards

Exterior-only negatives are easier than intracellular non-ER. The model could
learn to label broad cell structures. Save predicted-positive fraction within
ignored validation tissue plus fixed first/middle/last crop probability previews.
Do not claim success just because known-pixel Dice is high. Metrics are on sampled,
possibly overlapping validation patches, not independent full-image accuracy.
One validation organoid and imperfect pseudo-labels limit generalization claims.

No observed test image selects a checkpoint or rule. Only after reviewing this
pilot should a separate final evaluation be considered. Do not automatically
upgrade a production pipeline or batch-predict external images.

## Commands

```sh
python code/er_partial_student.py prepare \
  --data outputs/er_coherent_fiber_full_20261028 \
  --output outputs/er_partial_student_dataset_20261029
python code/er_partial_student.py verify \
  --data outputs/er_partial_student_dataset_20261029
# GPU only; not executed during preparation:
python code/er_partial_student.py train \
  --data outputs/er_partial_student_dataset_20261029 \
  --output outputs/er_partial_student_run_NEW
```

The prepare/train commands refuse existing output directories. Training requires
one visible GPU and a GPU matmul preflight. No local CPU training. GPU deployment,
cost watchdog, data transfer and cleanup are a separate operational step; running
this file does not provision a Pod. Use the previously verified Python3.11 /
TensorFlow2.21 environment when deploying; local forward checks do not certify
remote CUDA compatibility. No paid resource was created in this preparation.
