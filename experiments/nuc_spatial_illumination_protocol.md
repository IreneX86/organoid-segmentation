# Nuc spatial-illumination candidate: frozen one-factor protocol

**Status:** frozen before candidate training. This is an exploratory,
same-batch internal comparison, not an independent test of generalization.

## Question and evidence

The original Nuc U-Net sometimes marks fluorescent fragments and misses
structures the reviewer circled in `cell_22375_NZ_100/fluo_001.tif`. The red
circles indicate approximate positive locations only: they do not specify
complete pixel masks or negatives. The `fluo` inputs differ visually from the
Nuc training inputs. A smooth illumination gradient is a plausible contributor
to this shift, but that explanation is not yet established.

The frozen internal split also contains related images: 29 of its 34 validation
inputs have a training input with correlation at least 0.99. Thus even a strong
gain on this split will not estimate performance on a new acquisition.

## Fixed data and candidate

Use exactly the 167 original `Nuc_input`/`Nuc_output` pairs and the frozen
`results/nuc_internal_split_v1.json`, including all four true negatives.
Validate every TIFF SHA256 before fitting. Do not edit source images, masks,
the split or the completed baseline run.

The **only** training-method change is an optional image-only, zero-centred
linear illumination plane **after** the existing crop and augmentation:

- probability per training patch: **0.50**;
- independent horizontal and vertical slopes, each uniform in
  **[-0.06, +0.06]** normalized intensity units;
- resulting absolute offset at any point is at most **0.12**;
- clip intensity to [0, 1]; leave the mask and pixel coordinates unchanged;
- use a separately seeded random stream, so baseline patch choices and prior
  augmentation random draws remain paired.

The transform is training-only. Continue the original per-image 1–99.8
percentile normalization and unmodified full-image inference. Keep the U-Net,
BCE+soft-Dice loss, sampling probabilities, patch size, overlap, batch size,
optimizer, learning-rate schedule, 100-epoch cap, seed 42, checkpoint rule,
threshold grid and early stopping unchanged. Initialize a **new** Nuc model;
do not fine-tune baseline or other organelle weights.

## Evaluation and decision

Use the same 33 nonempty validation images for checkpoint selection at fixed
threshold 0.50, then apply the existing threshold selection grid to the
candidate's selected checkpoint. Neither the one empty validation image nor
the target `fluo` images select checkpoint or threshold. Compare the frozen
baseline and candidate **paired by validation filename at fixed 0.50**:
mean/median/SD Dice, paired Dice change and positive-change count,
10th-percentile and worst-case Dice, IoU, precision and recall. Report the
candidate's validation-selected threshold separately, plus false-positive
pixel counts on `nuc_005.tif` at both thresholds. The latter is a diagnostic,
not a selection criterion.

The preregistered **preview gate** is deliberately modest: candidate mean
fixed-0.50 Dice must be no more than 0.005 below baseline; median paired
Dice change must be at least -0.005; mean precision and recall must each be
no more than 0.010 below baseline. If the gate passes, generate masks for the
first 10 target `fluo` images in a **new** review-only directory. Do not
overwrite `Nuc_label`. If it fails, retain the experiment as rejected and do
not promote the candidate. This gate merely screens out obvious internal
degradation; passing it does not show better target-domain segmentation.

The reviewed circles may inform a qualitative post-run failure audit, but
they are incomplete annotations. Do not report Dice, precision, false-positive
rate, or statistical target-domain improvement from them, and do not revise
the model, threshold or this protocol after seeing them. No final production
model is selected without independent representative target labels or
reliable acquisition-group provenance.

Store training artifacts, predictions and logs only in ignored local storage;
commit only code, small metrics and the scientific record. Use at most one
Runpod Secure Cloud A40 Pod, verify its GPU before training, download and
SHA256-check the result archive before deletion, then confirm no Pod remains.
