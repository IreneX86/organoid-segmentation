# Mito experiment log

## 2026-10-01 — locked baseline and read-only data audit

The immutable fold-0 baseline is Git tag `baseline-mito-fold0-20261001`
(`df516e1`). Its 7 internal held-out images reached mean Dice 0.869262,
sample SD 0.085012, minimum Dice 0.726555, mean IoU 0.776824,
precision 0.844737, and recall 0.905731. These are **not** five-fold CV
metrics and must not be described as external generalization.

Read-only audit of the 36 paired Mito TIFFs in the existing local transfer
archive found 36 images and 36 matching masks. All images are 1200×1200
uint16; all masks can be interpreted as binary and contain foreground.
Foreground fractions range from 0.006063 to 0.134049 (median 0.066151).
There are no exact duplicate input pixel arrays. A coarse 60×60-pixel
thumbnail correlation screen found no pair above 0.98; this does **not**
prove that related images from one organoid are absent. Original organoid/Z
identities are unavailable and will not be reconstructed.

The baseline already uses positive-centred patch sampling with probability
0.6, flips/90° rotations, mild intensity augmentation, BCE+soft-Dice loss,
ReduceLROnPlateau, full-image validation, and validation-only threshold
selection. Adding these under new names is not an experiment. The current
five-fold assignment is stratified using each labelled mask's foreground
fraction, then frozen for fair comparison. This label-assisted split design
must be disclosed when reporting generalization.

**Missing baseline evidence:** only fold 0 has been trained. `code/mito_cv_report.py`
now refuses to report a five-fold score until all five outer-test reports
exist, their split maps match, train/validation/test sets are disjoint, and
each of the 36 images appears exactly once as outer test. No new model has
been trained or selected by this audit.

## Pre-registered first comparison — threshold policy

**Hypothesis:** A fixed semantic threshold of 0.50 may be at least as stable
as selecting the maximum validation Dice among 17 thresholds on only seven
validation images. This changes one inference decision, not architecture,
loss, augmentation, masks, or training. The comparison will use predictions
from the *same* five baseline-trained models; for each outer fold the 0.50
threshold is fixed in advance and the baseline threshold is selected using
that fold's validation images only. Neither policy may inspect its outer-test
labels for that fold before predictions and thresholds are locked. Both
policies must be reported across all 36 out-of-fold images, including mean,
SD, minimum, lower quartile, IoU, precision, and recall. A tiny mean gain with
worse spread or lower tail is not a reliable improvement. No threshold policy
will be chosen from a single fold's held-out results.

The fold-0 held-out metrics were previously inspected. Any method claim from
this internal CV remains exploratory; later independently labelled unseen
data would be needed for external confirmation.

## Next execution gate

Completing the unchanged baseline requires four additional full GPU folds.
The previous A40 fold took about 26 minutes of training/evaluation and about
US$0.26 in total Pod charges; four folds plus setup are roughly US$1.1–1.5,
with uncertainty from queue, transfer, and runtime variation. This is a
substantial GPU experiment, so it has **not** been started. Do not run a
candidate model, tune from outer-fold scores, or deploy a new Pod before the
baseline CV and its resource plan are reviewed.

## 2026-10-01 — five-fold baseline authorized; GPU stock blocker

The user authorized fold 1–4 unchanged baseline training with an additional
Runpod budget cap of US$5.00. No fold has yet started. A live Runpod read
showed Secure Cloud A40 at US$0.49/hour, but two creation requests returned
`There are no longer any instances available with the requested specifications`.
The second request was constrained to CUDA 12.8 after a capacity read; it also
failed. A subsequent capacity read showed CUDA 12.8 `Out` and CUDA 13.2 `Low`.
`list_pods` returned an empty list, so no GPU charges were started by this
stage. Do not create a different GPU class without a revised, validated
environment and budget check.

`code/mito_export_predictions.py` was prepared for post-training inference at
the validation-selected threshold and the pre-registered fixed 0.50 threshold
using the *same checkpoint*. The five-fold reporter now summarizes both
policies, all 36 per-image OOF rows, fold means, distribution statistics, and
foreground-fraction correlation. It still refuses incomplete or overlapping
folds. The prior fold-0 checkpoint must be exported in the same way before a
threshold-policy comparison is reported.
