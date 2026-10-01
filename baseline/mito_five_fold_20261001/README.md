# Frozen Mito five-fold baseline, 2026-10-01

This is the **unchanged** independent-2D U-Net baseline across 36 manually
labelled development images. Fold 0 is the previously frozen run, reused
without retraining. Folds 1–4 used the same `code/train.py` source (SHA256
`9aeb8bf0f5332dd692b3522250024f0edf3f7c3626cceb16348fefbdfae15e1c`)
and all original scientific defaults: 512-pixel patches, 0.6 positive-centred
sampling, batch 4, 100 steps/epoch, BCE+soft-Dice loss, original augmentation,
Adam 1e-4, at most 100 epochs, and EarlyStopping. Every run used one Secure
Cloud NVIDIA A40 with Python 3.11.16, TensorFlow 2.21.0, NumPy 2.4.6 and
tifffile 2026.3.3. Each new fold passed CUDA Driver API and TensorFlow GPU
matrix preflight before training.

The split assignment was frozen across all five runs. For each outer fold,
the next fold was validation and the remaining three were training. Only
validation labels selected checkpoint/epoch and the baseline threshold;
outer-test labels were used once for final metrics. `code/mito_cv_report.py`
verified identical fold maps, disjoint train/validation/test sets, and exactly
one outer-test result per image. The original fold assignment stratifies using
foreground fraction from all 36 masks; this label-assisted split design is
disclosed here. Original organoid/Z identities are unavailable, so related
images across folds cannot be ruled out. No future unseen organoid images were
used; none are in this project yet.

## Out-of-fold performance

Metrics are **per-image** on full 1200×1200 predictions. The ± value is the
sample standard deviation across 36 images, not a confidence interval.

| Validation-selected threshold | Mean ± SD | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Dice | 0.8543 ± 0.0628 | 0.8756 | 0.7266 | 0.9296 |
| IoU | 0.7507 ± 0.0940 | 0.7787 | 0.5705 | 0.8685 |
| Precision | 0.8391 ± 0.0996 | 0.8852 | 0.6406 | 0.9559 |
| Recall | 0.8773 ± 0.0515 | 0.8691 | 0.7360 | 0.9714 |

Dice lower quartile is 0.7979. Nine of 36 images scored below 0.80, while
13 scored at least 0.90. Fold mean Dice has sample SD 0.0182, with fold 3
the lowest at 0.8225. Full per-image numbers are in `oof_per_image.csv`.

| Outer fold | Images | Mean Dice | Mean IoU | Precision | Recall | Best validation epoch | Selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7 | 0.8693 | 0.7768 | 0.8447 | 0.9057 | 78 | 0.45 |
| 1 | 7 | 0.8604 | 0.7583 | 0.8327 | 0.8934 | 56 | 0.45 |
| 2 | 7 | 0.8579 | 0.7573 | 0.8398 | 0.8828 | 27 | 0.40 |
| 3 | 7 | 0.8225 | 0.7009 | 0.8239 | 0.8297 | 28 | 0.55 |
| 4 | 8 | 0.8607 | 0.7589 | 0.8522 | 0.8750 | 15 | 0.70 |

The lowest Dice images were `mask_010.tif` (0.7266, fold 0), `mask_009.tif`
(0.7484, fold 4), `mask_004.tif` (0.7632, fold 3), `mask_001.tif`
(0.7669, fold 0), and `mask_005.tif` (0.7724, fold 2). These are diagnostic
descriptions, **not** a tuning list. No fold was retrained because of a score.

Annotated foreground fraction correlated positively with OOF Dice
(Pearson r=0.555). Dice by ascending foreground-fraction quartile averaged
0.7938, 0.8667, 0.8525 and 0.9042 (nine images each). For images with at
most 5% foreground, mean Dice was 0.7914 and precision 0.7272; above 5%,
Dice was 0.8695 and precision 0.8661. Recall was about 0.877 in both groups.
This suggests false positives matter more on sparse images, but the association
is observational and sample size is small; it does not establish cause or
justify tuning on these outer-test labels.

## Pre-registered threshold-policy comparison

The same five checkpoints were scored twice, once at each fold's
validation-selected threshold and once at fixed 0.50. No model was retrained
for this comparison and no test labels selected either threshold. Fixed 0.50
gave mean Dice 0.8539 ± 0.0636, median 0.8727, min 0.7284, and max 0.9298.
The paired mean Dice change (fixed minus selected) was −0.0004 with SD 0.0051.
The difference is negligible relative to image-to-image variation; this
internal result does not establish a superior threshold policy.

## Generalization and next experiments

There is no obvious large training-versus-validation gap at the best epochs,
but patch-training soft Dice and full-image validation Dice are different
measurements and cannot be equated directly. The range of selected thresholds
(0.40–0.70), fold-3 drop and low-foreground tail show instability in this
36-image development pool. These OOF scores are internal, and fold-0 outcomes
had already been inspected before this five-fold run. Any next change evaluated
on these same folds is exploratory; independently labelled external data will
be needed for a fresh final generalization estimate.

1. **Data provenance and annotation review first.** If non-identical images
   come from related organoids, image-level folds can overstate independence.
   Audit source provenance if available and review ambiguous masks without
   editing them. A group-aware split would be a new evaluation protocol and
   must be declared before retraining.
2. **One-factor sampling experiment.** The baseline already uses 0.6
   positive-centred patches. Pre-register one moderate alternative (for
   example 0.35), select it using only inner validation, and compare a complete
   paired CV run. This tests whether the sampling balance helps sparse images
   without changing architecture, loss, augmentation or threshold policy.
3. **One-factor loss experiment only if sampling is inconclusive.** Compare
   current BCE+Dice with one pre-specified, modestly precision-aware variant
   using inner validation. Keep all other settings fixed. Do not tune it from
   the low-scoring outer-test images or claim a new unbiased test result from
   these already-inspected folds.

No optimization run was started during this baseline evaluation.

## Artifact custody

This Git directory contains only small split/config/history/metric files.
The five verified `.keras` checkpoints, both sets of same-size TIFF
predictions, GPU logs, and SHA256-checked result archives are stored locally
under `runpod_results/mito_cv_baseline_20261001/`, which Git ignores. All
five archives passed SHA256 verification; all five Keras ZIP files passed
integrity checks and had non-zero size. The temporary Pod was terminated
after the downloads, and `list_pods` returned an empty list. No Cetus job was
modified.
