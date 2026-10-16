# Nuc 2D internal prototype v1

**Status (2026-10-15 UTC): training complete and local artifact verification passed.** This is a **same-batch internal prototype diagnostic**. It measures performance on the frozen development validation split, not on an independent specimen, acquisition, or test set.

## Frozen run

The [predeclared protocol](../experiments/nuc_internal_prototype_protocol.md) and [split manifest](../experiments/results/nuc_internal_split_v1.json) cover 167 paired, single-page 1200×1200 Nuc TIFFs: 133 training images and 34 validation images. Four masks are confirmed true negatives. Three remain in training; the fourth, `nuc_005.tif`, is in validation. The run used a Nuc-specific 2D binary U-Net trained from random initialization with the frozen crop, augmentation, normalization, loss, optimizer, and tiled-inference settings. Checkpoint selection used mean per-image Dice on the 33 nonempty validation masks at threshold 0.50. The selected checkpoint was epoch **66**; training completed **86** epochs. Threshold selection on that same checkpoint and validation set chose **0.45**.

## Internal validation results

The table reports arithmetic means across the **33 nonempty validation images**. The reported standard deviations are those saved by the run; all metrics are full-image segmentation metrics.

| Threshold | Mean Dice | Dice SD | Mean IoU | Mean precision | Mean recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 0.50 | 0.9564518556724969 | 0.017187997810842053 | 0.9170265862656685 | 0.9625259427264676 | 0.9506154742414055 |
| Selected 0.45 | 0.9565138074004935 | 0.017192006726415063 | 0.9171400252837375 | 0.9590452727412606 | 0.9541328654806008 |

On the **one** empty-mask validation image, `nuc_005.tif`, the model produced **2,744 false-positive pixels** at 0.50 and **3,093 false-positive pixels** at 0.45. It had at least one false-positive pixel at both thresholds. This image was excluded from the Dice means; a single negative image does not establish a false-positive rate. The three empty-mask training images are not validation observations.

In a subsequent expert review, the project lead confirmed that the suspected nucleus in `nuc_005.tif` is too indistinct and its membrane structure insufficiently clear, so it remains a true negative. The structure in `nuc_006.tif` is slightly clearer; the reviewer also consulted `nuc_007.tif` and judged `nuc_006.tif` positive. The model receives each 2D image independently, so borderline decisions that use other images may be unresolved from its input alone. The false positives on `nuc_005.tif` do not establish a label error. This case does not establish acquisition provenance or Z order for the dataset; see the [annotation record](../experiments/nuc_data_audit_20261015.md).

The selected threshold improved the mean Dice on these same 33 images by about **0.000062** while adding **349** false-positive pixels on the validation negative. The threshold and checkpoint were both chosen using this validation set, so its selected score has selection bias. The [data audit](../experiments/nuc_data_audit_20261015.md) found high input similarity, including a similarity component linking 148 of 167 images, and no reliable acquisition, specimen, or batch mapping. These numbers therefore provide **no independent generalization estimate**. Performance on a new source remains unknown.

## Completion and retained artifacts

The run recorded `NUC_INTERNAL_TRAINING_COMPLETE` and exit status **0**. It ran on one Secure Cloud NVIDIA A40 with Python 3.11.17 and TensorFlow 2.21.0. Training began at 12:06:32 UTC and the runner ended at 13:00:18 UTC (**53m46s**). Across 215 GPU samples, mean utilization was **35.0%**, median **33%**, and peak **100%**. Peak reported memory use was **43,975 MiB of 46,068 MiB**; TensorFlow allocator reservation can dominate that figure, so it is not a measure of live model memory demand.

The downloaded archive passed the local read-only verifier, including archive and internal hashes, checkpoint integrity, frozen split and counts, selection consistency, metric consistency, and prediction TIFF checks. A separate local check successfully reloaded the Keras checkpoint. The archive SHA256 is `ae78d5d8bde8ede35520217263d735a373f129ec633db3df031a0beb1359c8f2`; the full Keras checkpoint SHA256 is `c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e`.

The full checkpoint, archive, per-image metrics, predictions, probability maps, metadata, and logs remain in ignored local storage under `runpod_results/nuc_internal_20261015/`. Source TIFFs, labels, model files, predictions, logs, and credentials are excluded from Git. The A40 Pod was deleted after verification, and a subsequent Pod listing was empty. An early billing API read showed **US$0.254** for the live Pod but lagged its lifetime. At the catalog price of **US$0.59/hour** over about **1h24m**, its GPU time is estimated at **US$0.83**, plus small storage charges; an earlier aborted Pod added **US$0.018**. Final itemized billing was pending, so these are not final charges.

The next performance evaluation requires independently sourced labelled Nuc images or trustworthy acquisition provenance, with a new source-grouped protocol frozen before scoring. The current validation split must not be presented as an independent test.
