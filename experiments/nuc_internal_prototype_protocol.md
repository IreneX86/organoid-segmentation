# Nuc 2D internal prototype protocol

**Status:** protocol frozen before Nuc training; the full internal run completed
on 2026-10-15 and is documented in the
[Nuc internal prototype report](../reports/nuc_internal_prototype_v1.md). It is
an internal diagnostic, with no outer test set or independent cross-validation
estimate. The
[data audit](nuc_data_audit_20261015.md) found no mapping from
`nuc_###` to acquisition field, specimen, or batch. Input similarity links
148/167 images in one large near-duplicate group; those links are a dependence
warning, not verified biological groups. A held-out filename here cannot
support a claim of performance on a new specimen or acquisition.

## Frozen data and split

Use the 167 same-name, single-page 1200×1200 TIFF pairs in
`label_ref/Nuc/Nuc_input` and `label_ref/Nuc/Nuc_output`. Inputs are `uint16`;
semantic masks are `uint8` with values 0 and 255. `nuc_002.tif` through
`nuc_005.tif` are confirmed true negatives. Preserve all four all-black masks
and the source TIFF bytes.

The Nuc internal v1 manifest at
[`results/nuc_internal_split_v1.json`](results/nuc_internal_split_v1.json)
freezes filenames, train/validation membership, and SHA256 of **each input and
mask file** before training. Of the 163 nonempty masks, sort filenames by the
hexadecimal SHA256 digest of UTF-8 `nuc-internal-v1:` followed by the exact
filename, ascending, and assign the first 33 to validation. Add `nuc_005.tif`
to validation. The other 130 nonempty pairs and `nuc_002.tif`–`nuc_004.tif`
form training: **133 train / 34 validation**. Check every file against the
manifest at run time; do not reshuffle, remove empty masks, or change labels
based on model results.

## Training and selection

Train a Nuc-specific 2D binary U-Net from random initialization using the
existing paired augmentation and these fixed baseline defaults: native-grid
512×512 crops, 0.60 foreground-centred and 0.40 uniform crop probability,
per-image 1st–99.8th percentile normalization, BCE + soft-Dice loss, Adam at
1e-4, batch size 4, 100 steps per epoch, and at most 100 epochs. Use
full-resolution tiled inference with 512-pixel tiles and 128-pixel overlap.
Keep its checkpoint and predictions in a separate Nuc output directory.
Training crops from the three empty training images retain their
all-background targets.

Use only the **33 nonempty validation images** for checkpoint, early-stopping,
and probability-threshold selection by mean per-image Dice. Select the
checkpoint at the fixed 0.50 threshold, then select a final threshold on that
checkpoint from 0.10 to 0.90 in steps of 0.05. Break equal-score threshold
ties by proximity to 0.50, then by the lower threshold. Save the chosen
checkpoint and threshold with their selection metrics and hashes. Do not tune
either choice against `nuc_005.tif`, training scores, or a later dataset.

## Reporting and interpretation

Report per-image and aggregate Dice for the 33 **nonempty** validation images
at both 0.50 and the selected threshold. Do not assign an empty-mask Dice of
1 to `nuc_005.tif` or include it in mean Dice. Separately report its false
positive pixel count and whether it has any false positive pixel at each
threshold; this is one validation negative, not a robust false-positive-rate
estimate. The three training negatives may be inspected diagnostically but
are not validation observations. Keep all output masks and probabilities
outside the source TIFF directories.

The selected validation Dice is subject to same-batch dependence and selection
bias, so label every result **Nuc internal prototype diagnostic**. Do not
describe it as an independent test score or a generalization estimate. If
acquisition provenance or independently sourced labelled images become
available, define and freeze a new source-grouped protocol and evaluation
before using them; do not retrofit this filename split into biological groups.
