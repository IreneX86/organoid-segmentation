# Nuc 2.5D stack-to-center segmentation: feasibility gate (no training)

**Status: blocked at the data-provenance gate, 2026-10-16.** This is a proposed
protocol, not an approved experiment or a claim that `nuc_###` is Z-ordered.
The frozen 2D Nuc model, its split, and all source TIFFs/masks remain unchanged.

## Evidence and the gate

The 167 paired Nuc TIFFs are single-page 1200×1200 files named
`nuc_001.tif`–`nuc_167.tif`. Their TIFF tags contain only shape and `tifffile.py`
software information; there is no acquisition, organoid/field, or Z identifier.
SHA256 comparison found **zero exact byte matches** between the 167 Nuc inputs
and the 1,330 `organoid_data/*/fluo/*.tif` files. This does not exclude a
re-exported or transformed copy, but it supplies no correspondence. The
[data audit](nuc_data_audit_20261015.md) also found a correlation-based
component of 148/167 similar inputs, and 29/34 current validation images have
a training neighbor with correlation at least 0.99. Image similarity warns
about dependence; it cannot certify that two files are consecutive Z planes,
their direction, spacing, field identity, or an organoid boundary.

Before any 2.5D fit, create and freeze a read-only provenance manifest with,
for every center image: input and mask SHA256; original acquisition/stack file;
independent organoid and field identifiers; channel; Z index and physical
spacing; XY dimensions, orientation, and pixel transform; and the matching
center-mask coordinate convention. Check that these fields agree with the
acquisition record, that ordered neighbors are truly within one stack with
no missing/duplicate planes, and that the center mask aligns pixel-for-pixel.
Photometric or perceptual matches can flag candidates for manual review but
cannot replace source provenance. If provenance is unavailable, **do not build
training windows by incrementing `nuc_###`**.
For every 3- or 5-plane window, verify that neighboring images are spatially
co-registered on the center image's XY grid and that the physical Z spacing is
known and consistent enough for the same channel policy. A visually similar
image with an unknown shift or Z gap is not an admissible neighbor.

The labels also need a consistent *center-plane* rule. The reviewer confirmed
`nuc_002`–`005` as true negatives and judged `nuc_006` positive partly using
`nuc_007`. A neighbor can help decide whether a weak structure is a nucleus,
but must not cause a nucleus in a different plane to be painted into the
center mask. Audit such transitions without changing original labels and
record whether existing masks follow a reproducible center-plane convention.

## Conditional experiment, only after all gates pass

1. Freeze the acquisition-level source groups before fitting. Use outer
   cross-validation with all slices from the same organoid/acquisition in one
   fold; any inner validation group used for checkpoint or threshold selection
   must also be disjoint from its training and outer-test groups. Five outer
   folds require **at least five verified independent organoids** and useful
   coverage of positive and true-negative images; more groups are preferable.
   Fewer than five groups means no five-fold claim. A single large biological
   source or similarity component cannot yield independent-group CV by
   subdividing its frames. Report group counts, group-macro and per-slice Dice,
   IoU, precision, recall, object recall, and true-negative false positives;
   do not treat neighboring slices as independent uncertainty units.
   The present 167 labels and reviewed target hints have already influenced
   research hypotheses; regrouping them can yield only exploratory evidence,
   not a pristine confirmatory test. A definitive cross-source claim requires
   pre-frozen evaluation rules and a previously untouched independent source.
2. First compare the unchanged 2D U-Net against a 3-channel 2.5D U-Net on
   *identical verified source-group splits*: input `[z−1,z,z+1]`, output only
   the existing `mask(z)`. Hold the architecture/loss/sampling/normalization
   and model-selection policy fixed except for the necessary channel change.
   Assess a five-channel `[z−2,…,z+2]` input only as a separately declared
   secondary experiment; adding context is not guaranteed to help. At stack
   boundaries, either exclude centers lacking neighbors or use a frozen
   within-stack edge-replication rule. Never borrow across a stack break or
   fold. Apply identical spatial crops/flips/rotations to all input planes
   and the center mask. Output remains 1200×1200 on the center pixel grid.
   The present 2D normalization uses each image's 1st–99.8th percentiles.
   Independently normalizing each neighboring plane could erase meaningful
   inter-slice intensity changes; instead, a clean first comparison should
   calculate the two percentile bounds from the **center input only** and
   apply that same scale to center and neighbors. This leaves the center
   exactly as normalized in the 2D baseline while preserving neighbor-to-
   center intensity relationships. Freeze clipping and this rule before
   training; do not fit any scale from outer-test labels or target outcomes.
3. `code/model/unet.py` already accepts `input_channels=3` or 5, but
   `code/patch_dataset.py` loads and augments one 2D image, while
   `code/sliding_window.py` and `code/predict.py` require a single-channel 2D
   input. They need a provenance-aware stack loader, synchronized channel
   transforms, multi-channel sliding-window inference, and a manifest that
   checks every output against the center file. The current checkpoint is
   one-channel and is not a directly reusable 2.5D model. Preserve the 2D
   baseline and compare after training from scratch on the same group splits.

These gates also apply to prospective prediction. A folder of apparently
numbered `fluo_###` files alone is insufficient to assert physical Z order or
spacing; validate its own acquisition record before using neighbors. Without
that record, the supported prediction route remains independent 2D images.

## One possible 2D preprocessing control, separately gated

The target images have weaker local detail after the existing per-image
percentile normalization, but a prior low-contrast augmentation improved a
synthetic stress test without visibly recovering the reviewed real nuclei.
If a new **source-grouped** evaluation becomes possible, pre-register one
deterministic local-background/contrast transform and apply it identically to
training and inference **before tiling**, for example a reflected Gaussian
background broader than a nucleus and a robust residual scale whose fixed
hyperparameters are chosen using training groups only. Compare that single
preprocessing factor against the same-group 2D baseline; explicitly measure
whether it amplifies bright-ring fragments or false positives on confirmed
empty masks. Do not choose its kernel/scale by repeatedly inspecting the
already-reviewed target images, and do not combine it with 2.5D in its first
test. Until source provenance is verified, restrict this work to read-only
intensity diagnostics and illustrative transforms, not new model selection.

## Why this is a hypothesis, not an expected gain

Stack-to-center segmentation has a clear mechanism for using evidence from
neighboring planes. A [microscopy study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7431134/)
demonstrated 2.5D stack-to-slice modelling for a different image-translation
task. Conversely, a [multi-dataset segmentation comparison](https://arxiv.org/abs/1912.09287)
found a significant multi-slice advantage over 2D in only one of five datasets.
Neither result establishes that this Nuc collection is ordered or that its
unverified masks can support an unbiased 2.5D comparison.
