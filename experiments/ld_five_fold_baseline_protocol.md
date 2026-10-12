# LD 2D U-Net five-fold baseline protocol

**Frozen before any LD GPU training.** This is an independent LD study. It
does not modify or initialize from the Mito checkpoint. The 36 manually
labelled TIFF pairs are a development pool, not independent external data.

## Fixed labels and image groups

Use only the versioned 0/1 derived masks produced by
[`ld_prepare_labels.py`](../code/ld_prepare_labels.py) under
[`ld_antialias_label_protocol.md`](ld_antialias_label_protocol.md). The source
TIFFs remain unchanged. The per-file source/derived SHA256 values and pixel
counts are in [`ld_binary_midpoint_v1_manifest.json`](results/ld_binary_midpoint_v1_manifest.json).
No model metric may change the label conversion rule.

Exact input-file hashes are distinct. However, some input views are visually
similar. Before training, an **input-only** audit scaled each image by its
1st and 99.8th percentiles, averaged each 4×4 block to a 300×300 thumbnail,
and compared mean-centered, unit-L2 vectors. The high-pass representation
subtracted a Gaussian blur with sigma 5 thumbnail pixels. An unshifted pair
was linked when raw correlation was at least 0.92 **or** high-pass correlation
at least 0.08. Six edges formed four conservative suspect groups:
`{001,003}`, `{006,007,034,035}`, `{030,032}`, `{033,036}`. Code in
[`ld_make_fold_map.py`](../code/ld_make_fold_map.py) recomputes these edges
from inputs and fails if the expected groups change. It then assigns whole
groups to five folds using seed 42, balancing image counts **without reading
labels**. The resulting immutable map is
[`ld_five_fold_map_v1.json`](results/ld_five_fold_map_v1.json), with fold sizes
7, 7, 8, 7, 7. Every candidate must reuse it without reassignment.

These are *suspect dependence groups*, not recovered organoid IDs or Z
indices. Other related views may remain unidentified. Consequently, OOF
performance is image-level internal evidence and may overestimate performance
on genuinely new organoids; acquisition metadata or independently labelled
external images would be needed for a stronger specimen-level claim.

## Unchanged baseline scientific configuration

- Independent 2D U-Net from random initialization, base width 32, four
  pooling stages; one image channel and one binary semantic output.
- 512×512 training crops from original 1200×1200 coordinates, no full-image
  resize; 0.6 foreground-centered and 0.4 uniform crop probability. The
  uniform branch retains true background examples despite sparse positives.
- Existing paired flips/90-degree rotations and mild intensity augmentation;
  per-image 1st–99.8th percentile normalization.
- BCE + soft-Dice loss, Adam learning rate 1e-4, batch size 4, 100 steps per
  epoch, at most 100 epochs. No Mito checkpoint transfer.
- Full-resolution sliding-window validation with 512-pixel tiles, 128-pixel
  overlap, and probability blending. Checkpoint/learning-rate/early-stopping
  selection use validation Dice at fixed threshold 0.50 (patience 8/20).
- Only after checkpoint selection, choose a deployment threshold from 0.10
  to 0.90 in steps of 0.05 using mean **validation** per-image Dice.

No factor is changed during these five baseline folds. New run directories
must not exist before launch. A fast GPU preflight and one small fit check
should precede the full run on a verified single RunPod GPU.

## Outer-fold isolation and outputs

For outer fold `f`, fold `(f+1) mod 5` is inner validation and the other
three folds provide training images. [`ld_train.py`](../code/ld_train.py)
reads only training/validation derived-mask bytes and writes its checkpoint,
history, threshold, source hashes, and `selection.lock.json`. The separate
[`ld_evaluate.py`](../code/ld_evaluate.py) may open outer-test labels **only
after** that lock exists and the checkpoint/threshold hashes match. No fold
is repeated because of a low score; technical failures may be retried with
unchanged science and a new output directory. Each outer image is scored once.

Aggregate five folds with [`ld_cv_report.py`](../code/ld_cv_report.py), which
checks complete disjoint OOF coverage. Report per-image and fold-level Dice,
IoU, precision, recall, mean/median/std/min/max, plus fixed 0.50 versus
validation-selected threshold from the **same** checkpoint. Save
probabilities and masks outside Git for later FP/FN and boundary review.
Object-component analysis (especially areas ≤25 px) is diagnostic, since
touching semantic regions are not verified droplet instances. Do not delete
small components by post-processing without a separate, locked validation
study.

## Selection rule for later candidates

Outer-test labels and OOF scores are **reporting only**, never the basis for
choosing loss, sampling, architecture, label threshold, checkpoint, or
prediction threshold. A new one-factor candidate must be defined in advance
and selected against the baseline using inner-validation evidence within
each fold before its outer test is opened. Repeatedly evaluating variants on
these same 36 outer folds and selecting the highest OOF result would create
selection bias. Any such later result must be labelled exploratory and
confirmed on independent labelled external data.

The baseline model family is motivated by the original
[U-Net paper](https://arxiv.org/abs/1505.04597). For small objects, pixel
overlap scores alone can be misleading; the secondary size/FP diagnostics
follow the measurement cautions in
[Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).
