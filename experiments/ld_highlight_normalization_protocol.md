# LD highlight-preserving normalization: prospective single-factor study

**Status (2026-10-14): protocol fixed before candidate training.** This is an
exploratory follow-up to the frozen LD 2D U-Net baseline. It does not alter the
baseline tag, original TIFFs, manual masks, derived binary labels, or existing
production predictions.

## Research question and evidence

The user confirmed six bright, compact structures omitted from
`cell_22375_NZ_100/fluo_label` are LDs. Their centers are recorded only as
points in the local, Git-ignored
`outputs/ld_miss_audit_20261014/confirmed_ld_centers_20261014.json`; no full
boundaries have been drawn. All six exceed their image's p99.8 and p99.95 raw
intensity values. The current per-image p1–p99.8 transform therefore clips
their cores. A **diagnostic only** change to p1–p99.95 at inference, with the
existing p99.8-trained final checkpoint, made four of six centers positive and
left two negative. This simultaneously reduced total predicted foreground in
all six images. On 36 inner-validation images, the same mismatched inference
change lowered mean fixed-0.50 Dice by 0.0094 and recall by 0.0514 while
raising precision by 0.0387. It is **not** a safe replacement for the current
deployed preprocessing.

Hypothesis: training and inferring consistently with a p1–p99.95 linear
mapping preserves useful highlight/halo contrast and improves some small LD
responses. The six known misses motivated the hypothesis and **cannot** serve
as an independent test of it. Their exact masks are unavailable. Existing
inner-validation missed components often had weak signal, so this change may
also harm recall; the experiment must measure that possibility.

## Sole scientific change

For the candidate, change only the **upper normalization percentile from 99.8
to 99.95**, applied identically when loading training patches and when
predicting complete validation/test images. Keep p1 as the lower percentile.
No inference-only switch is eligible for deployment. Do not scan other
percentiles after seeing results.

Reuse the frozen 36-pair derived-label manifest and grouped five-fold map from
[`ld_five_fold_baseline_protocol.md`](ld_five_fold_baseline_protocol.md). Keep
the independent 2D U-Net (width 32, four pooling levels), 512-pixel patches,
0.60 positive-centered sampling, 0.40 uniform sampling, augmentations,
BCE+soft-Dice, Adam 1e-4, batch 4, 100 steps/epoch, maximum 100 epochs,
seeds, early stopping, 128-pixel sliding-window overlap, and all label and
geometry rules unchanged. Start each candidate fold from fresh weights and
write to a new output root. The frozen baseline folds are comparisons only;
do not retrain them.

## Validation-only selection and outer-test isolation

For each outer fold, use its original training/inner-validation/outer-test
partition. Choose the candidate checkpoint using full-image inner-validation
Dice at fixed 0.50. Then choose its threshold on that same inner validation
using the existing 0.10–0.90 grid. Save model, split, configuration, history,
validation predictions, threshold and hashes, and lock the selection before
opening the fold's outer-test labels. Never use the six external point labels
for checkpoint, threshold, method or epoch selection.

Before evaluating a candidate fold's outer-test labels, compare **only that
fold's inner-validation images** with the matching frozen baseline fold. Lock
that fold's decision independently; no other fold's validation labels or
outer-test results may influence it. At fixed threshold 0.50, select the
candidate for that fold's provisional policy only if **all** of
these pragmatic safety conditions hold:

1. Mean per-image Dice is no more than 0.01 absolute below that fold's baseline.
2. At-most-25-pixel, eight-connected semantic-mask component **any-overlap
   fraction** is at least 0.10 absolute above that fold's baseline when the
   baseline fraction is at most 0.90. For a baseline fraction above 0.90,
   where a 0.10 gain may be impossible, require **no loss of component hits**
   and candidate mean per-image Dice at least 0.01 absolute **above** that
   fold's baseline. Report both overlap numerators and the common denominator.
   If the fold has zero such components, this condition is inconclusive and
   the candidate is not selected for that fold.
3. Mean per-image precision is no more than 0.03 absolute below that fold's
   baseline.
4. No validation image loses more than 0.10 absolute Dice.

These margins are operational guardrails, not significance tests. A semantic
mask component is **not** a verified individual droplet. If any condition
fails, select the frozen baseline for that fold's provisional policy; do not
retune the percentile, gate, threshold grid, model or labels to rescue it.
Record **all five** fold-local decisions and their input hashes before opening
**any** candidate outer-test label. Then evaluate the fixed candidate on each
outer fold exactly once, including folds whose validation gate chose baseline,
to obtain a fair raw candidate-versus-baseline paired OOF description. Report
the separately selected hybrid policy only as exploratory. OOF Dice, IoU,
precision, recall, small-component overlap, FP/FN, distant FP and worst-case
images cannot revise any locked gate decision or justify replacing the frozen
production model.

The ceiling rule was fixed before candidate training after inspecting only
the archived frozen baseline validation masks: at fixed 0.50, small-component
hits were 7/7, 40/62, 21/39, 6/11 and 21/31 across folds 0–4. A uniform
``+0.10`` requirement is mathematically impossible for fold 0. These
baseline-only counts did not use candidate or outer-test results.

Because the same 36 images have informed previous LD experiments and rotate
through inner validation in this grouped five-fold design, their OOF results
are not an untouched final confirmation of a globally selected method.
Before replacing the all-36 production model, obtain independently annotated
organoid images with full masks, including verified small LDs, and assess
size-stratified detection, Dice, precision and false positives there. The six
discovery points may be shown qualitatively but may not be counted as an
unbiased validation sample.

No new RunPod resource is part of this protocol-preparation step. Any later
GPU execution must use a fresh output directory, verified single GPU, a
stated cost ceiling, and download/checksum of all key results before Pod
termination.
