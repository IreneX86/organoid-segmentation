# LD experiment 01 — component-uniform positive patch centers

**Status:** prospective, single-factor protocol. Written from the frozen LD
label/input audit and baseline training specification, before inspecting any
baseline outer-test prediction or metric for this hypothesis. No candidate
training or outer-test evaluation has been run under this protocol. Record the
protocol file SHA256 before execution; later results must not change this rule.

## Research question and rationale

The 36 versioned LD masks contain only 0.00729%–0.4846% foreground per image.
The pre-training audit counted 1,851 eight-connected **semantic-mask
components**, with median area about 47 pixels. These are proxies for separate
annotated regions, **not** verified individual biological droplets. The
baseline chooses the center of a positive training crop uniformly from all
foreground pixels. Consequently, a component receives positive-center
probability proportional to its area; small annotated regions get relatively
few center selections. The fixed hypothesis is that sampling components
uniformly within each chosen training image will improve coverage of small
regions and reduce their false negatives without a material precision loss.
It is an untested hypothesis, not a finding from baseline predictions.
Because a 512×512 crop can include other annotated regions even when centered
elsewhere, the predicted benefit may be modest; the experiment tests it rather
than assuming it.

This policy changes *only the positive-center selection distribution*. It
does not relabel components, create instance targets, discard tiny regions, or
change the proportion of positive-centered crops. Component areas and any
small-object outcome must be interpreted cautiously because antialias-derived
boundaries and annotation uncertainty remain.

## Frozen data, split, and scientific controls

Use the immutable `ld_five_fold_map_v1.json`, the 36 same-name inputs, and the
`ld_antialias_midpoint_v1` derived binary labels with SHA256 verification. Keep
the original TIFFs untouched. For outer fold `f`, fold `(f+1) mod 5` is inner
validation, three folds are training, and `f` is outer test. Suspect
near-duplicate groups remain intact. The experiment is LD-only and trains
from random initialization; it does not load or alter a Mito model.

The comparator is the frozen five-fold LD baseline. Hold its 2D U-Net width
and depth, BCE + soft-Dice loss, image normalization, 512×512 patch size,
128-pixel inference overlap, augmentation, optimizer and learning rate,
batch size, 100 steps per epoch, 100-epoch cap, callback patience, random seed,
full-image checkpoint selection, threshold grid, and 0.60/0.40
positive-centered/uniform-crop branch probabilities fixed. Use the same
input-image sampling distribution and exactly one candidate run per fold.
Restart a fold only for a documented technical failure, never for a low score.

### The sole candidate change

For each selected **training** image, compute eight-connected components of
its derived 0/1 mask once and cache their pixel coordinate lists. In the
existing 0.60 positive-center branch, draw one nonempty component uniformly,
then draw one of its foreground pixels uniformly as the crop center. Keep the
baseline crop clipping and augmentation. In the existing 0.40 branch, keep
uniform crop-origin sampling exactly as before. If a training mask has no
foreground, use the baseline uniform-crop fallback. Do not filter components
by size or intensity, weight images by component count, introduce hard
negatives, or alter any validation/test mask. Record component counts and the
candidate code hash for each fold; calculate these counts from **training
masks only** in that fold.

Implementation must have unit tests showing: all **positive-branch** centers
lie in the selected training mask; components are approximately equiprobable
under a fixed synthetic mask with unequal component areas; the 0.40 uniform branch
and output patch geometry remain unchanged; a candidate run does not open
outer-test mask bytes before its selection lock. Keep the baseline sampler's
default behavior intact and the candidate in a separate fresh output root.

## Checkpoint, threshold, and per-fold decision

For each candidate fold, use its inner-validation full-image mean Dice at
**fixed threshold 0.50** for checkpoint, learning-rate and early-stopping
decisions, exactly as baseline. After checkpoint selection, use only that
fold's validation images to choose its deployment threshold on the same
0.10–0.90 grid in steps of 0.05. Archive the checkpoint, training history,
validation probabilities, fixed-0.50 and selected-threshold masks/metrics, split,
configuration, hashes, and threshold before any outer-test evaluation.

The comparison gate is specified in advance and applied **separately for
each outer fold** using the same images in the candidate and baseline inner
validation sets. At the saved best checkpoint of each run, compare per-image
Dice at fixed threshold 0.50. Select the component-uniform candidate for
outer fold `f` only if all of the following hold:

1. Mean paired validation Dice difference (candidate minus baseline) is at
   least **+0.01** absolute.
2. A strict majority of validation images have positive paired Dice change
   (at least 4/7 or 5/8, according to that fold's size).
3. No validation image loses more than **0.10** absolute Dice.
4. Mean validation precision at 0.50 falls by no more than **0.03** absolute.

Otherwise select the frozen baseline for that outer fold. These margins are
pragmatic guardrails against a single-image gain or large precision tradeoff,
not statistical significance thresholds. Ties fail conditions 1–2. If a
candidate cannot complete or lacks a valid locked checkpoint, mark that fold
unresolved; do not treat a technical failure as evidence for baseline.
Create an immutable decision record with per-image paired validation values,
all four gate checks, chosen checkpoint/threshold hashes, and protocol hash
**before** opening that fold's outer-test label. The candidate is never
selected by outer-test Dice, threshold, or visual appearance.

## Evaluation and interpretation

After all five per-fold decisions are locked, evaluate only the chosen model
for each fold on its outer-test images using its own validation-selected
threshold. Reuse the frozen baseline outer-test outputs for folds where
baseline won; do not retrain those models. Report all 36 images exactly once
as the OOF result of this **predeclared validation-gated procedure**, plus a
paired comparison to frozen baseline: Dice, IoU, precision, recall,
mean/median/sample SD/minimum, fold means, and per-image changes. Report the
fixed-0.50 comparison offline from the same probabilities. Diagnostic
secondary outcomes include foreground fraction, FP and FN pixels, distant FP
(>10 pixels from annotated foreground), and recall proxies for ≤25-pixel
eight-connected GT components (any overlap and ≥50% pixel coverage). State
explicitly that these components are not verified object instances. Inspect
representative input/GT/probability/FP-FN overlays for boundary and possible
annotation ambiguity; do not edit labels.

Do not call an apparent OOF increase a confirmed production improvement.
Because each image is outer test in one fold and inner validation in another,
using **aggregate results from all five validation folds** to choose a
single global candidate would expose an outer-test image's label to that
global choice. Only the fold-local gate above protects each corresponding
outer test from that direct route. Moreover, deciding to keep this candidate
or launch another variant because its five-fold OOF score is high introduces
study-level selection bias on the same 36 images. The gated OOF result is
internal, exploratory evidence; specimen-independent generalization and a
single production policy require independently labelled future data or a
separately predeclared validation-only production selection rule. A weak or
unstable candidate leaves the frozen baseline unchanged. Do not tune the
0.60 branch probability, component rule, gate margins, labels, or threshold
after seeing these OOF results.

## Reproducibility and resource safety

Run one verified Secure Cloud GPU Pod at a time, with a short GPU preflight.
Use new run directories and an explicit budget/cost record. Download and
verify all checkpoints, decision files, probabilities, metrics, and logs
before terminating the Pod; confirm zero running Pods and zero spend rate.
Commit only protocol, code, hashes, and small derived metrics to the private
repository. Keep TIFFs, masks, predictions, checkpoints, large logs, and
credentials out of Git. Do not alter the frozen Mito baseline or Cetus.

Background: [U-Net](https://arxiv.org/abs/1505.04597),
[Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z).
