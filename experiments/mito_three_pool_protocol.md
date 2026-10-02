# Mito round 2: one-factor three-pool U-Net candidate

Status: **completed; exploratory candidate not promoted**. This protocol was locked before GPU training. Frozen baseline tag
`baseline-mito-fold0-20261001` and five-fold split remain unchanged.

## Why this factor

A read-only 36-image audit found that the median labelled Mito foreground
pixel lies within about 1.4–2 pixels of a mask edge. Baseline U-Net pools
four times, reducing the bottleneck to 1/16 resolution. The hypothesis is
that a three-pool U-Net may retain thin detail and reduce boundary false
positives, with fewer trainable parameters and less small-data overfitting.
This is exploratory: the same 36 images' baseline OOF metrics were already
inspected. No claim of independent external validation is possible.

The candidate changes **architecture only**. It removes the final pooling,
deepest 512-filter block, and matching decoder stage. All retained 3×3
blocks, base width 32, skip connections, normalization, augmentation,
positive-patch sampling 0.6, BCE+soft-Dice, optimizer, learning rate,
512×512 training patches, batch size, 100 steps per epoch, 100-epoch cap,
EarlyStopping, full-image validation, threshold calibration, and 1200×1200
sliding-window inference are unchanged. LD and Nuc remain on their original
architecture.

## Leakage-safe evaluation

Use the exact frozen five-fold map. Each candidate fold trains and calibrates
using only that fold's training/validation images; `--defer-mito-outer-test`
must be enabled. Candidate checkpoint selection still uses its own full-image
validation Dice at threshold 0.5, and the threshold is selected from its
validation data. Before looking at any candidate outer-test mask, the
validation-only `mito_round2_gate.py` loads frozen baseline and candidate
checkpoints and compares same-image full-resolution validation predictions
at their respective validation-selected thresholds.

For each fold select the candidate only if:

1. Overall mean validation Dice increases at least **0.003**, **or** mean
   Dice on that fold's validation images with ≤5% annotated foreground
   increases at least **0.010** while overall mean does not decrease;
2. The sparse-subset mean does not decrease more than **0.005**;
3. No validation image's paired Dice decreases more than **0.020**;
4. The sample SD of validation Dice does not increase more than **0.015**.

These fixed margins are practical safeguards, not a significance test. The
rule is applied within each fold only; averaging validation scores across
folds to select a global method would cross the outer-test boundary. Save all
five decisions atomically before evaluating selected candidate folds on
outer tests. Reuse frozen baseline outer-test results for unselected folds.
Report all 36 selected-method OOF rows and baseline paired deltas, mean/SD,
median/min/max, IoU, precision/recall, low-foreground subset, fold spread,
and worst-case behavior. A small average gain with a worse tail is not enough
to call the candidate more reliable. Never rerun a fold because of its score.

## Audit findings informing priority

Read-only analysis of the frozen OOF predictions found 7 images with ≤5%
foreground. Their recall was similar to the other 29 images (~0.877), while
precision was much lower (0.727 vs 0.866). The false-positive rate per
background pixel was similar (~1.1%) in both groups, making the small
foreground denominator consequential. About 74% of false-positive pixels
belonged to predicted connected components that also touched true Mito,
so deleting isolated tiny components is unlikely to address the main error.
Raw TIFF intensity varied widely, but per-image percentile normalization
already removes much of that scale; a simple background-correction hypothesis
was not compelling enough to prioritize over the thin-structure architecture.
The per-image median normalized intensities of false-positive and false-
negative pixels averaged about 0.344 and 0.309 respectively, so a simple
global brightness cut is unlikely to separate these errors cleanly.
These are descriptive findings from previously viewed internal OOF images,
not data used to tune the candidate's parameters.

All TIFFs and manually labelled masks stay unchanged. Checkpoints, GPU logs,
prediction TIFFs, transfer archives and secrets must remain outside Git.

## Completed five-fold outcome (2026-10-02)

All five validation-only candidate folds completed on one Secure Cloud A40
Pod. The validation gate selected only fold 3. Its seven outer-test images
were scored **after** all five decisions were saved; the remaining 29 OOF
rows reuse frozen baseline predictions. The resulting selected-policy OOF
Dice was **0.854928 ± 0.062555**, compared with frozen baseline **0.854309
± 0.062776** (paired mean gain +0.000619). Low-foreground mean Dice changed
from 0.791388 to 0.791909; the worst-image Dice stayed 0.726555.
Precision fell from 0.839055 to 0.837998, while recall rose from 0.877258
to 0.879598. This small gain is restricted to one fold and does not establish
a robust improvement. The three-pool model is retained as an experimental
option, but the frozen baseline remains the recommended pipeline. Exact
fold decisions and all 36 selected-policy OOF rows are in
`experiments/results/mito_three_pool_decisions.json` and
`experiments/results/mito_three_pool_oof_summary.json`. Raw candidate
outer-test metrics were deliberately not measured for the four rejected folds.
