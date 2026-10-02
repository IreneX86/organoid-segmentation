# Mito loss experiment 02 — locked protocol

Status: **full five-fold validation-only run complete; candidate rejected by
the locked validation gate**. This is a
separate one-factor candidate, never
combined with the sampling experiment. Baseline reference is the frozen five-
fold run at commit `b6f2e87`.

## Hypothesis and single factor

The baseline loss is BCE + (1 − soft Dice). This candidate keeps BCE and the
same loss scale but replaces soft Dice with a mild Tversky term:

`(TP + 1e-6) / (TP + 0.6 FP + 0.4 FN + 1e-6)`.

Relative to Dice's symmetric 0.5/0.5 weighting, this penalizes false positives
slightly more. The value 0.6 is fixed before training; do not search weights
using the outer-test results. Sampling stays at baseline 0.60. Architecture,
augmentation, normalization, patch size, optimizer, learning rate, batch size,
steps, epoch cap, seed, validation, inference, and threshold selection stay
unchanged.

## Isolation and decision rule

Reuse the same frozen image-level five-fold map. Train five validation-only
candidate models with `--mito-loss bce_tversky_06`,
`--mito-positive-patch-probability 0.6`, `--expected-mito-fold-map`, and
`--defer-mito-outer-test`, each in a fresh output directory. The code does not
open the held-out test mask in this phase.

After all five validation-only runs, run `code/mito_candidate_gate.py
--variant tversky_06`. For each outer fold independently, choose the candidate
only if its best 0.50-threshold full-image validation Dice exceeds the
baseline's by at least 0.01. Otherwise retain that fold's frozen baseline.
The decisions file must be saved before candidate outer-test inference. Never
choose using the mean of all validation folds: those images are outer-test
images for other folds.

Evaluate only validation-selected candidate folds on outer test. Reuse frozen
baseline results for the other folds. Report all 36 per-image OOF Dice, IoU,
precision, recall, distribution (including lower tail), fold means, paired
differences and foreground-fraction relationship. Compare this loss experiment
with the baseline **separately** from the sampling experiment. Do not pick a
combined loss+sampling model from these outer-test results.

The baseline OOF findings helped motivate this factor, so scores on the same
36 images remain exploratory. Fresh independent labelled data will be needed
for an unbiased final generalization claim. Raw masks, TIFFs, checkpoints,
predictions, GPU logs, credentials, and SSH keys must stay out of Git.

On 2026-10-02 Sydney time, a two-fold validation-only pilot was attempted
within the existing Pod's cost allowance. Fold 0 reached 77 recorded epochs
but hit the pilot's 25-minute timeout (exit 124) before training and
calibration completed; fold 1 was not started. Its interim maximum validation
Dice was 0.863576 versus the frozen fold-0 baseline's 0.865050. This interim
number is diagnostic only and must not be used as a completed comparison.
No outer-test label was read and no Tversky model was selected. The incomplete
archive and timeout log are retained outside Git in `runpod_results/`.

The later full five-fold run used a fresh output directory and a 40-minute
technical limit per fold. The first launch stopped before training because
the GPU preflight script path was wrong; the path was corrected and the same
scientific configuration was restarted in another fresh directory. All five
folds then completed with exit 0 and GPU preflight PASS. The best candidate
validation Dice changed relative to baseline by −0.001800, −0.004255,
+0.000316, +0.000272, and +0.003454 across folds 0–4. None reached the
pre-registered +0.01 gate. Consequently, no candidate outer-test label was
read and the selected OOF result remains the frozen baseline. The decisions,
OOF comparison, and SHA256 manifest are in `experiments/results/`. Model
archives and GPU logs remain in ignored `runpod_results/`; the Pod was
terminated after local verification.
