# Mito round 2: one-factor inference overlap candidate

Status: **completed; discarded**. This protocol was locked before any candidate inference. This is separate from the
three-pool training experiment and always uses the frozen five baseline
checkpoints. No model is retrained.

## Hypothesis

The baseline uses 512×512 tiles with 128-pixel overlap, blended in probability
space. A read-only descriptive audit of already reported baseline OOF masks
found a false-positive rate of ~1.49% near internal tile boundaries versus
~1.02% elsewhere. Spatial content could confound that observation; it is not
evidence of causality. Test one fixed alternative, **256-pixel overlap**, to
see whether more shared context makes full-image predictions more robust.
Patch size, model, training, normalization, loss, all thresholds and original
1200×1200 pixel coordinates remain unchanged. No 3D or adjacent Z context is
used.

## Decision and evaluation

For each frozen fold, load the **same baseline checkpoint** and apply that
fold's already validation-selected threshold to both overlap settings.
Compare baseline overlap 128 and candidate overlap 256 on that fold's own
validation images only. Use the same predeclared mean/tail/sparse-foreground
decision rule in `mito_round2_gate.decide`: overall mean gain ≥0.003, or
sparse-subset mean gain ≥0.010 with nonnegative overall gain; sparse mean
must not fall >0.005, worst paired validation image >−0.020, and validation
SD must not increase >0.015. This rule is applied separately within each
fold. Save all five decisions before touching any candidate outer-test mask.

Only selected candidate folds receive new outer-test predictions and metrics;
other folds reuse frozen baseline reports. Report all 36 selected-policy OOF
rows, overall and low-foreground summaries, worst-case and fold consistency.
The baseline OOF audit motivated this experiment, so any apparent improvement
on these same 36 images is exploratory, pending independent labelled data.

## Completed inference-only outcome (2026-10-02)

All five folds used their frozen baseline checkpoints and frozen
validation-selected thresholds. Mean validation Dice deltas for 256 minus
128 overlap were −0.000040, +0.000041, −0.000037, +0.000068 and +0.000019.
The gate selected **zero folds**, before any candidate outer-test scoring.
Selected-policy OOF therefore equals the frozen baseline by construction;
this is not an independently measured OOF score for overlap 256. Larger
overlap is discarded because the measured validation effects are negligible.
The exact decisions and selected-policy OOF rows are saved in
`experiments/results/mito_overlap256_decisions.json` and
`experiments/results/mito_overlap256_oof_summary.json`.
