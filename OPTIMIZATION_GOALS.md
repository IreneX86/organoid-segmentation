# Mito 2D segmentation: optimization goals and evaluation rules

## Scientific objective

Use the 36 manually labelled Mito input/mask TIFF pairs as one labelled
development pool. Choose a reliable **independent 2D semantic segmentation**
pipeline, then train a production model on all 36 pairs. The external unseen
organoid input TIFFs are kept locally outside Git. An initial batch of 1,330
automatic masks has been delivered from the frozen fold-0 model, but these
images have no independent manual labels and cannot provide an external Dice.
A future production model still needs to be trained on all 36 labelled pairs.
Every prediction mask must preserve its input TIFF's filename, width, height,
and pixel coordinates.

The frozen fold-0 baseline is documented in `baseline/mito_fold0_20261001/`.
The unchanged five-fold baseline was subsequently completed and is documented
in `baseline/mito_five_fold_20261001/`; it is the comparison reference for
future **exploratory** one-factor experiments, not an external test estimate.
Its 22/7/7 train/validation/internal-test split is not an external test.
No hyperparameter choice may be made to improve the scores of particular
fold-0 test images. Those image-level results have already been inspected, so
fold-0 is not a pristine holdout for new hypotheses; report that limitation.

## Primary goal and acceptance criteria

Primary metric: mean **per-image full-resolution Dice** on held-out images
across a predeclared five-fold evaluation. Report all 36 out-of-fold values,
fold means, standard deviation, confidence/uncertainty interval, and the
median/lower tail. Compare any candidate to a five-fold run of the unchanged
2D U-Net baseline on exactly the same outer splits. A useful improvement target
is at least **+0.02 absolute mean Dice**, with no material worsening of the
lower-tail images; it is a research target, not a reason to select a model on
test-fold results. Also report IoU, precision, recall, false-positive burden,
and performance versus annotated foreground fraction.

## Leakage and overfitting controls

1. Freeze source, splits, environment, and fold-0 results before any method
   changes. Keep every future run in a new output directory and record the
   exact commit, settings, seed, and data manifest. Never overwrite baseline.
2. Lock the outer five-fold assignment before candidate evaluation. Within
   each outer training pool, use only an inner validation split or inner CV to
   choose augmentation, sampling, loss, architecture, epoch count, and threshold.
   Run the outer held-out fold only after those choices are fixed for that fold.
3. Do not inspect outer-fold predictions to revise that candidate and then
   report the same outer scores as unbiased. Pre-register a small number of
   candidate comparisons; if another method is inspired by outer errors,
   label its result exploratory and reserve external data for confirmation.
4. Check for exact and near-duplicate images across folds before evaluation.
   Do not infer organoid IDs, Z indices, or 3D order. If related-image leakage
   cannot be ruled out, state that limitation in reported Dice.
5. Keep threshold calibration on inner validation/OOF predictions only.
   Apply the chosen threshold unchanged to each outer held-out fold.
6. Preserve raw TIFFs and manual masks. Use the same preprocessing for train,
   validation, and inference; no whole-image resize. Record random seeds and
   all inclusion/exclusion decisions. Never use external unseen images to tune
   after their labels become available without a new locked evaluation plan.

## Planned experiments, one variable at a time

1. **Data/label audit:** verify 36 same-name pairs, binary mask semantics,
   image shape/dtype, blank masks, foreground fractions, duplicates, and
   possible annotation inconsistencies. This is read-only with respect to
   original TIFFs and masks. Visual review may identify ambiguous labels,
   but any relabelling requires separate approval and versioning.
2. **Five-fold baseline:** run the frozen model and settings on all outer folds,
   collecting full-image validation/test Dice and per-image predictions outside
   Git. This gives a comparable baseline distribution rather than relying on
   one 7-image fold.
3. **Sampling/augmentation:** compare foreground-aware versus uniform patch
   sampling, and modest intensity/geometric augmentation that preserves mask
   alignment. Specify ranges before evaluation; retain the 1200×1200 grid and
   overlapping sliding-window inference.
4. **Loss and training policy:** on inner validation only, compare the current
   BCE+Dice loss with a limited alternative emphasizing thin mitochondrial
   structures (for example a predeclared Dice/Tversky weighting). Keep
   architecture, split, and inference fixed for an interpretable ablation.
5. **Architecture benchmark:** compare the current 2D U-Net with nnU-Net v2
   **2D** under the same outer-fold protocol if simpler changes do not provide
   sufficient reliability. Do not use 3D or adjacent-slice models.
6. **Topology metrics, if needed:** add clDice/skeleton and branch metrics only
   when biological use requires connected mitochondrial morphology; do not
   replace semantic Dice as the primary comparison without a documented reason.

Make each implementation change in a separate reviewable commit with comments
explaining non-obvious data or geometry behavior. Add meaningful checks for
mask alignment, fold isolation, and full-resolution output. Do not start any
Runpod optimization job as part of this planning step.

## Production model after method selection

Fix the architecture, preprocessing, loss, threshold policy, and epoch policy
from inner-CV/OOF evidence **before** training the final model on all 36 labelled
pairs. Since all 36 are used for that fit, do not claim it has a new internal
test score. Save a versioned model and inference configuration. The existing
batch masks are preliminary fold-0 predictions, not output of this future
production model. For subsequent unlabelled organoid TIFFs, produce one
same-name, same-size semantic mask per input. Assess external performance only
if independently labelled external data becomes available.
