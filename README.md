# Organoid 2D semantic segmentation

This directory implements Phase 1 and Phase 2 of the project: three independent
2D semantic-segmentation pipelines for mitochondria (`mito`), lipid droplets
(`ld`), and nuclei (`nuc`). Each TIFF is treated as an independent 2D image.

## Repository guide and current stage

- [`code/`](code/) contains the reusable training and folder-prediction pipeline,
  tests, and the scripts used to reproduce individual Mito experiments.
- [`baseline/`](baseline/) preserves the original fold-0 run and the frozen
  five-fold reference, including splits and small metric files.
- [`experiments/`](experiments/) contains experiment protocols and selection
  decisions; [`reports/`](reports/) contains the corresponding analyses and figures.
- The new LD study starts with the
  [`antialiased-label rule`](experiments/ld_antialias_label_protocol.md) and
  [`frozen five-fold baseline protocol`](experiments/ld_five_fold_baseline_protocol.md).
- [`deliverables/mito_auto_labels/`](deliverables/mito_auto_labels/README.md)
  contains ten checksum-verified ZIPs with **1,330 preliminary Mito masks**.
  These are model predictions, not manually verified labels.
- [`runpod_ready/`](runpod_ready/) records the GPU setup and smoke-test scripts.
  [`MITO_EXPERIMENT_LOG.md`](MITO_EXPERIMENT_LOG.md) and
  [`OPTIMIZATION_GOALS.md`](OPTIMIZATION_GOALS.md) index the research history
  and evaluation rules.

The first batch-prediction stage is complete. Its input TIFFs remain in the
local `organoid_data/` directory and are not tracked by Git. The delivered
masks were generated with the frozen **fold-0** checkpoint, not a final model
trained on all 36 labelled images. Without independent manual labels for those
organoid images, their external Dice is unknown. Raw TIFFs, checkpoints,
runtime outputs, and credentials are excluded from this repository.

## Experimental data and current baseline

All 36 paired Mito input TIFFs and manually labelled masks in `label_ref/` are
the **labelled development pool**. The fold-0 split of 22 training, 7 validation,
and 7 test images is an *internal* model-development split. Its test images are
not the separate organoid images used for the first batch of automatic
predictions. The latter have no manual masks and were not used to select the
model or report internal test scores.

The frozen Mito fold-0 2D U-Net baseline completed on 2026-10-01. It trained
for 98 epochs (EarlyStopping; best epoch 78) and achieved best validation Dice
0.8651. With a threshold of 0.45 selected on validation data, the seven
internal test images had mean Dice 0.8693, IoU 0.7768, precision 0.8447, and
recall 0.9057. These estimates come from only one small internal split and do
not establish performance on future external organoid images. The exact split,
history, and metrics are in `baseline/mito_fold0_20261001/`. Checkpoints,
raw TIFFs, and Runpod logs are intentionally kept outside Git.

The unchanged full five-fold Mito baseline is documented in
[baseline/mito_five_fold_20261001/README.md](baseline/mito_five_fold_20261001/README.md).
Across all 36 out-of-fold images, mean Dice was 0.8543 ± 0.0628 (sample SD),
with median 0.8756 and minimum 0.7266. These are internal development data,
not an external unseen-data result. Subsequent one-factor Mito candidates did
not establish a reliable improvement, so the frozen baseline remains current.

The completed optimization record is in
[reports/mito_optimization_round2_20261002.md](reports/mito_optimization_round2_20261002.md).
The subsequent read-only error analysis, derived figures and local biological
panel index are in [reports/mito_error_analysis_20261002.md](reports/mito_error_analysis_20261002.md).
The updated 36-image error and annotation-semantics audit is in
[reports/mito_error_and_annotation_audit_20261008.md](reports/mito_error_and_annotation_audit_20261008.md).
Its blinded expert-review protocol is
[experiments/mito_annotation_semantics_review_protocol.md](experiments/mito_annotation_semantics_review_protocol.md);
expert ratings have not yet been collected.
The fixed bright-background hard-negative sampling comparison completed five
Mito folds but selected none by the validation-only rule; the frozen baseline
remains current. See [reports/mito_hard_negative_5fold_20261002.md](reports/mito_hard_negative_5fold_20261002.md)
for the paired OOF and validation-only FP/recall analysis.
The subsequent one-factor focal BCE + soft-Dice experiment selected 2/5 folds
using validation only, but its 36-image selected-policy OOF gain was just
+0.0008 Dice while low-foreground precision and distant FP worsened. It was
rejected; the frozen baseline remains current. See
[reports/mito_focal_dice_5fold_20261008.md](reports/mito_focal_dice_5fold_20261008.md)
for the paired metrics, aggregate chart, and limits of the claim.

Use [OPTIMIZATION_GOALS.md](OPTIMIZATION_GOALS.md) for the next experimental
stage. The original fold-0 baseline commit, tag and artifacts remain unchanged;
folds 1–4 were subsequently trained with the same baseline method.
The Mito data audit, baseline CV, and one-factor experiments are recorded in
[MITO_EXPERIMENT_LOG.md](MITO_EXPERIMENT_LOG.md).
For any future five-fold evaluation, validate reports with
`python code/mito_cv_report.py /path/to/mito/fold_runs` before interpreting an
out-of-fold score. The reporter fails if folds overlap or any test report is
missing; it does not train or alter the data. For each Mito fold, run
`python code/mito_export_predictions.py --project-root /path/to/project
--run-dir /path/to/fold` after training to save test masks at the
validation-selected and fixed 0.50 thresholds. The exporter uses the same
checkpoint for both policies and never selects a threshold from outer-test
labels.

## Geometry contract

- Whole images are never resized.
- Training uses random crops from the original pixel grid.
- Validation and prediction use overlapping sliding-window inference.
- Patch probabilities are blended before thresholding.
- Every output mask has the same filename and `height × width` as its input.
- Internal semantic labels are always `background=0`, `foreground=1`.
- Saved masks are `uint8 0/255` by default; `0/1` is also supported.

## Environment

Create an isolated Python environment, then install the dependencies:

```bash
python -m pip install -r code/requirements.txt
```

TensorFlow GPU installation depends on the operating system and GPU driver. The
pipeline also runs on CPU, but model training will be substantially slower.

## Data layout

The target configurations expect these same-name input/label pairs:

```text
label_ref/
├── Mito/mito_input + Mito/mito_output
├── LD/LD_input     + LD/LD_output
└── Nuc/Nuc_input   + Nuc/Nuc_output
```

The loader compares complete filenames. Missing inputs or labels stop the run
with an error instead of being silently skipped.
The original LD masks contain Photoshop antialiasing gray levels and cannot
be passed directly to `code/train.py`. Prepare separate binary labels with
[`code/ld_prepare_labels.py`](code/ld_prepare_labels.py); never edit the manual
TIFFs. The LD baseline uses its own frozen fold map and training entry point.

## Train the baseline

Run one target and one cross-validation fold at a time:

```bash
python code/train.py --target mito --fold 0 --num-folds 5
python code/train.py --target nuc --fold 0 --num-folds 5
```

For LD, use the versioned derived labels and
`code/ld_train.py` instead, with the frozen map and a fresh output root:

```bash
python code/ld_train.py \
  --input-dir /path/to/LD_input \
  --label-dir /path/to/derived/LD_output \
  --label-manifest /path/to/derived/LD_output/manifest.json \
  --fold-map experiments/results/ld_five_fold_map_v1.json \
  --output-root /path/to/new_ld_run --fold 0
```

Useful options include `--patch-size`, `--overlap`, `--batch-size`, `--epochs`,
and `--steps-per-epoch`. A patch size must be divisible by 16.

Each run writes to `code/artifacts/<target>/fold_<n>/`:

- `baseline_unet.keras`: best model selected by full-image validation Dice.
- `dataset_report.json`: preflight dimensions, dtypes, ranges, and foreground fractions.
- `split.json`: exact train/validation/test filenames and fold assignments.
- `training_history.csv`: epoch-level training and full-image validation metrics.
- `threshold.json`: validation-selected threshold and all threshold candidates.
- `test_per_image.csv`: independent per-image test metrics.
- `test_summary.json`: mean and standard deviation of test metrics.

For a full five-fold comparison, train folds `0` through `4` using the same
settings, then aggregate the five test reports.

## Predict a TIFF folder

```bash
python code/predict.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --model code/artifacts/ld/fold_0/baseline_unet.keras
```

The program reads `threshold.json` next to the model when available. It refuses
to overwrite existing masks unless `--overwrite` is supplied explicitly.

Example mapping:

```text
input/ld_001.tif -> output/ld_001.tif
```

`prediction_manifest.json` records the model, threshold, tiling parameters,
input dimensions, and foreground fraction for every generated mask.

## Scope

This pipeline does not use organoid IDs, Z indices, adjacent slices, 2.5D
inputs, 3D reconstruction, or 3D neural networks. StarDist, connected-component
analysis, watershed, clDice, and skeleton metrics remain optional later phases.
