# Organoid 2D semantic segmentation

This directory implements Phase 1 and Phase 2 of the project: three independent
2D semantic-segmentation pipelines for mitochondria (`mito`), lipid droplets
(`ld`), and nuclei (`nuc`). Each TIFF is treated as an independent 2D image.

## Experimental data and current baseline

All 36 paired Mito input TIFFs and manually labelled masks in `label_ref/` are
the **labelled development pool**. The fold-0 split of 22 training, 7 validation,
and 7 test images is an *internal* model-development split. Its test images are
not the unseen organoid images that the final model will predict; those external
images have not been added to this project.

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
not an external unseen-data result. No optimized model has been trained yet.

Use [OPTIMIZATION_GOALS.md](OPTIMIZATION_GOALS.md) for the next experimental
stage. The original fold-0 baseline commit, tag and artifacts remain unchanged;
folds 1–4 were subsequently trained with the same baseline method.
The read-only Mito data audit, first pre-registered comparison, and current
five-fold evidence gap are recorded in [MITO_EXPERIMENT_LOG.md](MITO_EXPERIMENT_LOG.md).
After all five unchanged baseline folds finish, validate their reports with
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

## Train the baseline

Run one target and one cross-validation fold at a time:

```bash
python code/train.py --target ld --fold 0 --num-folds 5
python code/train.py --target mito --fold 0 --num-folds 5
python code/train.py --target nuc --fold 0 --num-folds 5
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
