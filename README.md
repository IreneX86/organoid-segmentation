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
  Its [five-fold baseline report](reports/ld_baseline_5fold_20261012/README.md)
  records the 36-image out-of-fold metrics and limitations; the
  [validation error notes](reports/ld_baseline_5fold_20261012/validation_error_notes.md)
  describe FP/FN locations and small-component recall. The
  [literature context](reports/ld_literature_context_20261012.md) separates
  tested evidence from possible future hypotheses. The
  [component-uniform five-fold report](reports/ld_component_5fold_20261012/README.md)
  records the paired exploratory comparison and error audit. Its
  [validation-only method lock](experiments/results/ld_method_policy.lock.json)
  retained the frozen LD baseline. The
  [all-36 final fit and inference handoff](reports/ld_final_all36_20261012/README.md)
  records the locally verified production checkpoint, locked 50-epoch/0.35
  policy, and batch prediction procedure. This final model has no independent
  external accuracy estimate yet. A subsequent
  [small-LD miss / p99.95 normalization study](reports/ld_highlight_p9995_20261014/README.md)
  completed five folds. Its raw candidate did not improve mean OOF Dice or
  low-foreground performance, so the frozen p99.8 LD production model and
  existing predictions remain unchanged. The subsequent
  [symmetric focal BCE + soft-Dice study](reports/ld_focal_gamma2_20261014/README.md)
  also completed five folds; all five preregistered validation gates retained
  the baseline. Its candidate mean OOF Dice was 0.762659 versus 0.763693 for
  the baseline, with more distant false positives. The production model and
  existing predictions remain unchanged. Following a separate human visual
  review, a [focal five-fold voting prediction set](reports/ld_focal_human_preferred_inference_20261015/README.md)
  was first generated and verified for the 100 TIFFs in
  `cell_22375_NZ_100/fluo`. That first set now resides in `fluo_label`;
  its earlier prediction set was moved to `fluo_label_previous` locally.
  The same locked voting method subsequently processed the other nine cell
  folders. All **1,330** current LD masks are available as ten small,
  checksum-verified [LD delivery ZIPs](deliverables/ld_auto_labels/README.md).
  Those ZIPs contain the earlier 3/5 masks.
  After reviewing small-LD detections, the user requested an additional
  [1/5 voting mask set](reports/ld_vote1_auto_labels_20261015/README.md)
  in separate `fluo_label_vote1/` folders for all ten cells. Its 1,330 masks
  are also available as [separate checksum-verified 1/5 delivery ZIPs](deliverables/ld_vote1_auto_labels/README.md).
  The previous 3/5 `fluo_label/` folders and delivery ZIPs are retained.
  This is an operational review preference, not a reversal of the validation
  decision or an independent estimate of generalization. The first 76 images
  were already viewed during review and cannot serve as untouched external
  test images. The handoff has per-image SHA256 records.
- [`deliverables/mito_auto_labels/`](deliverables/mito_auto_labels/README.md)
  contains ten checksum-verified ZIPs with **1,330 preliminary Mito masks**.
  These are model predictions, not manually verified labels.
- [`deliverables/ld_auto_labels/`](deliverables/ld_auto_labels/README.md)
  contains ten checksum-verified ZIPs with **1,330 preliminary LD masks** at
  the earlier **3/5** voting cutoff. They are not manual labels.
- [`deliverables/ld_vote1_auto_labels/`](deliverables/ld_vote1_auto_labels/README.md)
  contains the separately packaged **1/5** voting masks for the same ten cells.
  These are the human-preferred review outputs, not manual labels or an
  independently validated accuracy improvement.
- [`deliverables/nuc_auto_labels/`](deliverables/nuc_auto_labels/README.md)
  contains ten checksum-verified ZIPs with **1,330 preliminary Nuc masks**.
  They are automatic predictions for human review, not manual labels.
- [`runpod_ready/`](runpod_ready/) records the GPU setup and smoke-test scripts.
  [`MITO_EXPERIMENT_LOG.md`](MITO_EXPERIMENT_LOG.md) and
  [`OPTIMIZATION_GOALS.md`](OPTIMIZATION_GOALS.md) index the research history
  and evaluation rules.
- [`membrane_cellpose/`](membrane_cellpose/README.md) is a separate snapshot of
  the membrane-based **cell instance** segmentation code and general usage
  notes. Its raw TIFFs, hand-corrected labels, predictions, model checkpoints,
  and local SHA256 inventory are not included. This snapshot does not change the
  independent 2D organelle pipelines above.

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
The [Nuc data audit](experiments/nuc_data_audit_20261015.md) records 167 binary
semantic-mask pairs and the project owner's confirmation that four all-black
masks are true negatives. Highly similar images make the generic per-image
split unsuitable for a claimed independent Nuc generalization estimate;
freeze an acquisition-aware or similarity-audited evaluation split before
making that claim.
The original LD masks contain Photoshop antialiasing gray levels and cannot
be passed directly to `code/train.py`. Prepare separate binary labels with
[`code/ld_prepare_labels.py`](code/ld_prepare_labels.py); never edit the manual
TIFFs. The LD baseline uses its own frozen fold map and training entry point.

## Train the baseline

Run one target and one cross-validation fold at a time:

```bash
python code/train.py --target mito --fold 0 --num-folds 5
```

The Nuc target is configured in the shared code, but its standalone training
and evaluation protocol is separate from the generic per-image folds. The
[Nuc internal prototype protocol](experiments/nuc_internal_prototype_protocol.md)
freezes a same-batch development split in
[`nuc_internal_split_v1.json`](experiments/results/nuc_internal_split_v1.json).
Use [`code/nuc_train_internal.py`](code/nuc_train_internal.py) only for that
internal diagnostic; it does not produce an independent test score. The source
TIFFs must remain in the project root supplied with `--project-root`, and
`--output-dir` must name a fresh location outside the labelled data folders.
For example, from this repository's root:

```bash
python code/nuc_train_internal.py \
  --project-root "$HOME/Desktop/Organoid" \
  --split-manifest experiments/results/nuc_internal_split_v1.json \
  --output-dir outputs/nuc_internal_v1
```

The full internal prototype run on the real Nuc TIFFs completed on 2026-10-15.
Its downloaded artifacts passed local verification. The
[Nuc internal prototype report](reports/nuc_internal_prototype_v1.md) records
the 33 nonempty validation images' mean Dice of **0.9564518556724969** at the
fixed 0.50 threshold and **0.9565138074004935** at the selected 0.45
threshold, with the validation true negative reported separately. These are
same-batch internal diagnostics, not cross-source generalization estimates.

The one-factor [spatial-illumination protocol](experiments/nuc_spatial_illumination_protocol.md)
and [paired evaluation](reports/nuc_spatial_illumination_20261016.md) record a
research candidate trained on the same frozen Nuc split. Its fixed-0.50
validation Dice improved by only 0.00143, while worst-case Dice and mean
precision declined; the first-ten target-image previews did not establish a
reliable fix for missed nuclei. The original Nuc prototype remains the working
baseline. All candidate checkpoints and TIFF previews stay in ignored local
storage, outside Git.

The separately [preregistered low-local-contrast ablation](experiments/nuc_low_contrast_protocol_20261016.md)
and [paired result report](reports/nuc_low_local_contrast_20261016.md) test a
measured input-domain contrast difference without changing Nuc masks or the
original 2D U-Net. The candidate's fixed-0.50 internal Dice changed from
0.956452 to 0.957296, but its one true-negative validation image had more
false-positive pixels, and the first-ten target previews still missed many
reviewer-circled structures. The original Nuc prototype and existing Desktop
`Nuc_label` remain unchanged. The report includes an input-only distribution
figure and explains why neither internal nor synthetic-stress scores establish
target-domain accuracy.

The subsequent [Nuc source and error audit](reports/nuc_provenance_and_error_audit_20261016.md)
documents the 167-image near-duplicate structure, validation overlap, partial
Nuc/LD source crosslinks, object-level misses and false positives, and one
input-only local-background diagnostic. It found no reliable mapping from the
labelled `nuc_###` files to organoid/Z coordinates. A [conditional 2.5D
protocol](experiments/nuc_25d_feasibility_protocol.md) is prepared, but no
multi-slice model or new background-corrected model was trained; the original
2D Nuc prototype remains the working model.

The [Nuc 2.5D decision and domain-shift follow-up](reports/nuc_25d_decision_and_domain_shift_20261016.md)
checks all ten independent target `fluo` folders, source-export interruptions
inside the 167-image labelled pool, and whether rough missed-nucleus hints
become clearer in adjacent images. The target folders are plausible 2.5D
inference stacks, but the labelled Nuc TIFFs cannot yet be mapped to verified
neighboring Z planes or independent source groups. The report includes four
aggregate figures and a prioritized, source-grouped comparison plan; no new
Nuc model was trained and the working 2D checkpoint is unchanged.

When acquisition/Z mapping is unavailable, the [CPU-only cross-slice review
pilot](reports/nuc_cross_slice_review_20261016.md) uses the archived first ten
target Nuc predictions and adjacent original images to make a local human
review pack. Its image-derived markers are browsing aids, not nuclei labels or
an accuracy estimate; it does not change the working prediction folders.

The [retrospective Nuc marker audit](reports/nuc_dark_core_validation_audit_20261017.md)
checks that fixed review rule on the 34 archived labelled validation images.
At 40 candidates per image it points to 14 of 22 nuclei with <10% frozen-model
coverage, but produces 285 outside-mask review points; the confirmed no-nucleus
`nuc_005` alone has 39 such points. This same-batch check supports manual review
only, not automatic label filling or a target-data accuracy claim.

### Current Nuc prediction method for human review

After inspecting the first ten organoid `fluo` images, the project owner chose
the [0.03-threshold Nuc review method](reports/nuc_review_postprocessing_20261019.md)
as the main **operational** Nuc mask style. It uses the original frozen 2D Nuc
checkpoint, not a newly trained model. The method removes <500-pixel regions,
fills holes, guards radius-4 closing against merges, smooths edges, and
separates only unambiguous two-core touching regions with a curved, smoothed
gap. The 0.45 strong cores guide splitting; they are model predictions, not
manual labels. Output TIFFs keep each input's name and size and encode
background as 0 and Nuc as 255.

Use [`code/nuc_predict_review.py`](code/nuc_predict_review.py) with the frozen
`nuc_unet.keras` checkpoint and a **fresh** output directory. For example:

```bash
python code/nuc_predict_review.py \
  --input-dir "$HOME/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo" \
  --output-dir "$HOME/Desktop/Organoid/organoid_data/cell_22375_NZ_100/Nuc_label_review_new" \
  --model "$PWD/runpod_results/nuc_internal_20261015/full_results/model/nuc_unet.keras"
```

The script verifies the reviewed checkpoint SHA256 and records source/output
hashes and per-image processing decisions. All ten current Desktop cell folders
now contain `Nuc_label/` with **1,330** GPU-generated masks, packaged as
[Nuc delivery ZIPs](deliverables/nuc_auto_labels/README.md). The earlier
approved preview masks remain local and Git-ignored. This preference came from
target-image visual review, so it does not demonstrate a higher external Dice
or justify treating these predictions as manually labelled nuclei. The
167-pair training data and internal prototype remain unchanged.

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

## Exploratory tissue and lumen gate

[`code/tissue_roi.py`](code/tissue_roi.py) adds an optional second stage for a
**manually guided, same-grid** tissue mask. It reads a fluorescence TIFF plus
separate antialiased `org` (tissue) and `bk` (outside background) TIFFs, with
an optional explicit `lumen` TIFF. The declared 16-bit midpoint converts each
guide to a candidate region. With only two guides, it finds the largest
enclosed region marked by neither guide as a *putative* lumen and checks its
topology. With an explicit third guide, it accepts an open lumen touching the
image border. Conflicting and unassigned guide pixels stay uncertain. The
review class map uses outside 0, uncontested tissue 1, lumen 2, and uncertain
255. The conservative inference gate
removes predictions only in definite outside or lumen pixels; it preserves
uncertain pixels for human review. For conventional `cell_*` paths, the tool
checks cell-directory identity and slice number for cross-channel fluo/Mito
pairs, then requires explicit XY-alignment confirmation. Outputs and SHA256
provenance go into fresh paths;
existing organelle predictions remain unchanged.

```bash
python code/tissue_roi.py prepare \
  --fluo /path/to/fluo_094.tif --org /path/to/org.tif --bk /path/to/bk.tif \
  --tissue-out outputs/tissue_pilot/fluo_094_tissue.tif \
  --class-out outputs/tissue_pilot/fluo_094_classes.tif \
  --manifest-out outputs/tissue_pilot/prepare.json

python code/tissue_roi.py apply \
  --fluo /path/to/fluo_094.tif --prediction /path/to/Mito_094.tif \
  --tissue outputs/tissue_pilot/fluo_094_tissue.tif \
  --class-map outputs/tissue_pilot/fluo_094_classes.tif \
  --prepare-manifest outputs/tissue_pilot/prepare.json \
  --out outputs/tissue_pilot/Mito_094.tif \
  --manifest-out outputs/tissue_pilot/gate.json \
  --cross-channel-alignment-confirmed
```

The cross-channel flag records an explicit XY-alignment check for differently
named fluorescence and Mito images; the tool performs no registration. On the
reviewed `cell_22382_140` slice 094, this conservative gate removes 2,463
existing Mito-positive pixels in the putative lumen while preserving 669
positive pixels in uncertain guide regions for review and keeping the
original 1200×1200 pixel grid. It cannot correct errors inside tissue, and
an inaccurate tissue boundary can remove true organelles. The project has one
paired tissue/background guide, so this example cannot establish a general
tissue model or an external Dice gain. A future automatic tissue model needs
representative manual guides from multiple organoids and validation grouped by
organoid; this pilot is for reviewing the two-stage workflow.

The supplied `fluo_105.tif` has a lumen that opens at the bottom image edge,
so the two-guide enclosed-hole rule is unsuitable there. If all three manual
guides are available, pass `--lumen /path/to/lumen_105.tif` to `prepare`; the
same `apply` command then accepts the new preparation manifest. This explicit
three-class option does **not** make tissue segmentation automatic. A fixed
unsupervised local-texture rule trialled on ten `fluo_100.tif` slices produced
incomplete tissue candidates on most slices and is not used to gate existing
organelle masks. Automatic tissue/lumen/background segmentation needs separate
validation, including edge-open cavities, before any batch deployment.

## Scope

The segmentation models use independent 2D images and do not use organoid IDs,
Z indices, adjacent slices, 2.5D inputs, 3D reconstruction, or 3D neural
networks. The optional tissue gate checks cell-folder and slice-name identity
only to prevent accidental cross-image pairing; it does not use neighboring
slices as model input. StarDist, connected-component
analysis, watershed, clDice, and skeleton metrics remain optional later phases.
