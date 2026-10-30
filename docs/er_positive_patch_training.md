# ER-positive paired patches — 2026-10-29

## Clarified objective

Reuse the accepted ER-derived positive mask and extract exactly corresponding
Fluo crops. Do not add nERdy+ predictions or require tracing all missing ER.
Fluo alone is the model input; the ER-derived mask is the supervision target.

`code/er_positive_patch_dataset.py` produced **760 paired 384×384 crops** from
the six training organoids, containing **13,608,024 old positive pixels**.
One densest fixed-grid tile is selected per positive training slice. This is
positive-enriched sampling, not exhaustive coverage of every positive pixel.
Adjacent slices are correlated and are not independent biological samples.

Local dataset: `outputs/er_positive_patch_dataset_20261029/`.
Each compressed NPZ contains:

- `fluo`: unchanged raw intensity crop, sole model input.
- `er_positive_mask`: identical crop of the old 0/255 pseudo-mask.
- `partial_label`: 1 at old positives, 255 at unknown pixels.

Raw ER-channel intensities are not included. The manifest records source hashes,
Z/XY positions and crop hashes. Whole validation organoid 23963 and test organoid
23300 are excluded, and their pixels are not read. Original data are unchanged.
Preparation refuses an existing output directory.

## Training status and next step

Paired positives are prepared; training has NOT started. Old labels remain
pseudo-labels, not complete manual truth. `supervision()` returns a binary target
and valid-pixel weight for an ignore-aware loader. Weights must be applied inside
both BCE and Dice. The old binary U-Net runner cannot consume this format as-is.

Positive-only supervision admits a trivial all-positive solution. The explicit
training guard therefore refuses a dataset without documented negatives. Next,
define conservative clear-background supervision separately from unknown tissue,
or preregister a positive-unlabeled objective and its assumptions. Do not turn
every unmarked tissue pixel into background. This does not require correcting all
missing ER lines. No paid GPU resource is created by this preparation.

Validation must use the same defined supervision semantics on a separate whole
organoid. Known-pixel agreement is not full biological ER accuracy. Generalizing
to previously unlabeled ER remains to be demonstrated.

## Commands

```sh
python code/er_positive_patch_dataset.py prepare \
  --mask-root outputs/er_coherent_fiber_full_20261028 \
  --output outputs/er_positive_patch_dataset_20261029
python code/er_positive_patch_dataset.py verify \
  --output outputs/er_positive_patch_dataset_20261029
python -m unittest discover -s code/tests -p test_er_positive_patch_dataset.py
```

The dataset and large artifacts remain local. No model training or paid compute
was started. Continue with negative-supervision design and an ignore-aware loss,
not another automatic missing-line generator.

## Next-step pilot completed: weak exterior negatives

`experiments/er_background_pilot.py` read the middle plane of each of the six
training stacks. Before viewing results, a conservative fixed rule was implemented:
Gaussian sigma 2; both ER and Fluo below their respective full-plane 20th
percentile; four-connected to a full-image boundary; erosion four pixels;
at least 12 pixels away from teacher positives. No threshold scan was performed.
Enclosed dark nuclei do not become negative just by being dark. Automatic
negatives are **weak proposals**, not certified background. Percentiles adapt
to each plane without learning from any held-out organoid.

Outputs: `outputs/er_background_pilot_20261029/`, including six aligned NPZs,
SHA256 manifest and `background_review.png`. The 760 original patches remain
unchanged. These full-plane proposals are a pilot, not yet a complete patch
training/validation dataset. No ER intensities are included in their model inputs.

| Training cell | Old positive pixels | Weak exterior negative pixels |
| --- | ---: | ---: |
| 23298 | 68,045 | 164,905 |
| 23299 | 89,050 | 188,680 |
| 23961 | 38,512 | 110,067 |
| 23962 | 57,925 | 209,610 |
| 23965 | 44,345 | 63,892 |
| 23966 | 79,885 | 169,207 |

Visual inspection shows proposals mainly outside the tissue, but this does not
establish their biological correctness. The rest remains ignored. Exterior-only
negatives cannot establish specificity among intracellular structures, so this
is not sufficient evidence that a student will avoid intra-tissue false positives.
The positive-patch strategy can be tested with this limitation stated, without
claiming the resulting network detects all true ER.

`code/er_partial_supervision.py` adds a TensorFlow masked BCE+Dice loss. Local
unit checks passed: ignored pixels cannot change the loss and have zero gradient;
all-ignored inputs return finite zero; correct known-pixel predictions lower loss.
A synthetic image also confirms enclosed dark holes are excluded and positives
protected. These are software checks, not model training or accuracy tests.

Next implementation step: build separate positive-rich and exterior-background
patch streams using the fixed rule, apply identical preprocessing to Fluo,
prepare whole-organoid validation with the same partial-label semantics, and
connect the tested loss to a separate student runner. Keep the old runner and
failed checkpoint unchanged. No paid resource or training started in this pilot.

Update: that implementation step is complete. See
[the separate partial-label student protocol](er_partial_student_protocol.md).
The prepared dataset has 1,520 train and 80 validation patches; the GPU-only
trainer is `code/er_partial_student.py`. No model has been trained in this step.
