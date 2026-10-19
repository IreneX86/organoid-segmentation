# Nuc 2D review-prediction method (2026-10-19)

## Decision and scope

The project owner reviewed `fluo_001.tif`–`fluo_010.tif` from
`cell_22375_NZ_100/fluo` and chose this mask appearance as the **main
operational Nuc prediction method for human review**. The original 167-pair
Nuc model, training split, labels, and prior masks are unchanged. This is a
postprocessing and threshold choice on already viewed target images, **not**
a retrained or independently validated model. These ten images can no longer
be treated as an untouched external test set for this choice.

The reviewed checkpoint is the original internal Nuc prototype
(`nuc_unet.keras`, SHA256
`c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e`).
It remains in Git-ignored local results; no model weights or TIFFs are stored
in the repository.

## Locked review transformation

Each input TIFF is processed as one independent 2D image without resizing or
using neighboring slices. The model input uses per-image 1st–99.8th percentile
normalization, 512×512 overlapping tiles and 128-pixel overlap. One blended
probability map supplies two binary masks: the **0.03 review candidate** and
the **0.45 strong-core guide**. The latter was the original internal
validation-selected model threshold; it is used here only to guide splitting
and is **not** a manual nucleus annotation.

1. Remove 8-connected candidate components smaller than 500 pixels.
2. Fill enclosed holes. Apply radius-4 morphological closing only when it
   does not merge previously separate components. Smooth jagged edges with a
   5×5 median operation; use component-wise remove-only smoothing when the
   global operation would merge objects.
3. Only when one cleaned component contains exactly two substantial (at
   least 500-pixel) separate strong cores, propose a curved split. Allocate
   pixels by distance to the two model cores, open a tapered background gap,
   and smooth both sides separately near the gap. Restore strong-core pixels.
   Accept the split only if both children are separate 8-connected objects of
   at least 500 pixels and neither contains a hole. Ambiguous or failed cases
   remain unsplit and are recorded in the manifest.
4. Save the input filename and exact pixel grid as a `uint8` TIFF with
   background **0 (black)** and Nuc foreground **255 (white)**.

This rule avoids a straight, visibly artificial dividing line where two
model-predicted cores can safely be separated. A narrow black gap between
separate masks is necessary; the split must not invent foreground outside the
cleaned candidate region. The transform may still merge or miss nuclei, and
the expert must review boundaries and nearby bright debris.

## Reviewed preview and provenance

The ten local-only masks and their per-file SHA256 audit live under
`runpod_results/nuc_threshold003_preview_20261019/` (Git-ignored). The
manifest is `curved_separated_10_manifest.json`. All ten were verified as
same-name, 1200×1200, `uint8` 0/255 TIFFs; the first two stayed byte-identical
to the previously approved two-image preview. Across the ten images, 26
two-core splits passed the geometric checks. These counts describe the output
only; they do not measure Dice or target-domain recall. The preview folder on
Desktop is separate from the earlier `Nuc_label` results.

The reusable implementation is [`code/nuc_predict_review.py`](../code/nuc_predict_review.py)
and [`code/nuc_postprocess.py`](../code/nuc_postprocess.py). Example from the
repository root, writing to a **new** sibling directory:

```bash
python code/nuc_predict_review.py \
  --input-dir "$HOME/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo" \
  --output-dir "$HOME/Desktop/Organoid/organoid_data/cell_22375_NZ_100/Nuc_label_review_new" \
  --model "$PWD/runpod_results/nuc_internal_20261015/full_results/model/nuc_unet.keras"
```

The script locks the checkpoint SHA256 and all reviewed thresholds and
postprocessing parameters. It refuses an existing output directory, writes
no input files, checks each TIFF by readback, and writes
`prediction_manifest.json` only after all selected images complete. For a
small inspection batch, add `--limit 10` and choose another fresh output
directory. An interrupted destination lacks the completion manifest; do not
mistake it for a complete folder or overwrite it. The model and its Python
dependencies must be available locally; no RunPod resource is launched by
this command.

## Interpretation and next evidence

The 167 labelled Nuc images contain extensive near-duplicates and lack a
verified mapping to target organoid/Z sources. The high same-batch validation
Dice of the frozen model therefore does not establish performance on the
organoid TIFFs. Threshold 0.03 was chosen after qualitative target review,
and may increase false positives. A future accuracy claim requires an
independently sourced, completely annotated Nuc set and a source-grouped
evaluation plan fixed before reading its labels. Treat all outputs from this
method as **automatic review masks**, never as corrected ground truth.
