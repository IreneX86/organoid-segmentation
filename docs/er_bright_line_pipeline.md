# Reusable ER bright-structure and line-refinement pipeline

## Current user-selected line candidate (2026-10-28)

For new paired **3D** ER/Fluo stacks, use
[`code/er_coherent_fiber_batch.py`](../code/er_coherent_fiber_batch.py). It
reproduces the user-reviewed topology-safe coherent-fiber method: the legacy
radius-4 line supplies a conservative anchor; low-threshold ER ridges are added
only when connected to stronger ridge signal, near the old line, supported by
an adjacent *source* Z slice, and part of a sufficiently long/contacting line
group. The old line is narrowed to radius 2, additions to radius 1. An old
8-connected region is restored if thinning would split it. The method uses
the fixed settings in [`code/er_coherent_fiber.py`](../code/er_coherent_fiber.py).
It is classical image processing, **not a trained U-Net or validated ER
ground truth**. The previous radius-4 method below remains available through
the unchanged historical entry point.

```bash
~/miniforge3/bin/python code/er_coherent_fiber_batch.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --base-mask-root outputs/er_ridge_full_batch_20261027 \
  --profile-map experiments/er_imaging_profiles_20261028.json \
  --output-root outputs/er_coherent_fiber_NEW_RUN
```

The base-mask root may mirror the input directory tree or contain one flat
`<cell_name>/ER_mask.tif` folder per cell, as the current eight-cell cache does.
It must already exist. The profile map must cover every discovered cell exactly
once; for a new imaging condition, review and define its profile first. To
inspect a single source plane, add `--z-start 24 --z-stop 25`; Z indices are
zero-based. The refinement still reads the true source planes 23 and 25, so
processing the plane alone or within a larger batch gives the same result.
Source stacks must contain at least two Z planes. The script validates all
pairs, base-mask shapes and profiles before making the output directory, then
refuses to overwrite any existing output directory.

Each cell gets `ER_mask_line.tif` (**selected candidate**),
`ER_mask_line_legacy.tif` (historical radius-4 comparison),
`ER_mask_bright.tif` (broad comparison) and `manifest.json`. Masks are uint8
with background 0 and candidate ER 255. Multi-plane TIFFs preserve ZYX axes
for Fiji/napari. The manifest records original Z indices, adjacent Z sources,
per-slice audit, image profile and SHA256 of sources, base and outputs. The
output directory is independent of all source TIFFs and prior predictions.
After a full run completes, verify every cell, Z layer, binary pixel value and
source/output hash without modifying any file:

```bash
~/miniforge3/bin/python code/verify_er_coherent_fiber_batch.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --output-root outputs/er_coherent_fiber_NEW_RUN
```

The selected algorithm reproduced all five saved
[`topology-safe review masks`](../outputs/er_hysteresis_fiber_topology_safe_20261028/)
pixel-for-pixel; a new single-plane batch invocation also reproduced its saved
mask. This confirms code parity with the visual review, **not biological
accuracy**. Review early, middle and late planes before quantification.
For the proposed next step of learning to predict ER from **Fluo alone**, see
the [pseudo-label U-Net protocol](er_fluo_only_unet_plan.md); these candidate
masks must not be treated as independent manual ground truth.

`code/er_bright_line_pipeline.py` turns aligned `ER.tif`/`fluo.tif` pairs into two candidate masks, preserving every input pixel coordinate and Z index. It processes either a single 2D pair or a directory of paired 3D stacks. This is **classical image processing, not a U-Net**, and the masks are **not validated ground truth**.

## Stages and provenance

1. Generate an ER ridge base mask with the existing `code/er_ridge_batch_predict.py` method, or read a previously generated, shape-matched `ER_mask.tif` tree via `--base-mask-root`.
2. Add bright, spatially supported ER-like ridges from the ER channel. Fluo supplies a broad tissue-support cue. Compact round bright spots and unsupported fragments are filtered.
3. If a reviewed `--review-rois` JSON is provided, search for ER-like pixels near approved missed structures and exclude explicitly marked ambiguous regions. A circle only selects a region for reassessment; it is **never copied into the mask**. Review coordinates are **zero-based source image pixels** and Z indices.
4. Remove isolated tiny/weak flecks. Produce `ER_mask_bright.tif`.
5. Narrow the existing thick candidate toward its centerline, keeping it inside `ER_mask_bright.tif`. If narrowing splits an existing 8-connected region, restore that region. Produce `ER_mask_line.tif`.

The resulting TIFFs are `uint8` with background `0` and candidate ER `255`. Both have the same Y/X coordinates as their source; a stack's Z range is recorded in `manifest.json`. The manifest also records settings, source/output SHA256 hashes, and per-slice pixel changes. The output root must be new; the script refuses to overwrite an existing directory.

### Optional slight thinning of the existing line mask

The historical line radius is 4 px. Use `--line-half-width 3` in a **new output directory** to keep the same bright candidate and narrow only its final line representation. This option is deliberately conservative: it cannot add pixels beyond the radius-4 line; if narrowing would break one of its 8-connected regions, that region falls back to the original radius-4 line. The default command without this flag remains the historical result. The option changes appearance/line width only; it does not recover a missed ER structure or establish biological correctness.

```bash
python code/er_bright_line_pipeline.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --base-mask-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_ridge_full_batch_20261027" \
  --profile-map experiments/er_imaging_profiles_20261028.json \
  --line-half-width 3 \
  --output-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_line_radius3_new_run"
```

A fixed-seed, five-image review (four spheroids and one dim organoid) is in [`outputs/er_old_line_thin_random5_20261028/`](../outputs/er_old_line_thin_random5_20261028/). Its `manifest.json` records selected source Z planes, per-plane hashes and changed-pixel counts. The old masks and original TIFFs were not modified.

## Batch commands

Use the pinned [ER dependencies](../code/requirements-er.txt). For a new directory of paired stacks:

```bash
python code/er_bright_line_pipeline.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --artifact-seeds experiments/er_artifact_seeds_20261027.json \
  --output-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_bright_line_new_run"
```

To reuse verified base masks and avoid repeating their ridge prediction:

```bash
python code/er_bright_line_pipeline.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --base-mask-root /path/to/mirrored/base_masks \
  --output-root /path/to/new_results
```

The base tree must mirror the input tree and contain one `ER_mask.tif` per cell folder. If available, `ER_mask_manifest.json` is read to preserve an already reviewed artifact exclusion. A single cell folder can be passed as `--input-root`; in that case, point `--base-mask-root` directly at that cell's base mask folder. For a test range, `--z-start 25 --z-stop 26` processes only source Z index 25 (the 26th slice). Other source slices are untouched, and the output manifest records the offset.

Optional review regions use this JSON structure:

```json
{
  "cell_example_NZ_100": {
    "25": {
      "exclude_ellipses": [{"center_xy": [500, 700], "radius_xy": [40, 20]}],
      "rescue_ellipses": [{"center_xy": [250, 300], "radius_xy": [30, 30]}]
    }
  }
}
```

Pass the file with `--review-rois /path/to/review.json`. These entries are field- and slice-specific and must **not** be reused for another cell or Z slice.

### Reproduce the two user-reviewed last-five-slice examples

The later review of `cell_20260814_23299_NZ_200` and
`cell_20260904_23966_NZ_100` is recorded in
[`experiments/er_last5_review_rois_20261028.json`](../experiments/er_last5_review_rois_20261028.json).
The first review removes the user's blue-marked diffuse/unclear candidates and
the upper-edge candidate. The second uses the red-marked upper-left region to
search for missed ER-like ridges; it does **not** fill the marked ellipse.
The approximate source-pixel review regions were intentionally propagated only
across each cell's five adjacent final slices. This is an exploratory,
review-guided correction, not a learned or validated automatic rule.

The two stacks have different depths. Use the tail driver, which resolves each
cell's own global Z indices and checks that the review configuration names
exactly those final five slices:

```bash
python code/er_reviewed_tail_batch.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --base-mask-root /path/to/er_ridge_full_batch_20261027 \
  --review-config experiments/er_last5_review_rois_20261028.json \
  --tail-slices 5 \
  --output-root /path/to/new_er_reviewed_last5_results
```

The base-mask root must contain `<cell_name>/ER_mask.tif` for each selected
cell. The output root must not already exist. Each selected cell receives a
five-slice `ER_mask_bright.tif`, a five-slice `ER_mask_line.tif`, and a manifest
with original source Z indices and hashes. The driver saves its run plan and
completion record at the new output root. Raw TIFFs, base masks and earlier
predictions are never overwritten. For a direct single-cell invocation of the
main pipeline, the review JSON may now use either that cell's name or the
historical `"."` key.

## Relationship to the eight reviewed previews

The eight previously reviewed middle-Z examples used rough hand-drawn circles to remove ambiguous fragments or inspect missed bright structures. Those location-specific choices cannot be inferred automatically in a new field of view. The automatic command applies the reusable image-processing steps; it therefore need not be pixel-identical to the reviewed eight-image masks. The final line-refinement function itself **does** reproduce the eight reviewed line-preview masks pixel-for-pixel when supplied their reviewed bright masks.

Before a full-stack deployment, review several early, middle, and late Z slices in each imaging condition. Compare `ER_mask_bright.tif` with `ER_mask_line.tif`: a blurred bright blob can acquire an artificial centerline even when its internal ER filaments are not visible. Do not interpret the thinner mask as improved biological accuracy without complete ER reference labels. Keep source TIFFs, manual labels, and prior candidate outputs intact.

## Dim-organoid contrast pilot (2026-10-28)

The final sampled stack, `cell_20260904_23966_NZ_100`, is an organoid; the earlier fields include tumor spheroids and have substantially different ER intensity distributions. Across five sampled planes its median ER p50/p99 were 32/154, compared with 796/5246 in the brighter `cell_20260814_23299_NZ_200`. In the dim organoid, p99 fell from 194 at Z22 to 82 at Z92. The original fixed 12-unit high-pass floor and 250-unit relative-background denominator are therefore disproportionately restrictive late in this stack. This is evidence of an intensity-domain mismatch, not proof that every newly detected pixel is ER.

The opt-in `--profile dim_organoid_candidate` changes only the three intensity floors: relative-background denominator 250→150, minimum high-pass 12→6, and strong-anchor high-pass 25→12. The historical `spheroid_default` remains the CLI default and reproduces the previous sampled masks byte-for-byte. No review ellipses or manually marked misses were used in this automatic pilot. On six tested dim-organoid planes Z65,72,81,87,92,96, the candidate line-mask pixels were 77,193 / 83,441 / 66,075 / 58,027 / 45,493 / 37,411 versus default 57,311 / 48,679 / 32,312 / 25,147 / 14,921 / 9,609. The candidate recovers visually plausible faint rings, particularly late in Z, but also adds uncertain internal flecks. These masks are **exploratory candidates, not validated labels**; the larger pixel count is not a measured sensitivity gain. Review the comparison at [`outputs/er_organoid_dim_profile_pilot_20261028/moderate/comparison.png`](../outputs/er_organoid_dim_profile_pilot_20261028/moderate/comparison.png) before choosing it for a full stack.

For a new output directory and this specific imaging condition only, pass `--profile dim_organoid_candidate`. Do not apply it automatically to every organoid or tumor spheroid; verify representative planes and any available ER ground truth first. Both profiles leave the existing base ridge mask, morphology filtering, and bright/line distinction unchanged.

For the eight currently reviewed cells, the explicit [profile map](../experiments/er_imaging_profiles_20261028.json) keeps seven cells on the original settings and opts only `cell_20260904_23966_NZ_100` into the dim-organoid candidate. The batch command is:

```bash
python code/er_bright_line_pipeline.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --base-mask-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_ridge_full_batch_20261027" \
  --profile-map experiments/er_imaging_profiles_20261028.json \
  --output-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_bright_line_profiled_new_run"
```

The map must contain **exactly one entry for every cell folder** discovered under the input root; missing, extra, duplicate or unknown entries cause an error before any output is created. Each output manifest records the selected profile and its numeric settings. This is a reproducible candidate-generation choice, not a claim that the last organoid's entire stack has been biologically validated. For another dataset, create a new map after reviewing the imaging conditions rather than reusing this one automatically.
