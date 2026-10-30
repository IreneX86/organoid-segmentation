# ER ridge candidate: batch prediction

The user reviewed a ten-slice preview from `cell_20260814_23298_NZ_50` and chose the relaxed ridge + context version as the next candidate. This is still **classical ER/Fluo image processing, not a trained U-Net**. There are no complete ER ground-truth masks in the current project, so no Dice/recall claim is possible.

The frozen preview method is:

1. Per slice, correct broad ER background and compute bright-ridge response at three spatial scales (1, 2, 4 px). Use Fluo only as a broad tissue-support cue.
2. Make a reliable ER mask with ridge response `>0.035` and high-pass signal `>30`; discard spatially sparse components and small single-Z fragments unless supported by an adjacent layer.
3. Detect additional weak ER candidates with response `>0.025` and high-pass signal `>20`. Keep only candidates within 40 px of the reliable mask; small additions require overlap with an adjacent Z layer.
4. For the **one** field of view whose top-left bright cluster the user called a non-ER artifact, track the corresponding candidate group from a reviewed seed over Z and exclude it. The seed is recorded in [`experiments/er_artifact_seeds_20261027.json`](../experiments/er_artifact_seeds_20261027.json). Other cells do not receive this location-specific exclusion; their suspicious bright spots need independent review.

In the 50-slice trial, the artifact group split at Z49 and its centroid no longer matched the tracked seed. The initial output preserved those uncertain pixels and flagged the layer. A narrow temporal fallback now excludes that layer's pixels only inside the union of the Z48/Z50 artifact bounds, and only when both neighboring matches have bounding-box IoU ≥0.9. The original result remains saved separately for comparison. This is a field-specific correction, not a general bright-spot classifier.

The integrated batch entry point is [`code/er_ridge_batch_predict.py`](../code/er_ridge_batch_predict.py). A ten-slice regression produced a mask pixel-identical to the approved preview (0 differing pixels). The full-stack output can differ slightly at the ten-slice endpoint because additional adjacent Z information is available.
For the completed 50-layer stack, slices 1–9 remain pixel-identical to the ten-slice run; slice 10 differs by 770 pixels because it can now use slice 11 for the predefined adjacent-Z rule.
The execution environment was Python 3.12.11 with versions recorded in [`code/requirements-er.txt`](../code/requirements-er.txt). It uses CPU image filtering; a GPU is not required.

Example for all paired stacks under the ER root:

```bash
python code/er_ridge_batch_predict.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --output-root "$HOME/Documents/ChatGPT/Organoid/outputs/er_ridge_batch_NEW" \
  --artifact-seeds experiments/er_artifact_seeds_20261027.json
```

The script preserves every source TIFF and writes one `ER_mask.tif` plus an audit manifest per cell in a new mirrored output tree. Masks are uint8 `0/255`, same Z/Y/X coordinates as inputs. Output files are compressed and SHA256-recorded. `--resume` only skips previously completed, SHA256-matching cells; it never overwrites a partial or unverified output. Without `--artifact-seeds`, no known-artifact exclusion occurs.

After batch completion, `python experiments/er_batch_verify.py --results-root PATH` checks every TIFF page, source alignment, binary values, manifest pixel counts, and file SHA256; it writes `batch_validation.json` into the output directory. Representative overlays can be rendered with `experiments/er_batch_qc.py`.

These are candidate masks for scientific review. Lower thresholds may recover dim ER but can also admit background texture. Examine representatives from every organoid and Z depth before using masks as measurements or as training labels.

The 2026-10-27 continuation produced [eight full-stack candidate masks and an output index](../outputs/er_ridge_full_batch_20261027/README.md): 900 layers in total. Page-level shape, 0/255 value, per-layer count and SHA256 checks passed. The middle-Z overview and per-cell contact sheets are provided for review. WT `23961` and `23966`, and late MCF `23299` slices, may have visibly missed dim ER; the accepted threshold was not changed after seeing those results.
