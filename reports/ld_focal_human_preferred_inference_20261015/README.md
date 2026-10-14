# Human-preferred LD focal voting predictions for one 100-image folder

**Status (2026-10-15, Australia/Sydney): 100/100 masks generated and verified.** This record documents a separate operational review output for `cell_22375_NZ_100/fluo` only. It is not a new model-selection result or a claim of improved external accuracy.

## Decision and scientific boundary

The user visually reviewed predictions and preferred the new focal five-fold vote for this folder. The first 76 images were already shown during the review, so they cannot later be counted as untouched external test images. A visual preference is useful for choosing a review copy but supplies neither complete independent manual masks nor an unbiased Dice, precision or recall comparison.

The [original focal experiment](../ld_focal_gamma2_20261014/README.md) and its locked decisions remain authoritative for model selection: **all five preregistered inner-validation gates rejected focal**. Across the 36 repeatedly inspected development images, raw focal candidate OOF Dice was **0.762659** versus **0.763693** for the frozen baseline, with more distant false positives. Those OOF values are exploratory internal evidence. This separate prediction request does not change the frozen baseline, its all-36 production checkpoint, `fluo_label`, the focal report, or any decision lock. These predictions are model outputs for expert review, not manually verified LD labels.

## Requested inference and output contract

| Item | Locked review procedure |
| --- | --- |
| Input scope | The 100 TIFFs in `~/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo/` only |
| Checkpoints | The five previously archived LD focal fold checkpoints (`fold_0` through `fold_4`) under `runpod_results/ld_focal_20261014/downloaded/assembled/full/ld/` |
| Preprocessing | Per-image intensity normalization from p1 to p99.8 |
| Geometry | 512×512 sliding tiles, 128-pixel overlap, blended full-image probabilities, no resizing |
| Fold thresholds | Validation-locked `[0.65, 0.80, 0.60, 0.35, 0.50]` for folds 0–4 |
| Vote | Threshold each fold's probability map, then foreground when at least 3 of 5 folds vote foreground |
| Masks | Same filename and pixel dimensions as input; single-channel `uint8`, background 0 and foreground 255 |
| Local handoff | `~/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo_label_2/`, separate from the original `fluo_label/` |

This 3/5 voting ensemble has not itself passed the preregistered validation gate or received an independent, complete-mask evaluation. The fold thresholds were selected for their individual models, not tuned for this vote on these 100 images. Do not interpret the handoff as a replacement for the frozen production method.

For a later fresh folder, the reusable [inference entry point](../../code/ld_focal_ensemble_predict.py) is:

```bash
python code/ld_focal_ensemble_predict.py \
  --runs-root /Users/irenex/Documents/ChatGPT/Organoid/runpod_results/ld_focal_20261014/downloaded/assembled/full/ld \
  --input-dir /Users/irenex/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo \
  --output-dir /path/to/a/new_output_directory
```

`--runs-root` points to five locally archived, Git-ignored checkpoints and selection locks; model weights are not stored in Git. The output directory must be new. On a rerun, the entry point accepts only its own already complete `prediction_manifest.json` with matching hashes; it does not adopt or overwrite partial or mismatched output. This command is the reusable path for future inference, not the procedure used to resume the interrupted first 76 masks in this handoff.

## Provenance and completion check

The Git-ignored workspace audit directory is `outputs/ld_focal_review_cell22375_20261015/`. Its complete [per-image manifest](../../outputs/ld_focal_review_cell22375_20261015/review_prediction_manifest.json) records all 100 filenames, source/mask SHA256 values, five checkpoint and selection-lock hashes, and the locked thresholds. Its own SHA256 is **`4ac869d957f9fdb1bda97598652f2b7071542315b6950ec3d59481a7dbbea9f5`**. The identical manifest is at `~/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo_label_2/prediction_manifest.json`, next to the 100 masks and `README.txt`. The five archived `focal_unet.keras` files and their `threshold.json` files remain in `runpod_results/ld_focal_20261014/downloaded/assembled/full/ld/fold_0/` through `fold_4/`.

The final verification checked the exact 100-name pairing, input and output dimensions, `uint8` 0/255 values, and every source and output SHA256 against the manifest. It also checked every Desktop mask against the audited workspace result. The first 76 Desktop masks kept their original hashes; only the remaining 24 masks were copied into place. The earlier `review_prediction_manifest_partial.json` covered 76/100 images and is retained only as an interruption record. The final 100 masks contain 321,307 foreground pixels in total; this count is a file-integrity summary, **not** an accuracy metric.

No manual full-image LD contours have been collected for this folder. A future unbiased comparison requires newly chosen images that have not driven method or output selection, complete independent manual contours, and endpoints fixed before scoring.
