# LD 1/5 voting masks for ten cell folders

**Status:** Complete on 2026-10-15. The ten `organoid_data/cell_*/fluo/` folders contain 1,330 independent 2D TIFF images. Each now has a new sibling `fluo_label_vote1/` with one same-name, same-size mask per input. Existing `fluo_label/` folders and source TIFFs were preserved.

The user compared representative masks and selected **at least one positive vote out of five** for this automated review set. This changes only the final vote cutoff of the existing focal ensemble. It does not retrain a model or revise the frozen LD baseline. The 1/5 rule favors detection of small candidates and can also add false positives; visual preference is not an independent accuracy estimate.

| Cell | Input and new mask TIFFs |
| --- | ---: |
| `cell_22375_NZ_100` | 100 |
| `cell_22376_NZ_100` | 100 |
| `cell_22377_NZ_120` | 120 |
| `cell_22378_NZ_150` | 150 |
| `cell_22379_NZ_150` | 150 |
| `cell_22380_150` | 150 |
| `cell_22381_140` | 140 |
| `cell_22382_140` | 140 |
| `cell_22383_140` | 140 |
| `cell_22385_140` | 140 |
| **Total** | **1,330** |

## Locked inference method

- Five archived LD focal U-Net fold checkpoints, each at its validation-locked probability threshold: `[0.65, 0.80, 0.60, 0.35, 0.50]`.
- Per-image p1–p99.8 intensity normalization; 512×512 overlapping inference patches with 128-pixel overlap. Whole 1200×1200 images were **not** resized.
- A pixel is LD foreground if **at least one** fold votes positive. TIFF output is single-channel `uint8`, background **0 (black)** and LD **255 (white)**, photometric interpretation `MINISBLACK`.
- The source [1/5 inference entry point](../../code/ld_focal_vote1_predict.py) is separate from the unchanged [3/5 entry point](../../code/ld_focal_ensemble_predict.py). Checkpoint files and raw biological images remain outside Git.

## Integrity checks

The single Secure Cloud A40 run passed a one-GPU TensorFlow matrix multiplication preflight and exited successfully. All ten cell archives and the combined result archive were downloaded with SHA256 verification. Local ingestion checked every input hash, five model locks, output name and shape, manifest hash, and binary mask pixel values. A separate pixel audit confirmed that all 1,330 new masks use `MINISBLACK` and that every old 3/5 foreground pixel remains foreground at 1/5. The first ten masks of `cell_22376_NZ_100` exactly reproduce the previously reviewed 1/5 preview. Final Desktop copies matched their verified local staging files by SHA256. The Pod was deleted and the subsequent Pod list was empty.

The Git-ignored local record is `runpod_results/ld_vote1_bulk_20261015/` and includes the locked input/model plan, result archives, per-image manifests, transfer checks, and publication verification. This report contains no TIFFs, model weights, or Runpod logs.

Two separate sets of checksum-verified ZIPs are available in the private repository: the [earlier 3/5 delivery](../../deliverables/ld_auto_labels/README.md) and the [new 1/5 delivery](../../deliverables/ld_vote1_auto_labels/README.md). The 1/5 ZIPs were packaged from the verified local `fluo_label_vote1/` folders; they contain only generated masks and portable metadata, not raw input TIFFs or checkpoints. The two voting cutoffs must not be mixed during downstream review.

These masks are **automated predictions for review, not manual labels**. No unbiased full-mask test of the 1/5 voting rule has been performed, so no improvement in Dice or precision is claimed.
