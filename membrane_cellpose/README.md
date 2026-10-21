# Membrane cell-instance segmentation snapshot (2026-10-21)

This directory backs up the **code and general documentation** currently found
in the local `Membrane/cellpose_seg` folder. Detailed source-file provenance
is retained locally outside Git. No
training, inference, relabeling, or model selection was performed for this
snapshot.

## Task and pipeline

- Inputs: aligned `Memb.tif` (membrane) and `fluo.tif` (cytoplasmic signal),
  both `(Z, 1200, 1200)` `uint16`.
- `cellpose_instance_seg.py predict` stacks the channels and runs Cellpose v4
  CPSAM on independent 2D slices. Its `stitch_threshold` then joins 2D masks
  by adjacent-Z overlap into `(Z, Y, X)` `uint32` instance labels. Label 0 is
  background; positive integers are instance IDs. This is **not** a learned
  3D segmentation network.
- `finetune` uses available instance annotations. `napari_correct.py` edits
  selected regions; `eval_dense.py` evaluates against corrected dense labels.
  `nucleus_split.py` is an unsuccessful exploratory split heuristic and is not
  recommended by the original notes.

## Verified local inventory

The checked local dataset contains four volumes with both required image
channels and previously generated instance-label TIFFs. The source directory
also contains sparse and corrected labels, fine-tuned model files, and
predictions. A detailed relative-path, byte-size, and SHA256 manifest was
saved **locally** in the ignored `outputs/membrane_snapshot_20261021/`
directory. It was not uploaded to this public repository. A hash manifest is
**not a data backup**.

## Important provenance and evaluation limits

- Corrected-region JSON sidecars still point to a historic Windows path
  and must be mapped to the current Mac input directory before `finetune` is
  run. Do not silently rewrite the original sidecars.
- The original README recommends v5, while a v6 checkpoint and predictions
  exist without a documented independent comparison. No version is promoted
  here.
- Some corrected regions were seeded from model predictions. Scores against
  those same seeds are not independent validation. Slices from one 3D volume
  are correlated, so a split of nearby slices cannot measure transfer to a
  different cell or organoid.
- Prior read-only QC found single-slice instances, which may be true cells or
  stitching artifacts. An uninterrupted ID span does not rule out incorrect
  cross-Z identity linking.

## Reproducing the code environment

`requirements.txt` records the historic Windows/Python 3.9 setup: PyTorch
2.4.1/CUDA 12.1, Cellpose 4.2.1.1 and napari with Qt5. The current Mac
`napari-cells` Python 3.12/PyQt6 environment has been used for visualization,
but this snapshot has **not** verified that it can run the historic Cellpose
training or inference scripts. Use a separate compatible environment and
validate on a small slice before any new run.

## GitHub backup boundary

The GitHub repository was reported by the GitHub API as **public** at snapshot
time. Therefore this commit includes only source and general documentation.
Raw biological TIFFs, corrected/manual labels, prediction TIFFs, checkpoint
weights, and the detailed local file manifest are excluded. A real off-site
data backup requires a private, access-controlled destination and a transfer
method suitable for large files; completion must be verified against the local
manifest.

## Next experiment

First annotate and freeze independent, dense review regions across cells and
Z depths. Score 2D merges/misses/fragments separately from cross-Z identity
errors. Keep whole-cell/organoid groups separate for final evaluation, then
compare v4/v5/v6 and one change at a time. Existing predictions are preserved.
