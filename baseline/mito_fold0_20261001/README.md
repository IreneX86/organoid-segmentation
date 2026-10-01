# Frozen Mito fold-0 baseline

This folder contains only small, non-image records from the successful 2026-10-01
2D U-Net run. The 36 labelled Mito pairs are a development pool; the seven
fold-0 `test` images are an internal assessment subset, not external unseen data.

- Source: `code/` in the baseline commit. The same source files were used in
  the successful Runpod archive; no model or scientific setting was changed.
- Split: `split.json` (22 train, 7 validation, 7 internal test).
- Training: 100-epoch cap, EarlyStopping after 98 completed epochs, best epoch
  78 (one-based), random 512×512 patches, batch size 4, 100 steps per epoch.
- Best validation Dice at the checkpoint-selection threshold 0.50: 0.865050.
- Validation-selected threshold after training: 0.45. Validation Dice at this
  threshold: 0.865344.
- Internal test, seven images: Dice 0.869262 ± 0.085012 (sample standard
  deviation as reported by the pipeline), IoU 0.776824, precision 0.844737,
  recall 0.905731.
- `training_history.csv`, `threshold.json`, `test_summary.json`, and
  `test_per_image.csv` are the original small output files. They are not
  predictions or masks.
- `environment.md` records the execution environment and command. The complete
  checkpoint and operational logs remain in the local ignored `runpod_results/`
  directory; the checkpoint's local SHA256 is
  `66db8edc3006a9b70fdd75901f6e4a40359699f7a372d28f054d8dec58ecd3`.

Do not tune future models to the two low-scoring images in this fold's test set.
Compare candidates with validation data inside each development split and use
proper cross-validation as specified in `OPTIMIZATION_GOALS.md`.
