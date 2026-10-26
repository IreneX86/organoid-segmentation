# Golgi 2D U-Net exploratory pilot — 2026-10-26

## Dataset and split

Fourteen same-name, 1200×1200 16-bit image / binary manual mask pairs were read from `label_ref/Golgi - Spheroid`. Pixel-identical raw TIFFs link 11 pairs to `cell_16471_NZ_150` and three (`mask_029`, `mask_052`, `mask_075`) to `cell_16472_NZ_150`. The former was used exclusively for training; the latter for checkpoint selection and validation. The older automated `Golgi label` folders were not used as training labels. Original TIFFs and manual masks were not changed.

This is an **exploratory, two-organoid pilot**. The three validation images are correlated slices from one organoid, and checkpoint selection uses their labels. There is no independent test set or demonstrated performance on other organoids.

## Locked run

- Code: `code/golgi_pilot_train.py`, 2D U-Net (32 base filters), BCE + soft Dice.
- Per-image 1–99.8 percentile normalization; 512×512 patches, 128-pixel inference overlap; foreground-centered patch probability 0.6; batch size 4; 40 steps/epoch.
- Adam 1e-4, max 60 epochs, fixed threshold 0.5 for checkpoint selection, patience 12 early stop, seed 42.
- Runpod Secure Cloud A40, one GPU, TensorFlow 2.21.0, Python 3.11, NumPy 2.4.6, tifffile 2026.3.3. GPU preflight included an actual TensorFlow matrix multiplication on `/GPU:0`.
- Payload SHA256: `28602609a9b0635580a469ec7ce570452d30920623463aa125e31bb0ebaffef1`. Full per-file SHA256 and split are in the local output `config.json` and `payload_manifest.json`.

Training stopped after epoch 42; the best checkpoint was epoch 30. The `.keras` file was loaded successfully on the Mac after SHA256-verified transfer.

## Full-image validation at fixed 0.5 threshold

| Image | New Dice | New IoU | New precision | New recall | Old prediction Dice* |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mask_029.tif` | 0.783 | 0.643 | 0.680 | 0.923 | 0.763 |
| `mask_052.tif` | 0.794 | 0.659 | 0.676 | 0.962 | 0.680 |
| `mask_075.tif` | 0.710 | 0.550 | 0.580 | 0.913 | 0.730 |
| **Mean** | **0.762** | **0.617** | **0.645** | **0.933** | **0.724** |

*The prior `Golgi label` masks are used as a descriptive same-image comparison only; their training history is not established here, so this is not a controlled or independent model benchmark.*

The new model recovers more manually labeled pixels but has markedly lower precision, with visible extra predicted spots. One of the three images (`mask_075`) has lower Dice than the old prediction. This is not enough evidence to promote it to a production model. The next reliable evaluation would need complete labels from more independent spheroids, grouped by spheroid rather than splitting neighboring Z slices.

## Local artifacts

- Model: `runpod_results/golgi_pilot_20261026/downloaded/output/golgi_unet.keras` (SHA256 `6f248ca44be752ccef3425e8cf7c2f953c2b1575f9eda7e719372ae83a263772`).
- Training history, config, validation CSV/JSON: same `downloaded/output/` directory.
- Three full-resolution validation predictions: `runpod_results/golgi_pilot_20261026/validation_predictions/`.
- Comparison figure: `runpod_results/golgi_pilot_20261026/golgi_validation_comparison.png` (input, manual GT, old prediction, new prediction, new FP/FN).
- Transfer verification: `runpod_results/golgi_pilot_20261026/local_verification.json` (PASS). Results archive SHA256 `4cf0df19bee6ac46ceaa476c3eab6c4a6e9bd580df285306f698e3e1eabde172`.

Pod `lqw0xh57jf3qdz` was terminated after local verification; Runpod's active Pod list was empty. The listed GPU rate was US$0.59/hour, implying roughly US$0.10 for ~10 minutes. The billing endpoint had not yet posted the final charge at report time, so this is an estimate, not an invoice.
