# Golgi 2D U-Net exploratory checkpoint

Download the `golgi_unet.keras` asset from the
[Golgi pilot GitHub Release](https://github.com/IreneX86/organoid-segmentation/releases/tag/golgi-pilot-20261026).
It is deliberately distributed as a Release asset rather than a normal Git
blob. Verify SHA256 before inference:

```text
6f248ca44be752ccef3425e8cf7c2f953c2b1575f9eda7e719372ae83a263772  golgi_unet.keras
```

This is the best checkpoint (epoch 30) from the 2026-10-26 **from-scratch**
Golgi pilot. It was **not** initialized with Mito, LD, Nuc, or an earlier
Golgi checkpoint. The network architecture and shared patch/sliding-window
utilities were reused, but the weights were trained exclusively on 11 manual
Golgi masks from `cell_16471_NZ_150`. Three masks from one other spheroid,
`cell_16472_NZ_150`, were used to select the checkpoint and report exploratory
validation metrics. Thus it has **no independent test score**.

Environment used for the training run: Python 3.11, TensorFlow 2.21.0,
NumPy 2.4.6, tifffile 2026.3.3. On a compatible Linux NVIDIA GPU, the
TensorFlow package may be installed as `tensorflow[and-cuda]==2.21.0`.
Prediction also works on CPU with `tensorflow==2.21.0`.

From the repository root, run:

```bash
python code/golgi_predict_folder.py \
  --checkpoint /path/to/golgi_unet.keras \
  --input-dir /path/to/Golgi \
  --output-dir /path/to/new/Golgi_label_pilot
```

The output folder must be new. Each independent 2D TIFF is normalized by
its own 1st and 99.8th intensity percentiles; overlapping 512×512 patches
with 128-pixel overlap are blended before applying the fixed 0.5 threshold.
The output mask retains the original filename, height, width, and pixel
coordinates, with uint8 background 0 and Golgi foreground 255. No Z context
or instance identity is inferred. A `prediction_manifest.json` records file
hashes and the applied method.

The [pilot report](../../reports/golgi-pilot-20261026.md) and
[small run records](../../experiments/results/golgi_pilot_20261026/) include
the exact split, training history and per-image validation metrics. Original
TIFFs, manual masks, prediction TIFFs and Runpod logs remain outside Git.
