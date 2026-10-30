# Public pretrained model: Fluo-only exploratory pilot

This experiment applies the public nERdy+ checkpoint directly to five full Fluo slices, without ER-channel input, existing masks, fine-tuning, or paid GPU resources. It does not establish that the predicted structures are ER.

## Why this model / limitations

[nERdy+](https://github.com/NanoscopyAI/nERdy) supplies a runnable pretrained ER segmentation model. Its intended domain is ER imaging, not this project's non-ER-specific Fluo channel. The earlier local nERdy experiment used ER-channel crops; this experiment is separate and uses only Fluo.

[Allen's label-free prediction work](https://www.allencell.org/label-free-determination.html) predicts intracellular structures from transmitted-light microscopy. That input is also different from the present Fluo channel; it is not a drop-in validated predictor here.

## Fixed experiment

Five previously designated training specimens: 23298, 23299, 23961, 23965, 23966. One middle slice per specimen; no selection by prediction quality. The validation and test specimens are excluded. No threshold tuning against these five images.

Each 1200×1200 plane is normalized once at its 1st–99.8th percentiles. Inference uses overlapping 384×384 tiles at origins 0,272,544,816 in both axes, with tapered probability blending. This is an adaptation for local memory usage, not an exact reproduction of author whole-image inference. The authors' sigmoid min–max rescaling, rolling-ball subtraction and Otsu threshold produce the candidate mask. Scores are not calibrated ER probabilities, especially after domain transfer.

Output masks are uint8, background 0 and candidate foreground 255. All original source files, teacher masks and historical models remain unchanged. External code remains in its separate AGPL-licensed clone; no external source or weights are vendored into this repository.

Script: `experiments/er_nerdy_fluo_five.py`. Full-resolution inputs, masks, raw model scores, per-image reviews and a five-image overview are under ignored `outputs/er_nerdy_fluo_five_20261029/`. The generated manifest records exact sources, slices and checksums.

These outputs must not be incorporated into training labels without biological validation. No accuracy/Dice against complete manual ground truth is available.

## Observed result

All five full-resolution images completed and passed shape/dtype/value/finite-score checks. Candidate foreground fractions were 5.85%, 4.98%, 5.42%, 3.36%, and 6.15% respectively. These are coverage measurements, not accuracy scores.

Visual review shows extraction of bright contours and fine texture, together with fragmented detections and potentially nonspecific outline/background labeling. In specimen 23965, many fine candidate marks appear across diffuse signal. Without complete reference labels their biological identity cannot be determined. Therefore this direct pretrained transfer is retained for review only, not accepted as a production ER predictor or a label-generation replacement. No original labels were changed and no remote GPU charges were incurred.
