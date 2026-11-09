# v7: longer training + intensity augmentation + 8x TTA + soft teacher targets (2026-11-09)
Same as v3 C7 (Fluo z-3..z+3, base-32 U-Net, 3 folds by spheroid) plus: 20000 steps (4x) with cosine LR, intensity
augmentation (scale/offset/gamma/blur/noise), 8-fold test-time augmentation, and the teacher's PROBABILITY map as target
(knowledge distillation; teacher re-run on the pod reproduced the saved pseudo-labels, Dice 0.9994).
RTX 4090, ~2 h incl. slow data download, about $1.8. Labels are pseudo-labels; manual = 14 cleaned manual masks.

Same planes for both models (36 held-out planes z071/z103 + 14 manual planes), default threshold 0.5 for v7:
| | v3 (best thr) | v7 (thr 0.5) |
|---|---:|---:|
| Dice within 40 px of Golgi, teacher labels | 0.380 | 0.422 |
| Dice within 40 px of Golgi, manual masks | 0.386 | 0.412 |
| Dice inside stained cells, teacher labels | 0.239 | 0.273 (0.292 at thr 0.6) |
| Dice inside stained cells, manual masks | 0.290 | 0.310 (0.330 at thr 0.6) |
| F1 with 6 px tolerance, teacher / manual | 0.599 / 0.634 | 0.638 / 0.667 |
v7 is the first model to clearly beat v3 (+0.03 to +0.04 plain Dice). Separate contributions of the four changes were not
measured (single combined run).
