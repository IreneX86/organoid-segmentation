# Fluo-only ER U-Net: proposed training protocol

## Task and limitation

The input at deployment will be `fluo.tif` alone. The target masks in
`outputs/er_coherent_fiber_full_20261028/` are *pseudo-labels*: they were
generated using both the aligned ER image and an ER/Fluo base-mask pipeline.
They are not independent manual ER ground truth. A model trained on them can
learn only image information present in Fluo and will also inherit systematic
errors in the pseudo-labels. In the eight-plane exploratory correlation report,
the mean within-tissue ER–Fluo Pearson r was 0.1265. That does not prove the
task impossible, but it makes fine ER reconstruction from Fluo alone uncertain.

## Data preparation

1. Confirm each input `fluo.tif` and its candidate `ER_mask_line.tif` have the
   same ZYX shape. The mask must be uint8, with background 0 and ER 255.
2. Save an immutable manifest of input/mask SHA256, cell ID, acquisition group,
   original Z indices and imaging profile. Never use ER pixel values as model
   input or as an inference-time preprocessing feature.
3. Split by **whole cell/organoid**, preferably also assess the acquisition
   domain shift between the MCF-7 spheroids and WT organoids. Adjacent Z planes
   from the same stack must never be divided between train and validation.
4. Keep a small set of manually corrected ER ROIs from held-out cells for
   biological evaluation. Without this, agreement with pseudo-labels measures
   teacher imitation, not ER segmentation accuracy.

## First model

- Reuse the existing single-channel 2D U-Net implementation in
  `code/model/unet.py`, starting with a modest base width (e.g. 16 or 32).
  Input is a normalized Fluo plane or patch only; target is the matching
  binary ER pseudo-mask. Do not feed the ER image to the student.
- Train on native-resolution patches with moderate geometric/intensity
  augmentation. Keep the exact normalization rule fixed between training and
  deployment, and use overlapping patches at inference to avoid seams.
- Start with BCE + soft Dice and early stopping on a **held-out whole-cell**
  validation set. Because the mask consists of narrow lines, evaluate a
  distance-tolerant boundary/centerline measure alongside Dice and false
  positives; a one-pixel shift can change ordinary Dice substantially.
- Compare against a trivial/structure-only baseline. Report both held-out
  pseudo-label agreement and manually reviewed ROI performance, separated by
  imaging profile and Z depth. Check whether the Fluo-only model predicts ER
  structures absent from Fluo evidence or merely learns cell outlines.

Only if the student works on held-out *manual* ER references should the model
be used for independent Fluo-only prediction. If it cannot recover fine ER
networks, the scientific limitation is missing information in the input; more
epochs or threshold tuning will not supply it. At that point, collect paired
ER/Fluo examples or use ER-channel inference where available.

## First exploratory run, locked before training

`code/er_fluo_student.py` implements the first Fluo-only student. Its dataset
archive contains the eight `fluo.tif` stacks and the eight completed
`ER_mask_line.tif` stacks, never the ER source channel. It checks SHA256 and
alignment against the completed batch manifests before packaging. The fixed
whole-organoid split has six train stacks, `cell_20260904_23963_NZ_40` for
validation and `cell_20260814_23300_NZ_100` for the untouched test. Both
acquisition domains occur in training. The validation stack selects the best
checkpoint and early stopping; the test stack is read only after that
checkpoint is locked. This is an exploratory *pseudo-label imitation* test,
not evidence of ER accuracy against manual annotation.

The 2D U-Net has one normalized Fluo channel, base width 16, 384-pixel patches,
BCE + soft Dice, batch size four, 48 batches per epoch, at most 40 epochs,
and patience eight. Inference uses 384-pixel overlapping tiles and a fixed
0.5 threshold. The training artifact records the split, input and target
hashes, environment, history, best checkpoint and per-slice validation/test
agreement. No threshold is tuned on the test stack.
