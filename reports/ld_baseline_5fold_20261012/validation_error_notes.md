# LD baseline validation error notes

These descriptive diagnostics use saved **inner-validation** probabilities and
frozen derived labels for 36 images, each appearing in one fold's validation
set. The diagnostic used that fold's validation-selected threshold. Counts and
distances below are in native image pixels; the source is the local, ignored
`outputs/ld_validation_error_analysis_20261012/validation_error_summary.json`.

- Of 22,773 false-positive pixels, 14,235 (62.5%) were within 2 pixels of
  labelled foreground, while 6,764 (29.7%) were more than 10 pixels from it.
  Proximity alone does not establish whether a pixel is a boundary error or a
  biologically meaningful false detection.
- Of 25,315 false-negative pixels, 22,124 (87.4%) were within 2 pixels of the
  labelled foreground boundary. Separately, 9,529 (37.6%) were more than 10
  pixels from any predicted foreground. These distances have different
  reference masks, so the two false-negative groups can overlap.
- There were 150 eight-connected labelled components of at most 25 pixels.
  Predictions touched 99 (66.0%) at least once and covered at least half of
  93 (62.0%); 51 received no predicted pixel. Connected regions in a semantic
  mask are **proxies**, not verified individual droplets.

At the fixed cutoff of at most 0.10% labelled foreground, the same eight images
had mean Dice 0.7306 in the [out-of-fold report](README.md) and 0.7982 in this
validation diagnostic. The other 28 images had means of 0.7731 out of fold and
0.7738 on validation. These are different fold-model predictions, and the
validation threshold was selected on the validation images. The contrast is
descriptive, not a generalization-gap estimate or a basis for changing the
model, label rule, threshold, or postprocessing.

Each image serves as validation in one rotation and outer test in another, so
the all-fold validation analysis and OOF report are not independent evidence
for selecting a later candidate. Annotation-edge uncertainty, unidentified
related views, and the lack of independent external LD labels also limit
interpretation. The local diagnostic montage contains image pixels and remains
outside Git; this note contains only aggregate measurements.
