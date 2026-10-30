# ER/Fluo correspondence audit — 2026-10-30

## Conclusion
No consistent small rigid XY or Z offset was found that explains the Fluo-only failure. This is not proof of perfect registration: cross-channel correlation is weak and ambiguous, and the bounded search cannot rule out large offsets or local deformation. No data, labels or transforms were changed. No GPU resources were created.

## Scope and verification
All40original320pxER/Fluo crops matched their source TIFF planes and coordinates exactly, across5specimens. Both channel series shapes match each other and their original source manifests. TIFFs were read only. Synthetic known-shift tests verified the correlation search sign and maximum. No label or prediction was used to select a shift.

XYsearch ±16originalpixels in2pxsteps; Zreference offsets−2…+2. At half resolution, smooth intensity sigma3 and detail DoG sigma1−sigma8 were tested as diagnostic feature representations. Constant Fluo core and ER windows were compared using normalized cross-correlation. Argmax coordinates are diagnostics, not validated transforms.

## Detail-feature results
| Specimen | Split | Median zero-shift NCC | Median XY gain | Median dy/dx (pixels) | Median Z gain |
|---|---|---:|---:|---|---:|
| cell_20260814_23298_NZ_50 | train | 0.2581 | 0.0128 | 2.0/2.0 | 0.0029 |
| cell_20260814_23299_NZ_200 | train | 0.2406 | 0.0000 | 0.0/0.0 | 0.0034 |
| cell_20260904_23961_NZ_100 | train | 0.2086 | 0.0040 | 0.0/0.0 | 0.0032 |
| cell_20260904_23962_NZ_170 | train | 0.2430 | 0.0006 | 0.0/0.0 | 0.0048 |
| cell_20260904_23965_NZ_140 | validation | 0.1916 | 0.0000 | 0.0/0.0 | 0.0017 |

The validation specimen median detail shift was0/0pixels; XYgain0, Zgain0.00169. Zoffset preferences vary among crops; there is no uniform off-by-one pattern. One validation detail estimate hits the search boundary and is not reliable. Smooth-intensity estimates often hit boundaries and disagree with detail estimates, indicating channel-specific brightness is a poor registration cue here. Do not shift labels by these estimates.

## Appearance comparison
Validation normalized mean0.231 lies within the training-specimen median range0.137–0.236; normalized std0.120 is within0.101–0.163. Fine-scale standard deviation0.0217 also lies within training medians0.0196–0.0273. Thus simple global brightness/noise difference is not established as the sole reason. These descriptive values do not exclude local contrast or morphology domain shift.
Prior paired input control showed mean label/background contrast around0.0755inFluo versus0.2622inER on validation. Visual overlays show corresponding cell-scale structures but many ER-rich networks are weak or different in Fluo. The teacher labels are ER-derived, so this difference is expected and not an independent proof of ER specificity.

## Interpretation
Combined with Fluo trainDice0.847/validation0.164 and ERcontrolvalidation0.742, the evidence favors poor cross-specimen learning of the Fluo-to-ER target under this small dataset over a simple fixed threshold or small global offset error. Information ambiguity, limited diverse training examples and local morphology remain hypotheses. It does not prove that Fluo-to-ER is impossible.

## Next recommendation
Before further architecture/loss searches, expand diverse paired crops from the existing source specimens with a fixed sampling rule and no edits based on validation predictions; avoid counting adjacent slices as independent examples. Retain whole-specimen isolation and use grouped cross-validation to test whether the failure is specific to the repeatedly viewed validation specimen. Keep current targets provisional and acknowledge missed ER outside confirmed crop labels. Do not promote any current Fluo model or adjust labels to match it. This audit does not authorize a new production model; no new training has been started.

## Outputs
`outputs/er_alignment_audit_20261030/five_specimen_alignment.png`: fixed first crop of each specimen, columns Fluo / ER / unshifted overlay(ERred, Fluogreen).
`audit.json`:40ROIoffsetdiagnostics, source shape/axes/bit-depth metadata.
`fluo_appearance.json`:40ROIinputstatistics.
Code:`experiments/er_alignment_audit.py`.
