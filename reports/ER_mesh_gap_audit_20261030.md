# ER mesh-window supervision audit — 2026-10-30

## Finding
The existing16 selected training windows contain153 geometrically enclosed small gaps (4–256 pixels),8546 gap pixels. **7548 pixels (88.32%) are ignored**, only998 carry provisional negative supervision. **120/153 gaps have no negative supervision at all.** These are holes in the pseudo-mask, not independently confirmed biological ER mesh openings.

Under the existing masked loss, changing predictions on ignored pixels cannot affect BCE or Dice. Therefore filling those ignored gaps is unpenalized. This directly establishes an objective blind spot, but does not establish the sole cause of the observed broad predictions. Merely cropping more positive-rich windows has not removed this blind spot.

## Cross-channel evidence
For each window, compare raw intensity in the one-pixel positive rim adjacent to each selected gap against the gap interior. The rank-AUC median is0.705 for ER and0.548 for Fluo;12/16 Fluo windows are below0.6. Here AUC measures raw-intensity separability, **not segmentation accuracy**. A value near0.5 does not prove that a neural network cannot use spatial context. No independent annotation, registration, or causal claim follows.

The fixed8-panel review (first2 windows per training specimen) shows substantial differences between the channel appearances: in some windows the ER mesh is visible while Fluo contains bright puncta and different diffuse structures. This supports examining channel correspondence and supervision semantics before more training; it does not prove absence of ER information from Fluo.

## Method and safeguards
Read only the16 already selected training windows, their original320px Fluo contexts, labels and paired ER references. No validation/test files opened. Source SHA256 checked before and after; original masks unchanged. Gap definition uses the already registered4–256px range; no fitting, new labels, threshold optimization, inference or GPU resources. Synthetic/biological ground truth is not inferred from a hole alone.

## Next direction
First build a fixed, reviewable **line-and-gap supervision candidate within clearly supported ROIs**, rather than more positive-only crops. Use ER reference as evidence, keep existing positives unchanged, and retain uncertain/bright gap pixels as ignore. Do not automatically relabel every enclosed hole as background: some may contain missed ER.

Before any new GPU comparison, review those proposed negatives and their Fluo correspondence. A useful small experiment would change only gap supervision, keep the same input context and stable BN configuration, and compare against the matched control. This audit itself does not authorize promotion of such candidate labels to biological truth, nor does it justify another model sweep. Existing failed models remain not production-approved.

## Files
- Script: `experiments/er_mesh_gap_audit.py`
- Full16 window reviews and per-window JSON: `outputs/er_mesh_gap_audit_20261030/`
- Fixed8-window overview: `eight_gap_reviews.png`
- New GPU cost: US$0; no resources created.
