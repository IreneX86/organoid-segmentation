# Fluo-visible ER-like segmentation: revised exploratory task

## Current route — 2026-10-29 user clarification

Use existing accepted ER-mask positives and spatially identical Fluo patches;
do not require recovery of all missing lines or incorporate nERdy+ additions.
The newer [paired-patch record](er_positive_patch_training.md) supersedes the
earlier requirement below for complete manual line correction before dataset
preparation. There are 760 verified training-only patches. Unmarked tissue
remains ignored; negative supervision is still required before meaningful
binary-model training. Historical experiments below remain unchanged.

## Scope and evidence

The requested input is **Fluo only**. The target is the subset of ER-like
filament/network structure that can actually be localized in the Fluo image.
It is **not** reconstruction of every line visible only in the paired ER
channel. The first Fluo-only U-Net was trained on ER-channel-derived labels and
returned empty masks at threshold 0.5; threshold 0.2 on three validation
slices produced Dice 0.144, precision 0.087 and recall 0.433. These are
pseudo-label agreement figures, not manual biological accuracy.

Training-only paired-crop review is saved at
`outputs/er_fluo_student_20261028/fluo_visible_target_preview.png`.
An automatic intersection of old ER labels with Fluo high-pass/ridge evidence
severely fragments networks. On four inspected 384-pixel training crops,
illustrative stricter filtering retained 1601/9575, 1481/5056, 196/9370,
and 737/5877 old-label pixels. These filters are **rejected as final label
generators**; the previews are diagnostic only. They also show that a bright
Fluo ridge does not by itself establish ER identity.

## Annotation semantics for a small development set

Use aligned Fluo and ER crops, but assess *Fluo visibility first*. Record three
states, with separate arrays if needed:

- **Positive:** a spatially coherent, line-like or reticular structure that
  can be traced in Fluo and is supported by the paired ER image. Boundaries
  follow the visible Fluo structure, not an ER-only extension.
- **Negative:** a clearly visible Fluo structure that lacks corresponding ER
  evidence, and unambiguous background. Cell rims, isolated bright puncta,
  and broad blobs are not automatically positive.
- **Uncertain/ignore:** an ER-only line, an indistinct Fluo region, or a place
  where registration/identity is unclear. Do not force these pixels to
  negative: that would teach the model to suppress potentially real structures.

Maintain the original ER TIFFs and old pseudo-label stacks unchanged. Store
any newly reviewed labels in a separate directory with provenance, source
hashes, crop coordinates, Z index, and reviewer/decision notes. Before model
training, render overlays on Fluo and check thin-line continuity, ambiguous
cell boundaries, bright puncta, and both acquisition domains.

## Evaluation and model gate

Use whole-organoid separation; adjacent Z slices or patches from one
organoid must never cross training/validation boundaries. The already viewed
eight-organoid result is exploratory development evidence, not a fresh
independent test of a revised target. If no new independent organoid is
available, describe subsequent results as internal development only.
Choose checkpoint and threshold using training/validation labels only. The
loss must exclude ignore pixels; the current binary-mask training code does
not implement this and must be revised before any new run. Report Dice,
precision, recall, line continuity, and error overlays against the manually
reviewed *Fluo-visible* target, separately by acquisition profile. A new
model is eligible for broader inference only after visibly coherent masks and
nonempty, useful validation predictions; do not infer quality from the old
ER-derived pseudo-label Dice or train longer simply to overcome empty masks.

No GPU retraining is started by this protocol. Existing first-run checkpoint
remains experimental and is not a production Fluo-visible ER-like model.

## Prepared review kit, 2026-10-29

`code/er_fluo_visible_review.py` generated
`outputs/er_fluo_visible_review_20261029/` from the six training organoids
only. It selected two deterministic 384×384 crops per organoid from the
previous training-only tile survey (lowest and median Fluo-intensity AUC among
tiles with 1–12% old pseudo-label foreground). Each crop has raw Fluo, raw ER,
the old pseudo-label, and a **suggested positive, not a label**. The manifest
pins cell, Z, XY, split, source hashes, and crop hashes. Twelve crops from six
training organoids passed shape, binary-value, subset, and SHA256 checks.
The MCF and WT contact sheets are `MCF_review_overview.png` and
`WT_review_overview.png` in that ignored output directory.

The suggestion preserves more line continuity by allowing a two-pixel
proximity to Fluo ridge/contrast evidence, but still overlaps cell outlines
and cannot distinguish ER identity automatically. It should be corrected or
rejected by a domain reviewer. **Do not treat these TIFFs as approved masks or
start another U-Net run from them.** The next substantive gate is a small
three-state reviewed set (positive, negative, ignore) covering both domains,
including hard negatives at bright cell rims. Only then revise the student
loss to ignore uncertain pixels and perform a grouped validation run.

### User correction, 2026-10-29

The user reviewed the red positive suggestions and confirmed **all of them
are target structures**, while also identifying many omitted structures.
This supersedes the prior speculation that the red regions might merely be
ordinary cell rims. The Fluo-support filter has a recall failure and must
not be used to create negative labels or to narrow the training target.
The objective remains `Fluo.tif` as input and the aligned ER mask as target,
with Fluo-only inference on new data. Keep the full existing ER mask as the
current exploratory teacher, inspect its residual omissions, and address the
first student's empty/low-recall output through training/evaluation analysis.
No candidate in this review kit has been promoted to a final annotation.

## Paired ER-positive crop preparation, 2026-10-29

`code/er_positive_paired_crops.py` prepared 18 native-resolution 384×384 crops
from six **training** organoids at Z quartiles. Each output keeps the same
global Z/XY window in ER and Fluo, the existing ER-derived positive seed, an
overlay, source SHA256s, and crop SHA256s. The two locked whole-organoid
validation/test stacks were not opened. The generated manifest is at
`outputs/er_positive_paired_crops_20261029/manifest.json` (ignored local data).
All 18 crop shapes, seed binary values and hashes were checked.

These are paired review material, **not ready-to-train complete masks**.
There are 333,322 marked pixels across the 18 crops, but ER.tif and the old
pseudo-mask both have omissions. A zero in the seed means *unknown*, not
confirmed absence of ER. Training ordinary BCE/Dice against these zeros would
repeat the false-negative label problem. The existing positive seeds also
need confirmation before being treated as biological truth. A future model
must see only Fluo as input; ER is used to establish labels during training.
Training requires either a reviewed three-state target (positive, definite
negative, ignore) with an ignore-aware loss, or an explicitly justified weak-
supervision method. The earlier weighted-BCE candidate was cancelled before
training when the user requested all Runpod billing to stop; it supplied no
new model or validation result.

## Literature check and missing-label decision, 2026-10-29

The [nERdy study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12589508/) used
manually traced ER centerlines, then dilated those lines into training masks.
Its [authors' repository](https://github.com/NanoscopyAI/nERdy) offers ER
segmentation code and a pretrained model. This is a useful *annotation
representation* for thin networks, but its input is an ER-labelled channel;
it does not establish performance for Fluo-only inference here. The
[Allen Cell & Structure Segmenter](https://www.allencell.org/segmenter.html)
likewise distinguishes algorithmic candidates from curated ground truth.
[ERNet-v2](https://github.com/charlesnchr/ERnet-v2) is another ER-channel
segmentation reference, not a registered label set for these stacks.

Our existing Sato/Meijering/Z-support pilots in
`reports/ER_line_detail_research_20261028.md` recovered some plausible
lines but also background texture and did not reliably fill missing internal
networks. Another unsupervised ridge-threshold sweep is therefore not a
justified replacement for label review. Public ER masks cannot be copied onto
these TIFFs: they are not pixel-aligned and may use a different target width
or biological definition.

For the 18 paired training-only crops, preserve the old mask as a positive
proposal. Correct missing clear ER-like centerlines against registered ER and
Fluo plus adjacent Z; delineate definite negatives and leave ambiguous areas
as ignore. Trace centerlines before choosing a fixed, image-resolution-aware
line width, so broad rings do not substitute for network detail. Record
correction provenance and evaluate missing-line recall, false positives and
connectivity in addition to pixel Dice. Keep entire organoids isolated across
training and validation. No pseudo-label or external prediction is promoted
to ground truth merely because it contains more positive pixels.

### Paired-channel missing-line pilot

`experiments/er_missing_line_candidates.py` created a separate review-only
overlay for all 18 training crops at
`outputs/er_missing_line_candidates_20261029/all_18_review.png`. Yellow is the
old candidate positive and cyan is a proposed addition supported by ER,
Fluo, and at least one adjacent Z slice. Candidate TIFFs are binary 0/255,
384×384, and SHA256-checked against the manifest. They contain 38,256 new
candidate pixels across the 18 crops; this count is **not** recovered true ER.
Visual inspection shows cyan also includes isolated speckles and fragments,
and the method still misses broad regions of visible network. It is rejected
as an automatic training-label generator. Preserve all source images and old
masks; use the overlay only to prioritize correction of missing centerlines.

### Fluo detectability audit before label correction

`experiments/er_fluo_detectability_audit.py` tests the 18 locked training
crops only. It skeletonizes the existing ER-positive seed and compares a
Meijering-ridge × local-contrast score on that skeleton with nearby copies
shifted 16 pixels in XY, excluding a 5-pixel neighborhood of the original
seed. Output: `outputs/er_fluo_detectability_20261029/summary.json`,
`per_crop.csv`, and `aligned_signal_auc.png`. The median per-crop AUC is
**0.939 for ER** and **0.685 for Fluo**; the six per-organoid median Fluo AUCs
range from 0.618 to 0.694. Thus Fluo contains some aligned structural signal
at existing positives, but substantially weaker and variable compared with
the direct ER channel. This does not measure ER segmentation accuracy or
prove that omitted ER lines are predictable from Fluo. Shifted controls may
themselves contain ER, and the seed has omissions; pixels are not independent
biological samples.

The next label-correction gate is to annotate a small set of *missing* lines
and definite non-ER regions in registered ER/Fluo/Z views. Prioritize the
weaker-Fluo organoid `cell_20260904_23965_NZ_140` as well as a stronger
representative, without using validation/test images to define the target or
choose parameters. Keep invisible or equivocal lines as ignore. Do not run
another U-Net on the 18 seed masks alone or treat this AUC as a model score.

`code/er_tristate_review_workspace.py` has created an immutable starter at
`outputs/er_tristate_review_20261029/` for all 18 training crops. Each draft
is uint8 with `1` for old **candidate** positives and `255` for all other
unknown pixels; it asserts no negative pixels. The separate cyan additions
remain review proposals. `manifest.json` records crop identity, source
hashes, draft hashes and `training_ready=false`. It is deliberately not a
reviewed label set. A corrected copy must add missing lines and confidently
non-ER regions, keep ambiguous sites at 255, and have review provenance
before any partial-label training.

### External ER model check (training crops only)

To test whether a published model could supply missing labels, the public
[nERdy+](https://github.com/NanoscopyAI/nERdy) pretrained ER-channel model was
run on three training-only crops by
`experiments/er_nerdy_reference_pilot.py`. The external code and weights were
kept outside this repository; its commit, weight SHA256 and the exploratory
1–99.8-percentile input conversion are recorded in
`outputs/er_nerdy_reference_pilot_20261029/manifest.json`. The authors'
rolling-ball/Otsu postprocessing was retained. The four-column review image
is `three_train_crops_review.png` (yellow old candidate, cyan nERdy-only).
On visual inspection the model densely marks tissue texture and even low-
signal background as a fine network in two of three crops, while also missing
some old candidate lines. Thus it is **rejected as an automatic label source**
for this imaging domain. No claim about nERdy's performance on its own
published data is implied. It was not run on Fluo as a deployment model, and
no result was incorporated into training labels.

The training-only three-state drafts can now be opened with
`code/er_tristate_review_napari.py`. It displays registered Fluo and ER
channels, old positives and automatic additions as separate proposal layers,
and an editable layer whose on-screen codes are `0=unknown`, `1=ER`,
`2=definite non-ER`. Pressing `S` writes a new timestamped *draft*; it never
overwrites the initial source or marks a draft training-ready. This tool was
dry-run against the weaker-Fluo `23965` crop. The biological review itself
has not been completed or asserted by the script.
