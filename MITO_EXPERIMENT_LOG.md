# Mito experiment log

## 2026-10-01 — locked baseline and read-only data audit

The immutable fold-0 baseline is Git tag `baseline-mito-fold0-20261001`
(`df516e1`). Its 7 internal held-out images reached mean Dice 0.869262,
sample SD 0.085012, minimum Dice 0.726555, mean IoU 0.776824,
precision 0.844737, and recall 0.905731. These are **not** five-fold CV
metrics and must not be described as external generalization.

Read-only audit of the 36 paired Mito TIFFs in the existing local transfer
archive found 36 images and 36 matching masks. All images are 1200×1200
uint16; all masks can be interpreted as binary and contain foreground.
Foreground fractions range from 0.006063 to 0.134049 (median 0.066151).
There are no exact duplicate input pixel arrays. A coarse 60×60-pixel
thumbnail correlation screen found no pair above 0.98; this does **not**
prove that related images from one organoid are absent. Original organoid/Z
identities are unavailable and will not be reconstructed.

The baseline already uses positive-centred patch sampling with probability
0.6, flips/90° rotations, mild intensity augmentation, BCE+soft-Dice loss,
ReduceLROnPlateau, full-image validation, and validation-only threshold
selection. Adding these under new names is not an experiment. The current
five-fold assignment is stratified using each labelled mask's foreground
fraction, then frozen for fair comparison. This label-assisted split design
must be disclosed when reporting generalization.

**Missing baseline evidence:** only fold 0 has been trained. `code/mito_cv_report.py`
now refuses to report a five-fold score until all five outer-test reports
exist, their split maps match, train/validation/test sets are disjoint, and
each of the 36 images appears exactly once as outer test. No new model has
been trained or selected by this audit.

## Pre-registered first comparison — threshold policy

**Hypothesis:** A fixed semantic threshold of 0.50 may be at least as stable
as selecting the maximum validation Dice among 17 thresholds on only seven
validation images. This changes one inference decision, not architecture,
loss, augmentation, masks, or training. The comparison will use predictions
from the *same* five baseline-trained models; for each outer fold the 0.50
threshold is fixed in advance and the baseline threshold is selected using
that fold's validation images only. Neither policy may inspect its outer-test
labels for that fold before predictions and thresholds are locked. Both
policies must be reported across all 36 out-of-fold images, including mean,
SD, minimum, lower quartile, IoU, precision, and recall. A tiny mean gain with
worse spread or lower tail is not a reliable improvement. No threshold policy
will be chosen from a single fold's held-out results.

The fold-0 held-out metrics were previously inspected. Any method claim from
this internal CV remains exploratory; later independently labelled unseen
data would be needed for external confirmation.

## Next execution gate

Completing the unchanged baseline requires four additional full GPU folds.
The previous A40 fold took about 26 minutes of training/evaluation and about
US$0.26 in total Pod charges; four folds plus setup are roughly US$1.1–1.5,
with uncertainty from queue, transfer, and runtime variation. This is a
substantial GPU experiment, so it has **not** been started. Do not run a
candidate model, tune from outer-fold scores, or deploy a new Pod before the
baseline CV and its resource plan are reviewed.

## 2026-10-01 — five-fold baseline authorized; GPU stock blocker

The user authorized fold 1–4 unchanged baseline training with an additional
Runpod budget cap of US$5.00. No fold has yet started. A live Runpod read
showed Secure Cloud A40 at US$0.49/hour, but two creation requests returned
`There are no longer any instances available with the requested specifications`.
The second request was constrained to CUDA 12.8 after a capacity read; it also
failed. A subsequent capacity read showed CUDA 12.8 `Out` and CUDA 13.2 `Low`.
`list_pods` returned an empty list, so no GPU charges were started by this
stage. Do not create a different GPU class without a revised, validated
environment and budget check.

`code/mito_export_predictions.py` was prepared for post-training inference at
the validation-selected threshold and the pre-registered fixed 0.50 threshold
using the *same checkpoint*. The five-fold reporter now summarizes both
policies, all 36 per-image OOF rows, fold means, distribution statistics, and
foreground-fraction correlation. It still refuses incomplete or overlapping
folds. The prior fold-0 checkpoint must be exported in the same way before a
threshold-policy comparison is reported.

## 2026-10-01 — authorized five-fold baseline completed

One Secure Cloud A40 Pod (`v84qbcjw5r6fld`, US$0.49/hour) became available
after a fresh CUDA-13.2 capacity read. Its container reused the previously
successful `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image and
Python 3.11 / TensorFlow 2.21.0 / NumPy 2.4.6 / tifffile 2026.3.3 setup.
The pod exposed one A40; `cuInit(0)=0`, TensorFlow saw one physical and one
logical GPU, and a matrix multiplication executed on GPU. A fresh preflight
passed before **each** unchanged baseline fold 1–4. Fold 0 was copied from
the frozen local checkpoint and used only for inference; it was not retrained.

One technical packaging issue was corrected on the temporary Pod: the Mac
archive contained 268-byte AppleDouble `._mask_*.tif` metadata sidecars.
They were moved to a separate directory; all 36 genuine input TIFFs and 36
manual masks stayed untouched. No scientific training settings changed.

All four new folds exited zero and produced model, split, history, selected
threshold, per-image metrics and two sets of full-resolution masks. All five
checkpoint archives were downloaded to the ignored local `runpod_results/`
directory, SHA256-verified, and passed Keras ZIP checks. The 36 × 2 masks
were checked for filename, 1200×1200 shape and uint8 0/255 values. The Pod
was then terminated; Runpod listed no Pods, Endpoints or Network Volumes.
Thus no resource remains accruing spend. The final invoice amount may lag in
Runpod billing; the run time at US$0.49/hour implies roughly US$0.65–0.70,
well under the US$5 cap.

The complete small-file CV record is in `baseline/mito_five_fold_20261001/`.
The locked five-fold OOF Dice is 0.8543 ± 0.0628 across 36 images, with
median 0.8756 and range 0.7266–0.9296. This is a baseline, not a model
improvement. The pre-registered fixed-0.50 threshold comparison changed
mean Dice by only −0.0004 relative to validation-selected thresholds.
No optimization training followed.

## 2026-10-01 — sampling candidate prepared, no GPU experiment started

The first Mito-only candidate is pre-registered in
`experiments/mito_sampling_035_protocol.md`: reduce foreground-centred patch
sampling from 0.60 to 0.35 while keeping the original model, loss,
augmentation, optimizer, training budget, and inference unchanged. This is a
single-factor, exploratory comparison motivated after seeing baseline OOF
results. It is **not** an independently confirmed improvement.

`code/train.py` now accepts the Mito-only sampling parameter with an unchanged
0.60 default. For candidate runs it can consume the exact frozen fold map,
refuse existing output directories, and defer all outer-test evaluation. In
that mode it validates only training/validation TIFF pairs; no outer-test mask
is opened during training or calibration. `code/mito_candidate_gate.py` fixes a
conservative per-fold choice using only the two models' validation histories:
0.35 is selected only if best full-image validation Dice improves by at least
0.01, otherwise the frozen 0.60 baseline is retained. Candidate outer-test
predictions can only be exported after this gate is saved. Tests check frozen
split membership and rejection of prematurely scored candidate runs.

No candidate model has been trained, no Runpod resource has been created, and
no new Dice result exists. The next step is a five-fold GPU candidate run,
which requires a separate execution/budget gate under the user's earlier
instruction to stop before large GPU experiments.

## 2026-10-01 — second independent loss candidate prepared

At the user's request for multiple factors, a **separate** Mito loss ablation
was specified in `experiments/mito_tversky_06_protocol.md`. It replaces only
the overlap term of BCE+Dice with mild precision-aware Tversky (FP weight 0.6,
FN weight 0.4); foreground-centred sampling remains at baseline 0.60. The
training entry point rejects attempts to combine this loss candidate with the
0.35 sampling candidate. Both candidate types use the same per-fold validation
gate and defer outer-test scoring until their decisions are fixed. This second
candidate has not yet been trained or evaluated.

## 2026-10-02 — sampling five-fold result; Tversky pilot timed out

The authorized Mito-only sampling experiment ran on one Secure Cloud A40 Pod
(`hn5syvugzi6vwk`) at US$0.49/hour, using Python 3.11, TensorFlow 2.21.0,
NumPy 2.4.6 and tifffile 2026.3.3. CUDA driver and TensorFlow GPU preflight
passed before every fold. The unchanged frozen five-fold split was used;
candidate training and threshold calibration did not open outer-test labels.
All five 0.35-sampling candidate folds completed. Before outer-test access,
`code/mito_candidate_gate.py` locked the pre-registered +0.01 validation-Dice
rule. Candidate minus baseline best validation Dice by fold was approximately
+0.00045, +0.00013, −0.00034, +0.00151 and +0.00636. **No fold passed the
gate.** Accordingly, no candidate outer-test mask was scored; the selected
36-image OOF method remained the frozen baseline with Dice 0.854309 ±
0.062776, IoU 0.750677, precision 0.839055 and recall 0.877258. These are
the baseline's metrics, not evidence that the raw 0.35 model achieved them.
The locked decisions and comparison are tracked in `experiments/results/`.

A separate BCE+Tversky pilot started on the same Pod only after the sampling
run ended. Fold 0 passed GPU preflight and wrote 77 validation-history rows,
but hit the pilot's 25-minute timeout (exit 124) before normal completion;
fold 1 did not start. Its interim best validation Dice was 0.863576 versus
0.865050 for baseline fold 0. This is **incomplete diagnostic data**, not a
valid Tversky CV result, and no outer-test label was read. The incomplete
checkpoint, history and logs are SHA256-verified in ignored local
`runpod_results/mito_ablation_20261001/`.

All five completed sampling checkpoints, histories and logs were downloaded
and SHA256-checked before Pod deletion. Runpod `list_pods` then returned an
empty list. Final pod-specific billing for this session was **US$1.373148**
(GPU US$1.352199, disk US$0.020949), below the US$2 guard. Cetus was not
changed. The result does not support replacing the frozen baseline with
0.35 sampling; further optimization needs a separately budgeted, complete
five-fold comparison, ideally one factor at a time.

## 2026-10-02 — complete BCE+Tversky five-fold comparison

After the incomplete pilot, the user approved a cumulative Runpod ceiling of
US$3.5. A new Secure Cloud A40 Pod (`a83ubzuyu2ypi8`, US$0.49/hour) reused
the successful Python 3.11 / TensorFlow 2.21.0 / NumPy 2.4.6 / tifffile
2026.3.3 setup and the unchanged 36-pair frozen Mito split. A first runner
launch stopped before training because its preflight path was wrong. Only
that path was corrected; a fresh output directory was used for the retry.

All five retry folds passed driver/TensorFlow GPU preflight and exited 0.
Their recorded epoch counts were 77, 52, 55, 41 and 55; all validation Dice
values were finite. The candidate's best validation Dice minus baseline's
best was −0.001800, −0.004255, +0.000316, +0.000272 and +0.003454. The
pre-registered +0.01 rule selected **zero of five** Tversky folds, before any
candidate outer-test evaluation. The selected 36-image OOF report therefore
reuses the frozen baseline: Dice **0.854309 ± 0.062776**, IoU 0.750677,
precision 0.839055 and recall 0.877258. This is not a measured outer-test
score for the raw Tversky candidate. It provides no evidence to replace the
baseline loss.

Each fold's checkpoint, split, history, threshold and GPU logs were downloaded
as a separate archive and SHA256-checked on Mac; final decisions and result
metadata were checked separately. The Pod was then terminated and `list_pods`
returned empty. The final Runpod invoice was still catching up at the time of
this record; at deletion the Pod had run about 1.92 hours at US$0.49/hour,
so this run is estimated at about US$0.96 including disk and the two-session
cumulative cost about US$2.33, under the US$3.5 ceiling. No TIFF, mask,
baseline checkpoint, Cetus job or future unseen data was changed.

## 2026-10-02 — second-round thin-structure and inference-context ablations

The research record covering the frozen baseline and all four completed
one-factor comparisons is indexed at
`reports/mito_optimization_round2_20261002.md`.

The frozen Mito five-fold baseline remained unchanged at OOF Dice **0.854309
± 0.062776** (36 images). A read-only audit of the labelled development pool
and frozen OOF predictions found seven images with ≤5% foreground: mean Dice
0.791388 and mean precision about 0.727, compared with Dice about 0.870 and
precision about 0.866 on the other 29. Recall was about 0.877 in both groups.
The background false-positive rate was similar (~1.1%) across the two groups,
indicating that the lower sparse-image precision is substantially a base-rate
effect. Roughly 74% of false-positive pixels were connected to predictions
that overlapped true foreground, weakening the case for isolated-component
removal. These are descriptive analyses of previously seen internal OOF data,
not a new independent test set.

Two separate, preregistered, one-factor candidates were assessed on one
Secure Cloud A40 Pod (`y8ifdpcq0r06oe`) with the original split and original
train/validation/test isolation. The environment was Python 3.11.16,
TensorFlow 2.21.0, NumPy 2.4.6 and tifffile 2026.3.3. Driver initialization,
one visible TensorFlow GPU and a GPU matrix multiplication passed before
experiment execution. Original TIFFs and manually labelled masks were read
only. Checkpoints/logs were archived outside Git and SHA256-verified locally
before the Pod was terminated; `list_pods` returned empty.

**Three-pool U-Net.** To retain finer structures and reduce capacity, this
candidate removed only the fourth pooling/deepest block and its decoder
stage (roughly 7.85M → 1.95M parameters). All five candidate folds trained
to completion with 98, 94, 54, 96 and 55 epochs. The prespecified
validation-only gate selected fold 3 only. All decisions were saved before
candidate outer-test evaluation. The other four folds reuse frozen baseline
OOF rows, so their *raw candidate* test Dice remains deliberately unknown.
The selected-policy OOF Dice was **0.854928 ± 0.062555**; paired mean change
**+0.000619**, IoU 0.751588, precision 0.837998 and recall 0.879598.
Low-foreground mean Dice increased only **+0.000522** to 0.791909, and the
worst image stayed at 0.726555. Six selected-policy images improved, one
worsened and 29 remained baseline by design. This is a one-fold effect, not
a stable five-fold improvement; the three-pool option is **not promoted**.

**Inference overlap 128 → 256.** This separate trial reused all five frozen
baseline checkpoints and their validation-selected thresholds; no model was
retrained. Validation mean Dice deltas by fold were approximately −0.000040,
+0.000041, −0.000037, +0.000068 and +0.000019. The same validation-only gate
selected zero folds before any candidate outer-test scoring. Its
selected-policy OOF report therefore equals the baseline by construction;
this does **not** measure raw overlap-256 outer-test performance. The overlap
change is discarded as negligible.

The frozen baseline remains the current best-supported pipeline. Across the
new explorations, no result justifies changing normalization, architecture,
inference overlap or the production recipe. The small sample and repeated
researcher inspection of internal OOF results make exploratory gains
particularly vulnerable to selection bias; independent newly labelled data
would be necessary for a stronger generalization claim. The next targeted
experiment should be motivated by an annotation-blind precision error audit
or a training-fold-only illumination/texture analysis, with its method and
validation gate fixed before any new outer-test scoring. Do not scan many
kernel sizes, sampling probabilities, Tversky weights or overlaps.

## 2026-10-02 — frozen-OOF error analysis; one next candidate designed only

See `reports/mito_error_analysis_20261002.md` for the 36-image read-only
measurements, derived figures and eight representative case panels. Archived
OOF masks reproduced the versioned per-image Dice; CPU probabilities differed
from archived thresholded masks at only 2 of 51,840,000 pixels. Biological
six-panel images remain in ignored local `outputs/`, not Git.

Seven ≤5%-foreground images had approximately as many FP pixels per image as
the other 29 (~14,968 vs ~14,801), but fewer TP (~39,268 vs ~102,284). Their
mean FP fraction >10 px from GT was 27.3% versus 16.0%; 84.1% of their FP
exceeded the same-image TN local-contrast p90 and 58.5% had model probability
≥0.9. These observations motivate an **exploratory** bright-background
hard-negative sampling test; they do not prove FP biological identity or
annotation inconsistency. The fixed one-factor plan is
`experiments/mito_hard_negative_protocol.md`. No sampler implementation,
training, Pod, TIFF/mask edit or Cetus change occurred during this audit.

## 2026-10-02 — bright-background hard-negative sampling, five folds

The fixed protocol was implemented as one Mito-only sampling change: 0.60
foreground-centred, 0.20 unchanged uniform, and 0.20 training-fold-only
bright-background hard-negative centres. Frozen splits, U-Net, loss,
normalization, augmentation, patch size, optimizer, validation threshold and
overlap were unchanged. All five folds completed (76, 82, 48, 57, 46 epochs);
each training image had eligible hard-negative centres.

The locked validation-only gate selected **0/5** folds. Fold mean validation
Dice changes were −0.002276, +0.000833, −0.000682, +0.000211 and +0.000383;
fold 4's low-foreground validation subset lost 0.023823 Dice. No candidate
outer-test label was read. The selected-policy 36-image OOF Dice therefore
remained the frozen baseline **0.854309 ± 0.062776**; paired OOF change is
zero by the predeclared fallback rule, not a measurement of raw candidate test
performance. A post-decision validation-only FP audit found, for the seven
low-foreground images, average FP −370 and distant FP −55 pixels per image,
but recall −0.005463 and Dice −0.002682; across all 36 validation images,
average distant FP increased by 115 pixels. The method is **rejected** and its
threshold/probability will not be scanned.

One Secure Cloud A40 Pod was used. The first preflight stopped before training
because macOS AppleDouble `._*.tif` metadata doubled the apparent file count.
A clean copy excluding only those generated metadata files was used for a
same-science technical retry on the same Pod. GPU preflight passed; all five
SHA256-verified model/results archives were downloaded locally before Pod
deletion. `list_pods` was empty after cleanup. The detailed research record,
validation and OOF FP tables, and billing are in
`reports/mito_hard_negative_5fold_20261002.md` and `experiments/results/`.
