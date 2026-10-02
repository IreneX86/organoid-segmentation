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
