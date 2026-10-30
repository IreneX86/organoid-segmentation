# Fluo-only ER partial-label U-Net pilot — 2026-10-29

## Outcome

The GPU run completed normally, but the model **failed the exploratory quality
check** and is not approved for production. No further training was started.

Six whole training samples supplied 1,520 positive-/negative-rich patches;
one separate validation sample supplied 80 patches. Test data stayed unopened.
Existing ER positives were retained, automatic exterior background was used as
weak negative supervision, and other tissue was ignored by masked BCE+Dice.
Fluo alone was supplied to the U-Net. Original data and historical models remain
unchanged. Fixed protocol: `docs/er_partial_student_protocol.md`.

## Actual run

- One Secure Cloud A40, Pod `[pod-id-redacted]`; no persistent volume or endpoint.
- Python3.11.18, TensorFlow2.21.0, NumPy2.4.6, tifffile2026.3.3.
- cuInit returned0; one GPU and actual TensorFlow GPU matrix operation passed.
- Maximum20 epochs; EarlyStopping patience5. Completed6, best checkpoint epoch1.
- Reported epoch durations:13s then five approximately5s epochs (~38s training,
  excluding preparation, compilation outside epoch timing, verification and evaluation).
- Exit0 with `ER_PARTIAL_TRAINING_COMPLETE` and `ER_PARTIAL_RUN_COMPLETE`.

| Validation metric, threshold0.5 | Value |
| --- | ---: |
| Known-pixel Dice | 0.030353 |
| Known-pixel precision | 0.113323 |
| Known-pixel recall | 0.017523 |
| TP / FP / FN | 7,510 / 58,761 / 421,063 |
| Mean predicted-positive fraction in ignored pixels, per patch | 0.038177 |

These are agreement metrics on **incomplete pseudo-labels and overlapping sampled
patches**, not biological ER segmentation accuracy or full-image independent Dice.

Training loss fell from0.1645 to0.0361. Validation loss was best at epoch1
(1.0174), then fluctuated from1.0633 to3.9955, ending1.7421. Fixed preview crops
show missed positive structures and strong patch-edge responses, not a recovered
ER network. Low training loss alone therefore does not establish useful learning.

## Interpretation

The current positive-plus-exterior-negative objective did not generalize to this
validation sample. Exterior negatives do not constrain intracellular non-ER,
and sampling can permit shortcut features. The large train/evaluation gap also
warrants checking BatchNormalization running statistics and preprocessing at
inference. These are hypotheses, not established causes. Do not conclude that
Fluo-to-ER prediction is impossible from this single run, and do not tune thresholds
on the test sample to hide the failure. Before another paid run, inspect the
saved model's normalization/inference behavior and sampling specificity locally.

## Technical issue and checks

The first launch failed *before training*: a reused GPU preflight passed GPU
matmul but then expected36 Mito TIFF pairs. The wrong dataset check was replaced
with the existing ER GPU preflight, and the same Pod ran once into a new retry1
directory. Architecture, labels, split, optimizer and other science were unchanged.
The first failure archive was retained. There was only one actual training run.

TensorFlow reported a nonfatal layout-optimizer rewrite failure at a Dropout node;
training continued with finite loss, cuDNN loaded, and exit0. This warning does not
establish the cause of poor predictions.

The result archive SHA256 matched; the model's SHA256 matched its result record;
its ZIP integrity, local model load and finite forward computation passed.

Checkpoint SHA256:
`01e39a7f3a4ce3f13705f9e2559136a823813a40aa0bd2eba5791e1e31d9abf0`

Local results:
`outputs/er_partial_run_20261029/retry1/extracted/results/student/`

- `er_partial_student.keras`
- `training_history.csv`, `config.json`, `environment.json`
- `validation_partial_metrics.json`, `result.json`, `review_summary.json`
- `validation_previews/`, `validation_review.png`

## Cost and cleanup

The Pod was stopped and terminated only after the complete result download was
verified. Account UI then showed GPUs0, vCPUs0, storage0GB, endpoints0 and
US$0.00/hour. Balance changed fromUS$5.34 toUS$5.25: approximatelyUS$0.09,
based on rounded displayed balances rather than a final itemized invoice.
The cost-guard automation was paused. Cleanup evidence is local at
`outputs/er_partial_run_20261029/cleanup_zero_spend.png`.

No source TIFFs, checkpoint, raw logs, credentials or large artifacts are intended
for Git. This report preserves a failed research result; it does not promote a model.
