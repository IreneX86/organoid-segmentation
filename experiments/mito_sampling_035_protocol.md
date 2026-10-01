# Mito sampling experiment 01 — locked protocol

Status: **five-fold validation-only run complete; candidate rejected by the
locked validation gate**. This is an exploratory experiment on the 36
labelled development images. The unchanged five-fold baseline is frozen at
commit `b6f2e87`; no baseline checkpoint, split, metric, or tag is overwritten.

## Hypothesis and one changed factor

The baseline samples a foreground-centred 512×512 patch with probability
**0.60**. A lower probability **0.35** may expose the model to more ordinary
background/context patches and reduce false positives on sparse Mito images.
This hypothesis was motivated after inspecting baseline OOF results, so any
gain on the same 36 images is exploratory, not a fresh unbiased test result.
Architecture, loss, augmentation, normalization, patch size, optimizer,
learning rate, batch size, steps, epoch cap, seed, validation, sliding-window
inference, and threshold-selection rule remain unchanged.

## Five-fold design and selection gate

For each outer fold 0–4, reuse the exact `fold_ids` in
`baseline/mito_five_fold_20261001/fold_0/split.json`. The outer test fold is
never opened during candidate training or validation. The next fold is the
inner validation set; the remaining three folds are training images. Candidate
models use `--mito-positive-patch-probability 0.35`,
`--expected-mito-fold-map`, `--defer-mito-outer-test`, and a fresh output root.

After **all five validation-only candidate runs**, run
`code/mito_candidate_gate.py --variant sampling_035` to compare candidate and
baseline on each fold's *same validation set* using the maximum recorded
`val_full_dice` at threshold 0.50. Select 0.35 for that outer fold only if its
best validation Dice exceeds the baseline best by at least **0.01**; otherwise
retain baseline 0.60 (including ties and incomplete candidate runs). This
per-fold rule is fixed before any candidate outer-test metric is calculated.
No fold's outer-test label may choose its sampling policy, checkpoint, epoch,
threshold, or whether to rerun. No decision may use the mean of all folds'
validation scores, because those images are outer-test images in other folds.

Only after recording all five decisions: evaluate the selected candidate
checkpoint on each fold where 0.35 won the validation gate, and reuse the
existing frozen baseline outer-test result on all other folds. Each selected
candidate uses its **own** validation-selected threshold. Report the 36-image
OOF Dice, IoU, precision, recall, sample SD, lower quartile, minimum, fold
means, paired per-image differences from baseline, and performance by
foreground fraction. The threshold-0.50 comparison remains offline on the
same checkpoint and cannot trigger retraining. A small mean gain with a worse
lower tail or substantially larger SD is not a reliable improvement.

This estimates the predeclared *per-fold validation selection procedure*;
it is not the score of one final 0.35 model. Production method selection and
external generalization will still require judgment and later independent
labelled images. Do not use low-scoring outer-test images to tune 0.35, the
0.01 margin, or a second factor after this experiment.

## Execution and safety

The user authorized this GPU experiment. Use the previously verified Runpod
Secure Cloud A40 environment. Run one Pod and
one fold at a time, with CUDA driver and TensorFlow GPU preflight before each
training run. Estimate the full cost from current live price and cap it before
launch. Save each fold to a new versioned output directory; first download
and verify checkpoints, histories, thresholds, metrics, predictions, and logs,
then terminate the Pod. Do not modify Cetus job 105331.

The five-fold candidate training completed on 2026-10-02 Sydney time. All
five validation Dice gains were below the fixed +0.01 gate, so no candidate
checkpoint was evaluated on outer test. The locked decisions and unchanged
baseline OOF comparison are in `experiments/results/`. The downloaded models
and GPU logs remain in ignored local `runpod_results/`; the Pod was deleted.
No TIFF, mask, checkpoint, prediction, secret, or log belongs in Git.
