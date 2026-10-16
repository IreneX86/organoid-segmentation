# Nuc low-local-contrast candidate: frozen one-factor protocol

**Status:** predeclared before candidate training or comparison with candidate predictions. This is a same-batch development experiment using only the existing 167 manually labelled Nuc pairs. It is not an independent estimate of performance on `organoid_data`.

## Evidence motivating this factor

The existing Nuc baseline and the spatial-illumination candidate leave many approximate expert-circled locations in `cell_22375_NZ_100/fluo_001.tif` uncovered. Rerun probabilities at several of these locations are approximately 0.001–0.013, well below either selected cutoff (0.45/0.50). Among known baseline validation false-negative pixels, 86.5% have probability at least 0.10 below 0.45; only 6.8% lie within 0.05 of the cutoff. A small threshold adjustment is therefore unlikely to recover most such pixels without adding unrelated foreground. The circles indicate rough positive locations, **not** complete target masks, so they cannot yield target accuracy, recall, or false-negative rates.

After the unchanged per-image 1–99.8 percentile normalization, the median 90th-percentile image intensity is 0.582 in the 133 training inputs, 0.577 in the 34 validation inputs, and 0.483 in the first ten target images. More specifically, the 90th percentile of absolute sigma-2 high-pass residuals on mid-intensity pixels is 0.0250 / 0.0251 / 0.0133 in train / validation / target. The approximate expert-circled structures also appear to have weaker dark-core versus surrounding-ring contrast than the training-labelled nuclei. Circle dimensions overlap the training-mask object-size distribution, so size alone does not explain the omissions. The target has no stronger smooth illumination gradient than the training set, consistent with the previous spatial-plane candidate's failure to recover the missed structures.

These are correlations, not causal proof. The screenshot's hand-drawn outlines are not pixel-accurate. Validation images are related to training inputs (29/34 have a training input with correlation at least 0.99), so all scores remain internal diagnostics.

## Frozen candidate and controls

Use the exact frozen 167 source/label TIFF SHA256 manifest and 133/34 split in [`results/nuc_internal_split_v1.json`](results/nuc_internal_split_v1.json). Confirm all hashes before fitting. The completed baseline and illumination experiment remain unchanged.

Only the **training image's local contrast** changes. After the existing crop and existing image/mask augmentation, independently for each training patch:

1. With probability **0.50**, form a reflect-boundary Gaussian low-pass image `G64(I)` with sigma **64 pixels** on that **512×512 patch**.
2. Sample `alpha` uniformly from **[0.50, 1.00]** and set `I_aug = clip(G64(I) + alpha × (I − G64(I)), 0, 1)`.
3. With probability 0.50, leave the existing augmented patch unchanged. Never change its mask or geometry.

Use an independent deterministic random stream so image-pair choice, crop, flips, rotation, gamma/scale/noise draws in the original pipeline remain unchanged. The implementation may use an FFT approximation to the Gaussian, but its reflect boundary behavior and numerical equivalence to a spatial reference must be tested before training. The existing source-image audit used full 1200×1200 images; patch-boundary behavior can differ, so check the actual patch implementation on representative training patches before interpreting it as a target-domain simulation.

Keep the original per-image normalization and **all** remaining scientific settings fixed: fresh 2D U-Net initialization, BCE plus soft-Dice loss, foreground sampling, other augmentation, 512-pixel patch, 128-pixel tile overlap, batch size 4, 100 steps/epoch, 100-epoch cap, Adam 1e-4, seed 42, clean-validation checkpoint/early-stopping rule, and the original validation-only threshold grid. The transform is opt-in and defaults OFF, leaving the baseline and Mito/LD paths unchanged. No target label or manual mask is modified; no Nuc model is fine-tuned from the baseline or illumination candidate.

## Locked evaluation

The **clean** 33 nonempty validation images select the candidate checkpoint at fixed threshold 0.50 and, subsequently, its deployment threshold according to the existing Nuc script. The one empty validation image `nuc_005.tif` does not select either. Pair the candidate with the frozen baseline by validation filename and compare both at **fixed 0.50**: mean, median, sample SD, 10th-percentile and minimum Dice, IoU, precision, recall, paired per-image changes and count of wins. Separately report the one empty-mask image's false-positive pixels. Report selected-threshold policies descriptively, with their differing cutoffs explicit.

After candidate training, perform **one synthetic contrast stress test**, never used for checkpoint or threshold selection. Apply the same formula at `alpha=0.50`, sigma 64 and reflect boundaries to each already-normalized **full validation input**, leaving every validation mask untouched. Evaluate the frozen baseline and the candidate at common threshold 0.50 on these 34 transformed inputs; report paired nonempty metrics and the one true-negative false-positive count. This test examines response to a known perturbation, not performance on real target images. It is expected to favor a model trained on that perturbation, so do not call it independent confirmation.

The candidate earns a **review preview**, not production status, only if on clean validation mean Dice falls by no more than 0.005, mean precision and recall each fall by no more than 0.010, and worst-image Dice falls by no more than 0.020 relative to baseline at fixed 0.50. Synthetic-stress gains and per-image consistency are reported regardless of gate outcome, without changing this rule. If the clean gate passes, generate same-name 1200×1200 `uint8` 0/255 masks for the first ten target `fluo` images in a fresh ignored local directory, with a side-by-side baseline comparison. Do not overwrite Desktop `Nuc_label`, and do not use the approximate target circles to choose the model, threshold, augmentation probability or strength.

Passing the clean gate or improving synthetic-stress Dice cannot establish better target segmentation. The candidate can only be promoted after a defensible target-domain assessment, which the current incomplete circles do not provide. If no stable internal advantage or convincing qualitative target improvement appears, retain the original Nuc prototype as the working baseline and stop this line rather than tuning several contrast ranges.

Use one verified Secure Cloud GPU Pod at most, with a fresh output directory and a cost guard; download and SHA256-verify archives/checkpoints before deleting the Pod. Keep TIFFs, masks, predictions, checkpoints, logs and credentials out of Git. Record code/config hashes, full metrics, cost and cleanup status in a small report.
