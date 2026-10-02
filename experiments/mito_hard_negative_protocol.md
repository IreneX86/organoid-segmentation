# Mito exploratory one-factor hard-negative sampling protocol

**Status:** designed after inspection of frozen OOF errors; **not implemented or trained**. It is a hypothesis, not independent preregistration. The frozen five-fold map and baseline tag remain unchanged.

## Why this candidate

Foreground ≤5% images have similar absolute FP counts to other images but far fewer TP pixels, and 27.3% of their FP lies >10 native pixels from GT. Many distant FP are locally bright and high-confidence. Merely lowering foreground-centred patch sampling from 0.60 to 0.35 did not improve five-fold validation. The candidate therefore changes *which annotated-background locations* the existing non-positive sampling branch presents, without changing its overall share. Ambiguous or missing annotations could make this harmful; visual review is needed before interpreting any apparent gain.

## The single changed factor

Keep the baseline's **0.60 foreground-centred patch branch**. Divide its original 0.40 non-positive branch into **0.20 unchanged uniform random crops** and **0.20 hard-negative-centred crops**. Select a hard centre uniformly from pixels in that fold's **training** images satisfying all of:

1. Manual training mask is background (0).
2. Native-pixel distance to any labelled foreground is **>10 px**.
3. Baseline-normalized intensity is at or above that training image's GT-background **90th percentile**.
4. Local residual `normalized image − Gaussian(normalized image, sigma=12 px)` is at or above that training image's GT-background **90th percentile**.

These thresholds define one fixed method; do not grid-search them. Candidate coordinates and intensity statistics must be derived only from the training images/masks for that outer fold. The centred 512×512 patch may contain labelled foreground elsewhere. Keep crop geometry, border handling, mask alignment and augmentation unchanged. If a training image has no eligible centre, fall back deterministically to the original uniform branch and record the event. A read-only feasibility scan found ≥4,274 eligible centres in each of the current 36 images.

Keep the original four-pool U-Net, BCE+soft-Dice, 1–99.8 per-image normalization, 512×512 patches, batch 4, 100 steps/epoch, 100-epoch cap, Adam 1e-4, current augmentation, EarlyStopping, 128-overlap inference and validation-selected threshold. Do not combine this with sampling 0.35, Tversky, three-pool or overlap 256. Use fresh output directories and the exact frozen split.

## Leakage and decision rule

For each outer fold, train on that fold's training images only. Use its validation images for checkpoint, early stopping, threshold and comparison with the frozen baseline on the *same* validation images. Before any candidate outer-test export, implement and lock a per-fold validation-only gate with the existing `mito_round2_gate.py` mean/low-foreground/tail/dispersion safeguards. Save all five decisions first; export candidate outer-test metrics only for selected folds and reuse frozen baseline OOF rows for rejected folds. Never retrain because a fold looks poor. Report selected-policy OOF Dice, paired per-image deltas, SD, worst case, low-foreground Dice, precision/recall and fold consistency. A one-fold gain or lower recall/tail is insufficient to replace baseline. Raw candidate test scores for rejected folds remain unknown.

Because the method was inspired by already-inspected OOF images, even favorable internal CV results are **exploratory**. Independent newly labelled data is needed for strong generalization evidence. If expert review suggests many candidate centres are actually unlabeled Mito, stop and resolve annotation definitions before weighting them more heavily; do not edit masks as part of this experiment.

## Resource boundary

No GPU run or Runpod Pod is authorized by this protocol. A read-only Runpod catalog query on 2026-10-02 listed Secure Cloud A40 at **US$0.49/hour**. Based on prior five-fold runs, budget approximately **2–3 hours and US$1.1–1.8** for one stable A40 Pod including modest setup/transfer overhead; recheck live price and funds before any authorized creation. Do not use the failed Community RTX 4090 setup or alter Cetus.
