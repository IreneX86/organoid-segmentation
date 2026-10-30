# Fluo-only Noise2Void pilot, 2026-10-29

## Scope and fixed split

Output the entire 100-plane `cell_20260904_23966_NZ_100/fluo.tif`; five review planes: 1,25,50,75,100. All eight source stacks are inspected, but this pilot does not batch-predict all eight stacks.

Train specimens: 23298,23299,23961,23962,23965. Validation:23963. Target held out of this denoiser training:23966. The earlier ER-student test specimen23300 remains unused for training. The target has been visually reviewed in earlier experiments; it is not a new blinded generalization study. No adjacent slices or overlapping crops cross specimen splits.

Eight evenly spaced planes per training/validation specimen, six non-overlapping 128px fields per plane; record exact grid in script. Random 64px training crops are permitted within the training set. 240 train fields,48 validation fields. Whole-source hashes checked against existing manifests. Exact duplicate crop hashes checked across splits; this does not prove independently acquired biological replicates.

## Official implementation

CAREamics0.3.5, original 2D Noise2Void (`use_n2v2=False`), U-Net depth2/base16, batch16, Adam lr0.0004, up to30 epochs, up to30 training batches per epoch, validation cap6batches, early-stop patience7. Select minimum blind-spot validation loss. Seed20261029. Normalization is CAREamics training mean/std. Native resolution; no ER channel or pseudo-mask used as a target or input. The noisy image itself supplies masked-pixel targets. Independent `.venv-n2v`, leaving prior scientific environments unchanged.

Official sources: https://github.com/CAREamics/careamics and https://careamics.github.io/latest/ ; original method https://arxiv.org/abs/1811.10980 . 3D is not attempted without reliable sampling calibration.

## Baseline, segmentation and checks

Traditional baseline: non-local means, patch5/search6/h0.8×wavelet sigma, with no clean reference. Compare raw, NLM and N2V using identical global Otsu and local Gaussian threshold (window51; threshold=local mean+1.5×raw estimated sigma). Remove components smaller than16pixels equally. This is a simple fixed comparison, not optimized ER-specific segmentation. Masks are uint8 0/255.

Save full float32 denoised stacks (not clipped), six full binary mask stacks, checkpoint, environment, training history, hashes and five native-resolution reviews using the same display range per plane. Check noise proxies, residuals, foreground coverage, connected components and skeleton length. These metrics cannot establish ER truth or true SNR. No Dice/IoU/precision/recall without trusted full manual masks. Existing algorithm-generated masks are excluded as ground truth.

Possible structured/correlated noise, blurry optics and insufficient Fluo specificity remain limitations. Inspect loss and images before recommending deployment. No automatic new experiment or parameter sweep.
