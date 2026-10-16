# Nuc spatial-illumination candidate: internal evaluation

**Decision (2026-10-16): retain the original Nuc prototype as the working baseline.** The candidate passed its predeclared *preview* gate, but it did not demonstrate a reliable improvement on the target `fluo` images. Neither model has an independent target-domain performance estimate.

## Question and frozen comparison

In the first ten `cell_22375_NZ_100/fluo` images, the original model misses several structures that the project owner marked as nuclei and also marks fragments that need review. The corrected red-circle screenshot supplies approximate positive locations, not complete pixel masks or negative annotations. An input-only audit found that normalized target-image intensity differs from the training set (median image 90th percentile 0.483 versus 0.582). It did **not** find stronger smooth background gradients in the target images. A gradient-related failure was therefore a testable hypothesis, not an established explanation.

The [protocol](../experiments/nuc_spatial_illumination_protocol.md) was frozen before this candidate run. It kept all 167 original Nuc input/mask pairs, their SHA256 manifest, the same 133/34 internal split, 2D U-Net, initialization, BCE plus soft-Dice loss, patch sampling, existing augmentation, percentile normalization, optimizer, checkpoint selection, threshold grid and early stopping. The only training-method change was a separately seeded, image-only linear illumination plane applied to 50% of training patches, with bounded offset ±0.12. Masks and coordinates were unchanged. The new model was trained from scratch; the completed baseline model was not overwritten.

The source snapshots and configuration were checked against the baseline. The candidate used the same image IDs and frozen split. Its archive and Keras checkpoint passed SHA256, gzip/tar and checkpoint CRC, metric, selection and TIFF-integrity checks. Training completed normally on one Runpod Secure Cloud A40: 79 epochs, best checkpoint at epoch 59, selected threshold 0.50. The baseline completed 86 epochs, best checkpoint at epoch 66, selected threshold 0.45.

## Paired internal validation

The direct method comparison uses **fixed threshold 0.50 for both models** and pairs the same 33 nonempty validation filenames. The one true-negative validation image is reported separately. Values below are arithmetic per-image means; Dice SD is the sample SD across the 33 images.

| Metric | Frozen baseline | Illumination candidate | Candidate minus baseline |
| --- | ---: | ---: | ---: |
| Dice | 0.956452 | 0.957885 | +0.001433 |
| Dice SD | 0.017188 | 0.016592 | −0.000596 |
| IoU | 0.917027 | 0.919626 | +0.002600 |
| Precision | 0.962526 | 0.959027 | −0.003499 |
| Recall | 0.950615 | 0.956923 | +0.006308 |
| Worst-image Dice | 0.900239 | 0.890711 | −0.009528 |
| 10th-percentile Dice | 0.940023 | 0.945136 | +0.005113 |

Paired Dice improved on **17 of 33** images and worsened on **16 of 33**; the median paired change was **+0.000914**. Precision worsened on **25 of 33** images. Summed false-positive pixels on the 33 nonempty images increased from **245,514** to **272,271**. The candidate's worst image was `nuc_163.tif`: its Dice fell from 0.904464 to 0.890711 (paired change −0.013753). On the **single** true-negative validation image, `nuc_005.tif`, false-positive pixels fell from 2,744 to zero at fixed threshold 0.50. One negative image is too little to infer a robust negative-image false-positive rate.

Both models also have validation-selected policies. The baseline selected 0.45; the candidate selected 0.50. The corresponding nonempty mean Dice values were 0.956514 and 0.957885. This is a comparison of two **model-plus-threshold policies**, not a controlled same-threshold test of the new augmentation. The selected threshold and checkpoint were both chosen on this development validation split, which makes those scores optimistic.

The preregistered preview gate passed: mean fixed-0.50 Dice was not more than 0.005 worse, median paired Dice was not below −0.005, and neither precision nor recall fell by more than 0.010. Passing this permissive gate only authorized qualitative target previews; it did not select a new production model.

## Target-image review and limits

New review-only masks were generated for `fluo_001.tif` through `fluo_010.tif`, alongside a fresh baseline rerun and ten side-by-side panels. Every preview mask retained the input's **1200×1200** grid and filename and used `uint8` background 0 / nucleus 255. They are stored in the ignored local `runpod_results/nuc_illumination_20261016/{candidate_masks,baseline_masks,comparison_panels}/` directories. The existing Desktop `Nuc_label` outputs, including the owner-edited `fluo_001.tif`, were not touched.

In `fluo_001`, the candidate and baseline cover many of the same prominent dark-core structures. Several locations circled by the reviewer, especially in the right-hand cell, remain uncovered in the candidate panel. This is a qualitative failure audit, **not** a measured target false-negative count: the circles are approximate and incomplete. The candidate did not visibly resolve the reported omission pattern well enough to replace the current output. No target Dice, precision, recall, or false-positive rate can be computed from these previews.

Across all ten target images, the two predicted masks have mean mutual IoU **0.9344**. This measures *agreement between predictions*, not accuracy. Candidate-exclusive pixels total 31,479 and baseline-exclusive pixels total 24,723; candidate total foreground increases by 6,756 pixels. Most differences are near common predicted structures. At `fluo_001` specifically, candidate foreground falls from 3.440% to 3.375% of the image, and no new candidate-only connected area of at least 100 pixels is wholly separate from the baseline prediction. These measurements reinforce the visual impression that the candidate did not newly recover the many circled locations. They cannot classify any changed pixel as a true positive or false positive.

The internal validation is also strongly related to training data: 29 of 34 validation inputs have a training input with correlation at least 0.99, and 148 of 167 images belong to one high-similarity component. These images cannot establish performance on a new acquisition. The target's stronger intensity-distribution shift is observed, but this experiment does not show that a smooth illumination plane caused or fixed the domain mismatch. The preview's two models also use different selected thresholds (0.45 and 0.50), which further limits attribution of their visual differences to augmentation alone.

## Artifacts, cost and next decision

The verified full local archive, checkpoint, per-image metrics, probabilities, validation predictions and logs remain under ignored `runpod_results/nuc_illumination_20261016/`. The archive SHA256 is `efc203ad0a7baa6513434f2ae2ea7d6c6fc123e9ae48104e5e7e02badb68b074`; the candidate `nuc_unet.keras` SHA256 is `c7c164912e3fe3fef1ee8a267cd004b0afeb011336cc85bdf50856d7904a579e`. `local_verification.json` reports PASS. The single A40 Pod `d6nfzjmkqi9pp8` was deleted after verification, and `list_pods` returned an empty list.

The Pod was created at 02:30:00 UTC at the then-quoted US$0.59/hour and lived about 61 minutes. A time-based estimate is about **US$0.61 plus small storage charges**. The Runpod billing API still exposed only a partial US$0.041785 record when checked; that figure is **not** the final charge. No TIFF, mask, checkpoint, prediction, log, or credential is committed to Git.

**Keep the original prototype as the working baseline; retain this checkpoint for research comparison only.** The next experiment should be motivated by a measured image-domain factor and evaluated with the frozen development split without treating its score as external generalization. A global photometric/background-normalization candidate is more consistent with the observed intensity-distribution difference than another ungrounded architecture change, but no such follow-up is selected here. Without full target labels or acquisition-group provenance, any claim that the target nuclei are reliably recovered remains unverified.
