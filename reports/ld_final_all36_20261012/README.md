# LD baseline: final all-36 fit and inference handoff

**Status (2026-10-12): final fit complete and locally verified.** The locked production choice is the LD baseline. The five-fold numbers below describe internal development data; they are not a performance score for the new all-36 model.

## Decision and training contract

The [predeclared global method rule](../../experiments/ld_production_selection_policy.md) and [locked decision](../../experiments/results/ld_method_policy.lock.json) retain the frozen baseline: the component-uniform candidate passed its fold gate in only **1/5** folds, its mean paired inner-validation Dice gain was **+0.004734** against a required **+0.01**, and one fold missed the precision guardrail. This choice did not use outer-test scores. The [final-fit policy](../../experiments/results/ld_final_all36_policy.lock.json) fixes **50 epochs** and a **0.35** inference threshold, each the median of the five baseline inner-validation selections. The [selection preflight](../../experiments/results/ld_final_selection_preflight_20261012.json) matched the locked policy hashes and found no outer-test evidence in its selection bundle.

The intended final model is a fresh, single-channel 2D U-Net fit on all 36 paired LD images and the separately derived `uint8` 0/1 labels. The [pretraining label rule](../../experiments/ld_antialias_label_protocol.md) defines foreground as at least half each source mask's observed maximum; original manual masks remain unchanged. The baseline uses 512×512 native-grid crops, 0.60 positive-centered sampling, BCE plus soft-Dice, batch size 4, Adam at `1e-4`, and 100 steps per epoch. The full-data fit has no validation checkpoint or threshold search. Inference uses per-image 1st/99.8th percentile normalization, 512-pixel overlapping tiles with 128-pixel overlap, and the fixed threshold of 0.35. It outputs binary semantic coverage, not verified individual droplet identities.

## Evidence and limits

The [frozen baseline five-fold report](../ld_baseline_5fold_20261012/README.md) scored each of the 36 development images once in an outer held-out fold using that fold's inner-validation-selected threshold. Image-level macro means ± sample SD were Dice **0.7637 ± 0.1070**, IoU **0.6283 ± 0.1264**, precision **0.7619 ± 0.0949**, and recall **0.7829 ± 0.1594**. The gated candidate comparison is [exploratory internal evidence](../ld_component_5fold_20261012/README.md); it did not change the global method lock. All 36 labels participate in the final fit, so training loss or predictions on those images cannot measure its generalization. Specimen identity and Z order are incomplete, suspected related views may remain across CV folds, and no independently labelled external LD set has been evaluated. Performance on future organoids is unknown.

## Completion record

| Item | Verified record |
| --- | --- |
| Final checkpoint | `runpod_results/ld_final_20261012/downloaded/model/final_ld_unet.keras` (90 MiB); SHA256 `0cbc8e4588748e0633549726dc4e04cdfbeb542f5533fb2c961a241b29977405` |
| Full archive | `runpod_results/ld_final_20261012/downloaded/archives/ld_final_model.tar.gz`; SHA256 `e35eb206326abbac6a2ea8d8fb6f5ed57763a2b93e0e908722ce972b68645484` |
| Local verification | The verifier printed `LD_FINAL_ARCHIVE_VERIFIED`; `runpod_results/ld_final_20261012/downloaded/verification.json` records 50 epochs and the locked policy/model hashes. The Keras ZIP passed CRC checks. |
| Training completion | 36 input/derived-label pairs, 50/50 epochs, runner exit 0, 2026-10-12 10:41:59–10:53:46 UTC (**11m47s runner elapsed**, including run instrumentation but excluding archive publication) |
| Final training diagnostics | Last loss **0.2521** and patch soft-Dice **0.7588**. These are training metrics only. Validation columns are `NA` because the all-36 fit has no validation subset. |
| Compute | One Secure Cloud NVIDIA A40; Python 3.11.17, TensorFlow 2.21.0, NumPy 2.4.6, tifffile 2026.3.3. GPU preflight and actual GPU matrix operation passed. During the fit, 71 ten-second `nvidia-smi` samples had mean utilization **82.5%**, median **92%**, and maximum allocated memory **43,993 MiB** of 46,068 MiB; TensorFlow allocator reservation may dominate that memory figure. |
| RunPod cleanup and cost | Pod `ochexmpzgnriwh` was permanently deleted after artifact and inference checks. RunPod subsequently listed **zero Pods, network volumes, and endpoints**. The catalog price was US$0.59/GPU-hour and Pod lifetime was about 26 minutes, implying roughly **US$0.26 GPU time**; actual itemized billing had not posted at the first post-deletion read, so disk/total charges remain unconfirmed. |

The [final-run script](../../runpod_ready/ld_final_all36.sh) checks the one-A40 environment, locked policies, dataset hashes, and model tree before it publishes a ready archive. The [artifact verifier](../../runpod_ready/ld_final_artifact.py) checks the archive hash, checkpoint integrity, manifest, source snapshots, and 50-epoch history before making a local verified directory available. Keep the model, biological TIFFs, probability maps, logs, and credentials in ignored local storage rather than Git.

The exact archived inference code also completed a one-image **geometry-only** check on `ld_001.tif`: the saved mask retained its filename, 1200×1200 pixel grid, and `uint8` 0/255 encoding. The ignored local record is `runpod_results/ld_final_20261012/inference_smoke/prediction_manifest.json`. Because this was one of the 36 training images, it is **not** an independent accuracy test.

## Future unseen LD TIFF batches

Use a **verified** all-36 model directory and the exact inference code saved with it. Put new single-plane LD TIFFs in a separate input folder; use fresh output folders so an existing batch cannot be overwritten. From the repository root, after setting the paths below, run:

```bash
MODEL_DIR=runpod_results/ld_final_20261012/downloaded/model
INPUT_DIR=/path/to/new_ld_tiffs
PRED_DIR=/path/to/new_ld_predictions
REVIEW_DIR=/path/to/new_ld_review

python "$MODEL_DIR/source_snapshot/code/ld_predict.py" \
  --model-dir "$MODEL_DIR" --input-dir "$INPUT_DIR" \
  --output-dir "$PRED_DIR" --save-probabilities

python "$MODEL_DIR/source_snapshot/code/ld_review_queue.py" \
  --prediction-dir "$PRED_DIR" --input-dir "$INPUT_DIR" \
  --reference-manifest experiments/results/ld_binary_midpoint_v1_manifest.json \
  --output-dir "$REVIEW_DIR" --contact-sheet
```

The predictor verifies the checkpoint, policy, and code hashes, preserves each TIFF's filename and pixel grid, writes `uint8` 0/255 masks by default, and records file hashes in `prediction_manifest.json`. The review command checks that inputs, masks, and probability maps match one-to-one, then writes `review_queue.csv` and `review_summary.json`. Its ranking uses uncertain predicted boundaries and foreground-fraction outliers as **uncalibrated triage cues**; the contact sheet contains biological image pixels. Review flagged and representative unflagged cases with an LD expert, keep the review material outside Git, and record any manual correction as a separate version. If independent manual labels become available, define the evaluation and specimen grouping before scoring; do not retune this locked threshold on the 36 development images.
