# LD experiment continuation

Updated: 2026-10-15 (Australia/Sydney). This file records operational state; the frozen
scientific protocol and fold map remain in `experiments/`.

## Completed and verified

- The 36-image LD baseline has five verified fold archives and a frozen OOF
  report under `runpod_results/ld_baseline_20261012/` (local, Git-ignored).
- The component-uniform positive-center candidate completed folds 0–4. Their
  archives, checkpoints, logs, validation decisions, and SHA256 manifests are
  saved and verified locally under `runpod_results/ld_component_20261012/`.
  Locked fold-local decisions are baseline, candidate, baseline, baseline,
  baseline. The final metadata archive and 36-image gated report passed the
  existing integrity verifier.
- Fold 3 was safely finalized after the earlier pause, with no fold 4 started
  by the original runner. Its verified archive SHA256 is
  `b3fc6bd8db4946644a12a77fbd81d73ebefdb29bc0fb5946ba6a83b0baaf1bf1`.
- The original runner was stopped. A separate, guarded fold-4-only resume on
  the same Secure Cloud A40 Pod `786a4s19lh6zmo` verified all source/data/lock
  hashes, trained only the missing fold, and exited 0. Its reviewed source is
  `runpod_ready/ld_component_resume_fold4.sh`. The Pod was terminated after
  fold 4 and final metadata were downloaded; RunPod listed no remaining Pods,
  network volumes, or endpoints.
- The predeclared global production-method rule was locked using only inner
  validation evidence in `experiments/results/ld_method_policy.lock.json`
  (Git commit `ca3eefb`, pushed). It selected the frozen baseline: candidate
  won one of five fold-local gates. No outer-test metric can revise that rule.
- Pause metadata and the exact study data/source recovery archive are saved
  locally with SHA256 manifests. The original TIFFs and derived labels were
  not changed. `._*.tif` files from macOS metadata are excluded by the
  dataset discovery rules; the input pairing checks still require exactly 36
  real TIFF/mask pairs.

## Completed analysis and locked conclusion

- The guarded local OOF audit verified both baseline and candidate fold
  archives, the final report archive, all 36 per-image records, full-size
  TIFF probabilities/masks, and recomputed pixel metrics. Its large working
  results are Git-ignored under `outputs/ld_oof_error_audit_20261012/`.
- The small paired metrics, five validation decisions, audit summary,
  four SVG figures, and scientific interpretation are in
  `reports/ld_component_5fold_20261012/`. The fold-gated OOF Dice was
  0.763570 ± 0.108188, compared with frozen baseline 0.763693 ± 0.106970;
  paired mean change −0.000123. This is exploratory internal evidence, not
  an independent confirmation of the global production method.
- Candidate component-uniform sampling is **rejected for production under
  the predeclared rule**. The frozen LD baseline remains current. Do not
  change the gate, split, labels, or baseline tag to reinterpret the result.
- The RunPod account listed zero Pods, zero network volumes, and zero
  endpoints after termination. Pod billing was still catching up; the full
  lifetime was estimated at about US$5.2, while the new spend after this
  continuation began was below US$1, within its US$5 cap. See the report for
  the precise billing limitation. Cetus was not changed.

## Final all-36 model and next phase

- The locked baseline production method was recomputed from five-fold
  **inner-validation-only** evidence before training. Its final policy is
  `experiments/results/ld_final_all36_policy.lock.json`: 50 epochs and a 0.35
  inference threshold. The small attestation is
  `experiments/results/ld_final_selection_preflight_20261012.json`.
- The model was fitted from scratch on all 36 original inputs and their
  versioned derived binary labels. The RunPod runner completed 50 epochs with
  exit 0. The archive, checkpoint, source snapshot, logs, and hashes passed
  the local verifier. They are saved outside Git in
  `runpod_results/ld_final_20261012/downloaded/`; see its `verification.json`.
  Checkpoint SHA256:
  `0cbc8e4588748e0633549726dc4e04cdfbeb542f5533fb2c961a241b29977405`.
- A one-image prediction check confirmed same-name 1200×1200 `uint8` 0/255
  output; it was a training image and provides no independent Dice estimate.
  The RunPod Pod `ochexmpzgnriwh` was deleted after local verification.
  RunPod listed zero Pods, network volumes, and endpoints. Its itemized bill
  had not yet posted at the first check; estimated GPU time was about US$0.26,
  excluding disk or billing adjustments.
- The [final-fit report](reports/ld_final_all36_20261012/README.md) includes
  the exact batch inference and review handoff. Next, run that locked model
  on **future unseen LD TIFFs** in a fresh output directory and have an LD
  expert review predictions. Evaluate generalization only when independent
  manual labels are available. Do not retune the method or threshold on these
  36 development images, repeat completed folds, or modify the original TIFFs.

## Latest: small-LD miss / p99.95 single-factor follow-up

- Six user-confirmed bright LD misses in existing organoid predictions have
  center coordinates only, not full masks. They motivated a fixed p99.8→p99.95
  normalization hypothesis; they were not used to choose folds, checkpoints,
  thresholds or the method. The [preregistered protocol](experiments/ld_highlight_normalization_protocol.md)
  changed this one factor in both training and inference.
- All five new folds completed on one Secure Cloud A40 Pod. All 16 archives
  downloaded with SHA256 verification; local verification status is `PASS` at
  `runpod_results/ld_highlight_20261014/downloaded/local_verification.json`
  (Git-ignored). Five checkpoints, histories, masks/probabilities and decision
  locks remain there. All five fold-local validation gates were locked before
  any candidate outer-test evaluation; gates chose candidate for folds 0, 3,
  4 and baseline for 1, 2.
- Raw p99.95 five-fold OOF Dice is 0.763508 ± 0.109167 versus frozen baseline
  0.763693 ± 0.106970. Low-foreground eight-image Dice is 0.727321 versus
  0.730649; outer OOF small-component any-overlap is 96/150 versus 97/150.
  The exploratory fold-gated combination has Dice 0.764357 but a worse
  worst-case Dice and lower recall. There is **no reliable improvement**;
  baseline and existing `fluo_label` remain untouched. Detailed paired results,
  FP/FN audit, figures and limitations are in
  [the report](reports/ld_highlight_p9995_20261014/README.md).
- RunPod Pod `gyptsee63uws8v` was terminated after local verification; live
  Pod, Endpoint and Network Volume lists were empty. The cost-guard automation
  is paused. The observed GPU rate was US$0.59/hour; approximately US$1.10
  including temporary disk is the Pod-lifetime estimate, while US$0.586682
  had posted in an incomplete billing read. Check the final itemized bill once
  its remaining time bucket has posted.
- Next step: obtain complete independent manual LD masks on unseen organoid
  images, including small confirmed droplets and similar bright background.
  Predefine the size-stratified evaluation before opening those labels. Do not
  train an all-36 p99.95 candidate, switch the production model, rescan nearby
  percentiles, or interpret the six center points as a Dice test without a
  fresh scientific decision supported by new evidence.

## Provisional correction of six known LD misses (2026-10-14)

- The user confirmed that six previously missed center points are LD but did
  not supply pixel contours. A deterministic local-intensity procedure proposed
  **algorithmic, object-only** 1200×1200 masks for those six centers. Their
  boundaries are not manual ground truth. The first, fourth and fifth centers
  have notable boundary ambiguity; the fifth is an elongated bright band.
- A separate review-only 100-image prediction set was built under the local,
  Git-ignored `outputs/ld_external_review_20261014/provisional_fluo_label_100/`.
  Ninety-four masks are byte-for-byte copies of the existing `fluo_label`; the
  six known misses receive the proposed objects by pixelwise union, adding 275
  foreground pixels in total. Independent reread confirmed all 100 TIFF names,
  1200×1200 uint8 0/255 geometry, per-file SHA256, and that no other pixels
  changed. Its manifest, contour QC, reproducible scripts and six-row local
  visualization are in `outputs/ld_external_review_20261014/`.
- The original inputs, existing `fluo_label`, 36 labelled training pairs and
  frozen model were unchanged. No new GPU work or training was started. These
  targeted revisions are a practical review copy, **not** evidence of a better
  general model. Do not use the object-only masks as training targets or full-
  image truth, and do not compute an unbiased Dice from these six selected
  errors. The frozen LD baseline remains the scientific production model.

## Latest: symmetric focal BCE + soft-Dice LD ablation (2026-10-14)

- The [preregistered single-factor protocol](experiments/ld_focal_dice_protocol.md)
  replaced only the BCE term with symmetric, unweighted focal BCE (gamma 2).
  The five grouped folds, labels, sampling, architecture, normalization,
  thresholds and all other training settings remained frozen. Implementation
  commit `63c5ed7` preceded candidate fitting and outer-test scoring.
- All five candidate folds finished. The five validation-only decision locks
  chose **baseline in every fold** before any candidate outer-test labels were
  read. Do not relax the fold-1 small-component gate (+0.09677 versus the
  preregistered +0.10). No fold needs retraining or another threshold search.
- Sixteen local archives passed SHA256 checks and the full local verifier
  returned `PASS`. Checkpoints, training histories, logs, 36 full-size OOF
  predictions and metrics are under Git-ignored
  `runpod_results/ld_focal_20261014/downloaded/`; the verification record is
  `local_verification.json` there. Small decisions and metrics are in
  `experiments/results/` and the [paired report](reports/ld_focal_gamma2_20261014/README.md).
- Raw candidate OOF Dice was **0.762659 ± 0.094865**, versus frozen baseline
  **0.763693 ± 0.106970**; paired mean change **−0.001033**. Small semantic
  component detections rose only 97/150→100/150, while total FP increased
  21,917→23,516 and distant FP increased 6,636→8,105. This is exploratory
  internal evidence, not an independent external confirmation. **Reject focal
  for production; retain the frozen baseline and existing `fluo_label`.**
- The sole A40 Pod `8p4tkdl8ffda12` was deleted after local verification.
  Live Pod, Endpoint, Network Volume and Cluster lists were empty. The
  observed rate was US$0.59/hour. Estimated lifetime cost was about US$1.10;
  the itemized bill was delayed (US$0.579292 posted at the latest read), so
  reconcile the final charge after all hourly buckets post. No new GPU work
  is running.
- Next scientific step: obtain independent images with **complete manual LD
  contours**, including small confirmed droplets and comparable bright
  background. Predefine size-stratified detection, distant-FP and full-image
  Dice/Precision evaluation. The eight previously confirmed discovery points
  and six provisional object-only corrections are not complete independent
  ground truth and must not select a new model or threshold.

## Completed: human-preferred focal voting review set (2026-10-15)

- After visual review, the user preferred a separate five-fold focal-vote
  prediction set for **only** `cell_22375_NZ_100/fluo` (100 TIFFs). This is an
  operational choice for review, not scientific promotion of focal loss. All
  five preregistered validation gates rejected focal; its raw candidate OOF
  Dice was **0.762659** versus **0.763693** for the frozen baseline. The
  baseline model, original `fluo_label`, experiment decision locks and focal
  scientific report remain unchanged.
- Inference uses the five saved focal fold checkpoints, each image's p1–p99.8
  normalization, 512×512 sliding tiles with 128-pixel overlap, and the folds'
  locked validation thresholds **[0.65, 0.80, 0.60, 0.35, 0.50]**. Each fold
  casts a binary vote; at least 3/5 foreground votes produce a `uint8` 0/255
  mask with the input TIFF's name and dimensions. The voting ensemble itself
  has no independent validation result.
- Workspace review and audit files are under Git-ignored
  `outputs/ld_focal_review_cell22375_20261015/`; the verified local handoff
  is `~/Desktop/Organoid/organoid_data/cell_22375_NZ_100/fluo_label_2/`.
  All **100/100** same-name masks passed dimensions, `uint8` 0/255 encoding,
  input/mask SHA256 and Desktop-copy checks. The first 76 existing Desktop
  masks retained their hashes; only the missing 24 were copied in. The full
  per-image hash manifest is
  `outputs/ld_focal_review_cell22375_20261015/review_prediction_manifest.json`,
  copied to Desktop as `fluo_label_2/prediction_manifest.json`; both copies
  have SHA256
  `4ac869d957f9fdb1bda97598652f2b7071542315b6950ec3d59481a7dbbea9f5`.
  The prior 76-image partial manifest is historical only.
- The first 76 images were already viewed by the user during this review;
  these images are no longer untouched external test cases. Visual preference
  does not provide complete independent manual contours or an unbiased Dice
  comparison. See the [separate handoff record](reports/ld_focal_human_preferred_inference_20261015/README.md)
  for scope, provenance, the reusable inference command and verification
  record. The archived `--runs-root` is a local, Git-ignored model path.

## Current LD prediction delivery (2026-10-15)

- The first focal-voting 100-mask review set now occupies
  `cell_22375_NZ_100/fluo_label`; the prior baseline output is preserved
  locally as `fluo_label_previous`. The other nine `fluo_label` folders were
  predicted with the same five locked focal folds on one Secure Cloud A40 Pod.
  Their **1,230** masks passed source/output SHA256, name, shape and `uint8`
  0/255 checks. Across all ten current folders there are **1,330** masks.
- The generated masks are archived by cell in
  [`deliverables/ld_auto_labels/`](deliverables/ld_auto_labels/README.md).
  The ZIPs contain only mask TIFFs and portable, path-sanitized metadata;
  the original fluorescence TIFFs, older first-cell output, checkpoints and
  RunPod logs are not in Git. The full local-only audit is under Git-ignored
  `runpod_results/ld_focal_bulk_20261015/`.
- The A40 Pod `bmyathrhq8j7hx` finished with exit 0 and was deleted after
  all nine remote result archives and the combined archive had passed local
  SHA256/full-output verification. RunPod listed zero active Pods, Endpoints,
  Network Volumes and Clusters. No further GPU work is running.
- These outputs remain **automatic predictions for expert review**. Focal
  did not pass the earlier validation-only model-selection gates, and the
  3/5 vote has no independent external accuracy estimate. Do not train on
  these predictions as if they were manual labels or report their Dice as
  measured external performance.
