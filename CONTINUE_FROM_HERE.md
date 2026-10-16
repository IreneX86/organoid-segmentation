# Organoid experiment continuation

Updated: 2026-10-16 (Australia/Sydney). This file records operational state; the frozen
scientific protocols and fold maps remain in `experiments/`.

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

## Current Nuc investigation (2026-10-16)

- The working Nuc model remains the original 167-pair internal prototype in
  Git-ignored `runpod_results/nuc_internal_20261015/full_results/model/`.
  Its frozen 133/34 split has many highly similar train/validation images;
  internal Dice is not an independent target-domain accuracy estimate.
- The [low-local-contrast protocol](experiments/nuc_low_contrast_protocol_20261016.md)
  and code were pushed as commit `682248b` before candidate results were read.
  A single Secure Cloud A40 run finished 80 epochs, best checkpoint epoch 60,
  and completed the fixed synthetic-stress test. Full archive, checkpoint,
  334 source hashes, validation TIFFs and result provenance passed the local
  verifier in Git-ignored `runpod_results/nuc_low_contrast_20261016/`.
- On the same 33 nonempty internal validation images at fixed threshold 0.50,
  Dice changed **0.956452→0.957296** (paired +0.000844, 17 wins/16 losses);
  the only true-negative validation image's FP pixels changed **2,744→5,821**.
  The candidate is much more robust to the *synthetic contrast transform it
  was trained on*, but the first-ten real `fluo` previews did not visibly
  recover many user-circled nuclei. See the [paired report](reports/nuc_low_local_contrast_20261016.md).
- Keep the original Nuc prototype and existing Desktop `Nuc_label` untouched.
  Candidate previews, masks, checkpoint, logs and raw TIFF copies are all
  local-only. The one Runpod Pod `21spiybvyusvqx` was deleted after SHA256
  verification; Pod, Endpoint, Network Volume and Cluster lists are empty.
  Estimated GPU charge is about US$0.66 plus small disk cost; Runpod billing
  buckets were incomplete at the last read.
- A read-only semantic audit is complete. The rough target-circle sizes overlap
  labelled-nucleus sizes, but many hints remain missed. The train-only nearest
  comparison of true-negative `nuc_002`–`004` with positive `nuc_008` found
  whole-image correlation 0.985–0.988 yet much weaker local dark-core contrast
  in the negatives. The user's `nuc_006` decision used adjacent `nuc_007`,
  which suggests some labels may rely on context unavailable to an independent
  2D model. This does not establish label error or target accuracy; details
  are in the paired report and ignored `semantic_audit/` directory.
- Next work: assess a near-duplicate-grouped validation design using existing
  labelled images before claiming improved generalization. Determine whether
  single-image 2D input can support the intended human Nuc decision rule.
  Do not change original masks or claim target Dice without complete target
  contours. Do not rescan local-contrast alpha/probability after this weak
  real-target result.

### Source and failure-mode audit completed later on 2026-10-16

- [Full read-only report](reports/nuc_provenance_and_error_audit_20261016.md),
  [source-dependence chart](reports/assets/nuc_provenance_summary.svg), and
  [object-error chart](reports/assets/nuc_error_object_audit_20261016.svg)
  are in Git. Original TIFFs, masks, prior checkpoints, and Desktop
  `Nuc_label` were not modified; no GPU Pod or new training was started.
- 149/167 Nuc inputs have a nearest-image neighbor whose filename index is
  adjacent, and mask overlap decreases with index gap. This is consistent
  with a sequence but does **not** prove Z order or identify organoid/field
  boundaries. TIFF metadata contain no acquisition/Z mapping. Two Nuc inputs
  exactly match LD inputs; none exactly match the 1,330 current
  `organoid_data/*/fluo` images. Only a verified source manifest can unblock
  2.5D; the [feasibility protocol](experiments/nuc_25d_feasibility_protocol.md)
  gives the required checks and same-group 2D comparator.
- Under a new thumbnail method, 32/34 validation images have a training input
  with global correlation >=0.99. This is descriptive dependence evidence,
  not a verified biological grouping. Do not interpret current internal Dice
  as cross-organoid accuracy or claim fivefold grouped CV from these files.
- At the frozen 0.45 threshold, 22 of 1,175 nontrivial labelled validation
  components have <10% pixel coverage, and the weakly detected objects tend
  to be smaller and have less dark-core/bright-ring contrast. Target red
  circles are only rough hints; no target Dice was computed. A single fixed
  input-only local-background correction narrowed one train/target detail
  statistic but amplified texture on confirmed negative training images;
  its segmentation effect has **not** been tested.
- Next work is source recovery: find acquisition metadata or export mapping
  from each labelled TIFF/mask to original organoid, field, and Z plane.
  Until it is verified, keep the original 2D Nuc model. Do not concatenate
  files by filename, tune against the target red circles, or train 2.5D.

### Target-stack and labelled-source follow-up on 2026-10-16

- [Decision report](reports/nuc_25d_decision_and_domain_shift_20261016.md),
  [labelled-source batch audit](reports/nuc_provenance_followup_20261016.md),
  [target-stack continuity chart](reports/assets/nuc_target_stack_audit_20261016.svg),
  [target-domain chart](reports/assets/nuc_target_domain_audit_20261016.svg),
  and [fixed rough-hint context chart](reports/assets/nuc_fixed_hint_context_20261016.svg)
  are the new read-only research outputs.
- User confirms each target `cell_*` folder is an independent organoid with
  consecutive slices. All ten `fluo` folders together hold 1,330 gap-free,
  single-page 1200×1200 uint16 TIFFs. Sampled adjacent-image similarity and
  native-pixel XY checks support candidate 3–5-slice inference, though TIFF
  headers do not give physical Z spacing or direction.
- In the labelled Nuc pool, the 148-image high-similarity component matches
  the November 2025 export-time cohort exactly. Several October/May files
  interrupt the numeric sequence (e.g. `nuc_009` and `nuc_011` correlate
  0.999 while `nuc_010` correlates only about 0.396 with either). The export
  timestamps do not identify biological sources. No exact or credible
  rotation/reflection/shift mapping to current target stacks was found.
- The pre-existing 133 training inputs have median fixed local-detail q90
  0.01799 versus 0.01129 on 120 systematic target inputs; 103/120 target
  samples are below the training tenth percentile. This is an input
  distribution difference, not proof of a particular imaging cause or a
  correction that improves segmentation. In 11 low-coverage rough hints on
  `fluo_001`, only one showed much stronger dark-core/bright-ring contrast
  in `fluo_002/003` at the same unregistered XY box; this does not establish
  that 2.5D will or will not help.
- No model training, GPU resource, data/mask overwrite, baseline edit, or
  Desktop `Nuc_label` change occurred. Current gating task: recover the
  acquisition/export mapping for the **labelled** Nuc files and verify at
  least several independent organoids and center-plane mask semantics.
  Only then compare source-grouped 2D and 3-channel 2.5D models under the
  existing conditional protocol. Without this provenance or new independent
  complete target labels, do not claim a reliably improved Nuc model.

### Nuc source-search and export-cohort check on 2026-10-16

- A read-only search of relevant Desktop, Downloads, Documents files and Git
  history found no original microscopy stack or per-image Nuc source/Z
  mapping. Cetus, external drives, cloud, and unrelatedly named archives were
  not covered; the provenance gate remains open, not proven impossible.
- [Frozen-validation export-cohort audit](reports/nuc_export_cohort_validation_20261016.md)
  reuses the archived scores with all 167 current input hashes matching the
  split. October 2025, November 2025, and May 2026 mtimes have 1, 32, and 1
  validation images; fixed-0.50 nonempty Dice averages are 0.9488, 0.9584,
  and 0.9045. File mtimes are export/copy clues, not organoid identities. Each
  cohort appears in both train and validation, so these are descriptive, not
  independent source tests. The true-negative `nuc_005` has 2,744 FP pixels
  at 0.50.
- Next useful input, if available, is the original acquisition/export manifest
  or microscopy stack that maps each labelled center TIFF to a verified
  organoid, field, and Z plane. Do not infer that mapping from `nuc_###`
  numbering or these cohort metrics. Keep the 2D model and all masks frozen;
  no new training or target-mask overwrite occurred in this follow-up.

### Nuc cross-slice review pilot on 2026-10-16

- With labelled-source/Z records unavailable, a [read-only review tool](code/nuc_cross_slice_review.py)
  and [research note](reports/nuc_cross_slice_review_20261016.md) were added.
  The local-only pack is `runpod_results/nuc_cross_slice_review_20261016_v2/index.md`.
  It covers `cell_22375_NZ_100/fluo_001…010` using the archived frozen 2D
  masks, not the possibly edited Desktop `Nuc_label` masks.
- Ten input manifest hashes and the archived mask hashes passed. The pack has
  native-grid previous/center/next grayscale, full overlay, and mask-outside
  marker overlay for each center. Its 400 image-only browsing points include
  213 whose center coordinate falls outside the frozen mask. These are **not**
  confirmed missed nuclei or new labels.
- A neighbor-mask-only gap rule produced zero sizable interior single-slice
  omissions under its fixed definition; it cannot detect structures missed
  across several layers. There is still no complete target ground truth, so
  no target accuracy or Dice claim. Original TIFFs, frozen predictions,
  current `Nuc_label`, model, and scientific split remain unchanged. No GPU
  training or billable resource was started.

### Nuc marker retrospective audit on 2026-10-17

- The fixed dark-core review rule was checked, without parameter changes or
  neighboring `nuc_###` assumptions, on all 34 archived labelled validation
  images. See [report](reports/nuc_dark_core_validation_audit_20261017.md),
  [small metrics](reports/nuc_dark_core_validation_audit_20261017.json), and
  [reproducible script](code/nuc_review_internal_audit.py).
- Among 1175 GT components of at least 100 px, 22 had <10% coverage by the
  frozen prediction. Original top-10/20/40 ranks pointed into 6/8/14 of these
  22 components; top-40 left 285 outside-mask review points across 34 images.
  The confirmed no-nucleus `nuc_005` alone produced 39 outside-mask points.
- Decision: keep these markers only as optional human browsing aids. Do not
  merge them into `Nuc_label`, call them new labels, tune this rule on the
  viewed target images, or claim target recall from this near-duplicate,
  same-batch diagnostic. No new training or GPU resource was used.
- Next scientifically valid model comparison requires an independently
  sourced target annotation set or verifiable labelled-stack/source mapping.
  Until then retain the frozen 2D Nuc model and treat the target review pack
  as qualitative only.
