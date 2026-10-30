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
  the same Secure Cloud A40 Pod `[pod-id-redacted]` verified all source/data/lock
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
  The RunPod Pod `[pod-id-redacted]` was deleted after local verification.
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
- RunPod Pod `[pod-id-redacted]` was terminated after local verification; live
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
- The sole A40 Pod `[pod-id-redacted]` was deleted after local verification.
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
- The A40 Pod `[pod-id-redacted]` finished with exit 0 and was deleted after
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
  local-only. The one Runpod Pod `[pod-id-redacted]` was deleted after SHA256
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

### Nuc operational review-mask choice on 2026-10-19

- The user reviewed the first ten `cell_22375_NZ_100/fluo` predictions and
  selected the 0.03-threshold, minimum-500-pixel, hole-filled, merge-guarded
  radius-4 closing, smoothed-edge, curved two-core separation style as the
  main Nuc **review prediction** method. The [method record](reports/nuc_review_postprocessing_20261019.md)
  explains the exact transformation and limitations.
- The original Nuc checkpoint remains frozen. The ten masks and SHA256
  manifest remain local under Git-ignored
  `runpod_results/nuc_threshold003_preview_20261019/`. No original TIFF,
  manual label, or earlier `Nuc_label` mask was replaced.
- Reusable inference is `code/nuc_predict_review.py`, writing same-name,
  same-size `uint8` 0/255 masks to a fresh directory. This was a qualitative
  choice on viewed organoid images, **not** evidence of external accuracy or
  a new scientific baseline. Further batches require human review; a target
  Dice claim requires independently sourced complete annotations.

## 2026-10-29 ER Noise2Void experiment completed
- Official CAREamics0.3.5 2D N2V, five train specimens /23963 validation /23966 target; original data and historical masks unchanged.
- 19epochs, selected12, full23966 100-plane inference + NLM and six Otsu/local candidate stacks; all verified in `outputs/er_n2v_20261029/verification.json`.
- Report: `reports/ER_Noise2Void_20261029.md`. Five comparisons and residual details available.
- NOT production approved: simple thresholds still select nonspecific texture/bright tissue. No manual GT accuracy claims. No automatic retraining, no paid resources used.

## 2026-10-29 ER user-selected ROI student — ready, not trained
- User approved next small-scale experiment after 40 enriched crops. Five original examples remain unchanged.
- New code `code/er_roi_student.py`, protocol `docs/er_roi_student_protocol_20261029.md`; GPU-only training, 32 train /8 exploratory validation crops, Fluo only, partial masked BCE+Dice,40epochs max/patience8/fixed0.5.
- Ready data `outputs/er_roi_student_ready_20261029`; bundle `outputs/er_roi_student_bundle_20261029.tar.gz` with adjacent SHA256; no ER-channel TIFFs or credentials packaged.
- Dataset verifies40 paired crops, disjoint specimen splits. Loss unit test confirms ignored pixels have zero gradient; exterior-background test skipped because this TF environment lacks SciPy (not used by ROI trainer).
- Current session has no exposed Runpod MCP tools; runpodctl/flash absent from PATH; RUNPOD_API_KEY absent and ~/.runpod/config.toml absent. Restore authenticated control before deployment. No Pod created, upload performed, or training started in this step; no new fees incurred.
- These remain incomplete pseudo-labels with provisional negatives, not manual truth. Historical23965 was seen before and is only exploratory validation. Do not advertise independent generalization.

## 2026-10-29 ER ROI pilot completed, no active Pod
- Browser authenticated Runpod control worked. OneA40 Pod[pod-id-redacted] used, then terminated after verified result download. Balance4.86,spend0,approx cost0.39.
-17epochs,best9; fixed0.5 outputs all-positive on validation. Reject for production; pseudo-Dice0.7703 is degenerate all-positive baseline, not success.
-Report `reports/ER_ROI_student_pilot_20261029.md`; full results `outputs/er_roi_student_run_20261029/extracted/results/student/`. SHA verified and checkpoint loads locally. No retraining initiated.

## ER ROI BN train-only recalibration completed
- `experiments/er_roi_bn_recalibrate.py`: exact layerwise population moments from32 training inputs only, no labels/validation calibration, no optimization. Original checkpoint unchanged, trainable weights identical.
- Output `outputs/er_roi_bn_recalibration_20261029/`; validation pseudoDice0.8170,precision0.7272,recall0.9322,positive area75.04%. Still broad tissue-like regions, not ER fine network; NOT production approved.
- Report `reports/ER_ROI_BN_recalibration_20261029.md`. Next: inspect trusted within-ROI inter-line negative supervision; do not automatically convert all unlabeled pixels to negatives. No GPU fees.

## ER inter-line negative candidates — prepared, not trained
- `experiments/er_interline_negative_review.py` applied one fixed geometry+ER-trough rule to32 train crops only. Validation unopened/unchanged.
- Added75,178 provisional negatives; positives425,365 unchanged. Most pixels remain ignored. Not manually confirmed, not automatically promoted.
- Output `outputs/er_interline_negative_review_20261029`, fixed8-panel review `eight_training_proposals.png`; report `reports/ER_interline_negative_review_20261029.md`.
- Source/hash/readback/positive-preservation/synthetic checks PASS. No training or paid resources. Next training comparison must retain original validation labels and explicitly handle BN; no threshold search based on these previews.

## ER inter-line ablation completed — Pod deleted
- Supersedes prepared-only status above. One fixed experiment:32 train crops,75,178 extra provisional negatives; original8 validation files unchanged.19 epochs,best11,EarlyStopping.
- Same train-only BN calibration: pseudo-Dice0.8170→0.8198,precision0.7272→0.7495,recall0.9322→0.9046; positive area75.04%→70.03%. Only4/8 crops improve, still broad regions, REJECT for production.
- Report `reports/ER_interline_ablation_20261029.md`; comparison `outputs/er_interline_bn_calibrated_20261029/paired_all8.png`; all checkpoints/archives preserved and verified.
- Pod [pod-id-redacted] terminated; Billing resources0,spend$0/hour,balance$4.79; approx new cost$0.07. No active work or automatic retraining.
- Next useful step: single-training-crop line-fitting diagnostic, not more negative-ratio scanning. Original images and masks unchanged.

## ER train-only supervision diagnostic completed
- `experiments/er_supervision_identifiability.py`:32 train crops,75.23% pixels ignored. Constructed wider masks add596,279 unknown positives with identical masked loss; actual TF loss/zero ignored-gradient test PASS.
- These are constructed counterexamples, NOT model predictions. Report `reports/ER_supervision_identifiability_20261029.md`; figure under `outputs/er_supervision_diagnostic_20261029/`.
- Prepared `experiments/er_single_roi_fit.py` (syntax checked only):GPU-only fixed first training crop,calibrated checkpoint,frozen BN/dropout off,500 updates/10min maximum. NOT YET RUN. No new paid resources.
- Do not conclude all causes are labels: calibrated training known-pixel Dice0.793 also suggests fitting limitation. Next run diagnoses capacity; no validation selection or production promotion.

## ER single-ROI GPU diagnostic attempt — infrastructure timeout
- One A40 Pod [pod-id-redacted] created2026-10-29~09:12UTC,CA-MTL-1,$0.59GPU+$0.007disk/hour. Stayed initializing ~10min; no SSH, container/system logs empty. No uploads, no training.
- Stopped then terminated~09:24UTC. Billing verifiedGPU0,vCPU0,storage0,endpoint0,spend$0/hour,balance$4.79 (same displayed value). No second Pod created.
- Reusable payload7.46MB and runner under `outputs/er_single_roi_run_20261029/`; SHA25679c1b2af1b86333f1c2bbe47143801681018012904b8ac2686b1c2f63d4fd8c2. Only one Fluo crop,partial label,source model,code. No ER raw channel or validation data.
- `docs/er_single_roi_fit_protocol_20261029.md`, `experiments/er_single_roi_review.py` prepared. Experiment NOT completed. Next resume same protocol in new unique attempt directory; do not claim fitting results or rerun preceding full experiments.

## ER single-ROI fitting diagnostic COMPLETE (retry1)
- One Secure A40 [pod-id-redacted] in Europe succeeded; original failed allocation remained deleted. Same payload/science, no validation data.
-500 updates,18.36s; known-pixel train Dice0.71388→1.0,precision/recall1.0; ignored-positive fraction63.37%→24.19%. This is memorization, NOT a production/generalization result. Unknown regions still coarse.
- Report `reports/ER_single_ROI_fit_20261029.md`; complete model/logs/probabilities/figure under `outputs/er_single_roi_retry1_20261029/`. ArchiveSHA verified,checkpoint ZIP/load pass,sourceunchanged. CPU/GPU probabilities maxdiff0.05925,meandiff0.000306;32 threshold differences allignored,knownmetricsunchanged; retained originalGPUoutputs.
- Pod stopped+terminated,accountGPU0/storage0/endpoints0,spend$0/hour. Balance$4.72,approxnewcost$0.07. No paid resources or ongoing training.
- Next: improve complete trustworthy within-ROI line/interline supervision plus controlled stable-normalization training; do not relabel allunknown asbackground or deploysinglecropcheckpoint.

## ER dense supervision window audit — do not train this set
- `experiments/er_dense_supervision_windows.py` selected15 non-overlapping64px loss windows with320px context from32traincrops,≥75% known labels (aggregate84.39%),no new labels/validation reads. Output `outputs/er_dense_windows_20261029/`.
- Visual review reveals mostly dark nucleus-like interiors and boundary arcs. REJECT coverage-only selection for next training. Descriptive audit: only16/164 overlapping candidates have majority negatives within8px of lines; all23962. Not independent examples or validated biological threshold.
- Report `reports/ER_dense_window_audit_20261029.md`. Original data/masks unchanged,no training/no GPU fees.
- Next selection must target ER internal mesh+inter-line troughs across training specimens, not coverage/positive area alone. Preserve uncertain labels; do not deploy the single-crop memorization model or train the rejected15-window set.

## ER mesh-specific windows prepared
- `experiments/er_mesh_supervision_windows.py`: train-only topology+ER-contrast selection,16 windows (4/specimen),64px loss support with320px Fluo context. Unlike rejected coverage-only set, review shows internal mesh/gaps.
- Output `outputs/er_mesh_windows_20261029/`, manifest/protocol/verification and per-window reviews; report `reports/ER_mesh_window_selection_20261029.md`.
- No new positive/negative labels. Known coverage50.56%,positive amongknown77.73%; incomplete weak supervision remains. Source masks may be thick,Fluo detail can be weak. No GT/generalization claim.
- Synthetic single-loop reject/grid accept,source hashes,label preservation/readback pass. No validation images/labels opened,no training,no paid resources.
- Next controlled comparison needs identical stable BN settings in original-crop control and mesh-window arm to separate normalization stability from sampling effects; fixed original8val labels/threshold0.5. Do not promote on pseudoDice alone or change reviewed selection rules after outcomes.

## ER mesh controlled pilot — infrastructure failure before upload
- Fixed two-arm script and protocol ready: `experiments/er_mesh_controlled_train.py`, `docs/er_mesh_controlled_protocol_20261029.md`. Both arms500 updates,same calibrated source,frozenBN/dropoutoff,threshold0.5; no new labels or validation selection.
- Payload `outputs/er_mesh_controlled_20261029/payload.tar.gz` verified56 NPZ,16.51MB,no sourceER TIFF.
- Secure A40 Pod [pod-id-redacted] (EU-SE-1) immediately failed container creation: NVIDIA GPU requires reset. No SSH/upload/training; no experiment outcome. Stopped and terminated; resources0. Displayed balance$4.72.
- Resume from payload/protocol on a healthy host, excluding this failed host if possible; do not repeat prior scientific experiments or alter science. No result/checkpoint exists for this attempt.
- After Billing refresh: confirmed spend$0.00/hour,GPUs0,vCPUs0,storage0,endpoints0,balance$4.72; cleanup proof in attempt directory.

## ER mesh retry1 — 2026-10-30 Sydney, blocked by stock
- Reused unchanged SHA256-verified payload/protocol. No Pod created, no upload/training, new cost$0.
- Alternative regions excluded known failed EU-SE-1/CA-MTL-1 but offered no A40. Any-region A40 briefly appeared at$0.59/hour then disappeared before selection; final full catalogue had no A40 and Deploy disabled.
- Resume same controlled pilot once A40 available; no scientific changes or new results. Attempt state: `outputs/er_mesh_controlled_retry1_20261030/run_state.json`.

## ER mesh controlled GPU retry — active 2026-10-30
- User authorized alternative GPU. One Secure Cloud RTX A6000 48GB Pod [pod-id-redacted],EU-SE-1,$0.597/h total,starting displayed balance$4.72.
- Same unchanged payload/science; local task root outputs/er_mesh_controlled_gpu_20261030. Runner started once,collector session [pid-redacted] downloads/verifies results. No second Pod. New spending cap$1; save results and delete promptly.

## ER mesh comparison COMPLETE — 2026-10-30 Sydney
- Supersedes blocked/active statuses above. AlternativeGPU authorized;SecureRTXA6000 [pod-id-redacted] completed both500-update arms,unchanged science.
- Pseudo-valDice control0.6080 vsmesh0.7704 (8/8 crops improve);positivearea38.12%→65.00%;mesh remains coarse/broad. REJECT for production,do not deploy or relabel with these checkpoints. One previously seen validation specimen,notindependentaccuracy.
- Both checkpoints,16predictions and archive SHA256/ZIPverified locally at outputs/er_mesh_controlled_gpu_20261030. Report reports/ER_mesh_controlled_20261030.md.
- Pod deleted;resources0,spend$0/hour,balance$4.66;approxnewcost$0.06. No running training/resources.
- Next useful work is reliable aligned line/inter-line supervision/evaluation audit,notmorepositivecropping or thresholdsearch. Originaldata/masks/previousmodelsunchanged.

## ER gap-supervision audit COMPLETE
- Read16trainwindows only,sourcehashesPASS,no validation reads/newlabels/training/GPUfees.153geometricgaps;7548/8546gappixelsignored(88.32%);120/153gapsentirelyunsupervised.
- Rawintensityrim-vs-gapAUCmedian ER0.705,Fluo0.548 (12/16Fluo<0.6);descriptive crosschannelcontrast,notaccuracy/proof ofunidentifiability.
- Report reports/ER_mesh_gap_audit_20261030.md;reviews outputs/er_mesh_gap_audit_20261030/eight_gap_reviews.png.
- Next prepare conservative ER-supported gap-negative proposals within sameROIs,retainuncertainty,noautomaticallholebackground andnoGPUuntilreviewable. Currentstudentremainsrejected.

## ER enclosed-gap weak-negative proposals READY — no training
- New derived16windowset outputs/er_gap_negative_candidates_20261030;1832additionalweaknegatives,25759positivespreserved,old7379negativespreserved. Sourcesunchanged/hash/readback/syntheticchecksPASS.
- Report reports/ER_gap_negative_candidates_20261030.md;cyanproposals on fixed8windowoverview. NotmanualGT/notproductionapproved;mostgapuncertaintyremains. No validationopened,noGPUcost.
- Technicalmanifestduplicatekeyfailurepreservedin_partial_technical;correctedwithoutchangingrule.
- Next scientificallycleancomparison: SAME16meshwindowswith/withoutgapnegatives,samecheckpoint/updates/BN/threshold. Noarchitecture/losschange,noautomaticpromotion.

## ER gap-only ablation active — 2026-10-30
- One A6000 Pod[pod-id-redacted],$0.597/hour,cap$1. Local outputs/er_gap_controlled_20261030,protocol docs/er_gap_controlled_protocol_20261030.md.
- Same16inputs perarm,only1832ignored→negativepixelschanged;pairedinput/labelchecksPASS. No validationselection. Awaitcontainer/execution,downloadverifythenDELETE. No secondconcurrentPod.

## ER gap-only ablation COMPLETE — 2026-10-30
- Supersedes active status. Both500updatearmsfinished,allmodels/resultsarchived andSHA256/ZIPverifiedat outputs/er_gap_controlled_20261030.
- Same16meshwindows,1832negativeproposalsonly. PseudoDice0.7434→0.7748,5/8cropsimprove;positivearea63.45%→67.95%;broadmaskspersists. REJECTproduction;noGT/generalizationclaim. Report reports/ER_gap_controlled_20261030.md.
- Priororiginalmesh0.7704vscurrent0.7434 showsnondeterministicrunvariation;notfixed byseedalone. Noextraseeds/sweepsrun.
- Pod[pod-id-redacted]deleted;accountresources0,spend$0/hour,balance$4.59;approxnewcost$0.07.
- Nextinformativeoption: ER-inputpositivecontrol withsamepartiallabels toseparatelabel/objectiveissuesfromFluocrosschannelinformationissues;NOTYETRUN,noproductionchange. Do notkeepaddinggapnegativesorsearchthresholds.

## ER binary ROI semantics change — user authorized, run active
- UserclarifiedyellowER,gray/blueBACKGROUND. Newlabels map1→1,0/255→0 withinoriginal40selected320pxcrops ONLY. Not16artificial64pxsupportlabels andnotfullstacks. Oldpositives/sourcefilesunchanged.32train4specimens/8validation1specimenunchanged;newmetricsemanticsnotcomparabletopartialDice.
- Fixed500stepFluo-onlypilot,sourcecalibratedcheckpoint,frozenBN/dropoutoff,0.5threshold,noearlyselection. Protocol docs/er_binary_roi_protocol_20261030.md.
- OneA6000 [pod-id-redacted],EU-SE-1,$0.597/hour,budget$1,startbalance$4.59. Resultsroot outputs/er_binary_roi_20261030;collector [pid-redacted]. DownloadverifythenDELETE;do notlaunchduplicate.

## ER binary ROI pilot COMPLETE — supersedes active status above
- 500-step pilot finished; new binary validation Dice 0.1558, recall 0.1164; train Dice 0.7528. Reject for production, no new run.
- All results/checkpoint downloaded and SHA256 verified under outputs/er_binary_roi_20261030. See reports/ER_binary_roi_20261030.md and local all8_binary_before_after.png.
- Pod [pod-id-redacted] deleted; console resources zero and $0/hour. Balance $4.53; approximate incremental cost $0.06.
- Next proposed diagnostic: ER-input positive control with the same binary labels/split; not started. Preserve originals and old checkpoints.

## ER matched input control active — 2026-10-30
- User approved probability diagnostic and ER-input control. Probability check complete: threshold0.1 recall0.195 versus0.116 at0.5, FP nearly doubled; no threshold adopted. See reports/ER_binary_probability_20261030.md.
- Matched fresh initial weights: Fluo vs ER input, 1000fixedupdates each, existingbase16UNet, same32train/8valcropbinarylabels/split, seed20261030; no selection. Protocol docs/er_channel_control_protocol_20261030.md.
- Exactly one Secure A6000 Pod [pod-id-redacted], rate~$0.597/hour, newcap$1, startingbalance$4.53. SSH[ssh-endpoint-redacted]. Runroot/workspace/channel_control_run. Payload11.74MB only40pairedcropsandlabels.
- Collector exec [pid-redacted] auto-downloads and verifies archive under outputs/er_channel_control_20261030; runner exec [pid-redacted]. Do not launch duplicates. Run experiments/er_channel_control_review.py after transfer; downloadverifiedbeforedelete; deletePod thenverify$0/hour.

## ER channel control COMPLETE — supersedes active status above
- Both1000-update matched arms complete and verified. Fluo train/valDice0.8466/0.1642; ER0.9049/0.7425; ERbetter8/8correlatedvalcrops. Report reports/ER_channel_control_20261030.md; figure outputs/er_channel_control_20261030/all8_channel_control.png.
- No production promotion. Labels ERderived, validationonepreviouslyseen specimen. Difference supports cross-channelgeneralization limitation, not impossibility. Nextsuggestedread-onlyalignment/domainaudit, notanotherthresholdscan.
- Pod[pod-id-redacted]deleted; allresources0 and$0/hour. ApproxUS$0.06; balance$4.47. Allcheckpointsand80predictionNPZs downloaded/SHAverified. Noactiverunneroradditionaltraining.

## ER alignment audit complete — 2026-10-30
- 40pairedcrops matched originalTIFFcoordinates/Z exactly. Bounded label-freeXY±16px/Z±2 correlation diagnostics show no consistent small globaloffset explaining failure; NOT proof of perfectalignment. ValidationdetailmedianXY0/0, gain0, Zgain0.00169. No transformations applied.
- Validation simpleFluointensity/texturestats lie withintraining-specimenranges; uniformbrightnessaloneunproven. See reports/ER_alignment_audit_20261030.md and outputs/er_alignment_audit_20261030/five_specimen_alignment.png.
- Nextrecommendation: morediversefixed-rulepairedcropsfromexistingdata + organoid-groupedvalidation, notthresholdorarchitectureblindsearch. NoGPU/no newtraining; originalTIFF/masks/modelsunchanged.

## ER diverse ROI expansion prepared — 2026-10-30
-114newpaired320pxcrops +40immutableanchors=154, same5specimens. outputs/er_diverse_roi_20261030/manifest.json; allhashes/shapes/labelsverified; noexactduplicateinputs. Newlabelsareautomaticcandidates, notnewmanualtruth.
-5groupedfoldslocked; same8oldtestanchors/fold forcontrolvs expanded; wholeheldoutspecimenexcluded, itsnewcropsunused. Control32trainvs expanded122–125train. Protocol docs/er_diverse_roi_cv_protocol_20261030.md.
- NoGPU ortrainingstarted. Next is10matchedshortfits onlyafterexecutionbudgetresolved; no claims ofCVimprovementyet. OriginalTIFF/masks/modelsunchanged.

## ER diverse paired CV ACTIVE — 2026-10-30
- Supersedes prepared-only status. One Secure A6000 Pod [pod-id-redacted] at about $0.597/hour; new cap $1. Hard cleanup by 04:30 UTC. Training launched once with lock; CUDA/TF preflight passed.
- Frozen 5 grouped folds, 2 arms each, 1,000 updates, same per-fold initial weights. Only crop pool changes. No held-out selection or production promotion.
- Local outputs/er_diverse_cv_20261030; collector session [pid-redacted] automatically downloads and SHA256 verifies. Run experiments/er_diverse_roi_cv_review.py after collection, then delete Pod after verified checkpoint preservation. No duplicate training or Pod.

## ER diverse paired CV COMPLETE — supersedes active entry
- All10fits/5folds complete, downloaded and verified; outputs/er_diverse_cv_20261030. Original-pool OOF Dice0.1914 vs expanded0.0908; 0/5folds and2/40crops improve. REJECT expanded configuration; neither arm production-ready.
- Recall0.1773→0.0653; newcrop meanforeground6.78%vs old13.43%; fixedupdates reduceanchor exposure. Poorertrainingfit suggests optimization/exposure is also a confound; do not claim moredata intrinsically harmful.
- All10checkpoints SHA256/ZIP verified, all80heldout predictions verified; protocol/source/initialization/split and recomputedcountsPASS. No further experiment launched.
- Pod[pod-id-redacted] terminated afterverification. Console GPUs/vCPU/storage/endpoints0, spend$0/hour; balance$4.31, approximate cost$0.15.
- Report reports/ER_diverse_cv_20261030.md; smallmetrics reports/metrics/er_diverse_cv_20261030. Next: training-crop label/visibility and exposureaudit before preregistering a new training-only/inner-validation convergence comparison; do not tune on outer-foldresults.

## ER exposure audit complete — 2026-10-30
- Read-only exact sampler replay: anchor exposure125→32.53; actual sampledpositivefraction13.42%→8.49%. NewFluopositivecontrast0.1219vs old0.0736; globallydimmernewpositives explanationunsupported. Fixedmedian-densityreview10crops saved. No exhaustive manualtruth; labelcompleteness unresolved, no labelchanges.
- Report reports/ER_diverse_exposure_audit_20261030.md; outputs/er_diverse_exposure_audit_20261030. NoGPU/no newtraining.
- Next proposed: preregister specimen-isolated INNER validation convergence comparison with matched original/expanded pools, same science; never chooseupdates on previouslyseen outerfoldscores. Do not auto-promote or conflate sufficienttraining with guaranteedgeneralization.

## ER inner-convergence study prepared — 2026-10-30
- Frozen 5 folds:3training/1inner-validation/1outer specimen; same24oldanchors vs91–94expandedtrainingcrops. Hash-locked outputs/er_inner_convergence_20261030/protocol.json.
- GPU-only runner experiments/er_inner_convergence.py: matchedweights,1000–4000updates,innercheck250,patience4,min_delta.002,threshold.5. No outerdata loaded by selection runner; all10decisions must lock before separateouterevaluation.
- Local isolation/hash/tie/compile checksPASS. No Pod or newtraining. Proposed newbudgetUS$1; estimated$0.4–0.7 using prior rate, livepricecheckrequired. Previous studybudgetnotreused. Docs docs/er_inner_convergence_protocol_20261030.md.

## ER inner-convergence historical launch — superseded by completion below
- One SecureA6000 [pod-id-redacted] created~2026-10-30T05:05UTC; rate~$0.597/hr, balance$4.30; hard cleanup06:25UTC. No secondPod. SSH[ssh-endpoint-redacted] pinnedknown_hosts under outputs/er_inner_convergence_20261030.
- Runner exec [pid-redacted], collector [pid-redacted]; remote /workspace/inner_convergence_run. Fixed protocol unchanged. All ten innerdecisions lock before outer evaluation. No CPUtraining.
- Collector downloads+SHA256verifies; then experiments/er_inner_convergence_review.py verifies20checkpoints,decisions,80OOFoutputs. Downloadverifiedbeforedelete; verifyzeroresources/spend. Do not rerun for poor metrics.

## ER inner-convergence COMPLETE / CLEANED UP — 2026-10-30
- Five matched folds / ten models completed with locked inner-only selection before any outer evaluation. Control Dice0.2072, expanded0.1753; paired−0.0319, 1/5folds and13/40crops improved. No production approval, no retraining.
- Local outputs/er_inner_convergence_20261030: archive SHA256 verified;20 checkpoints and80outer predictions verified; decisions unchanged. Review report reports/ER_inner_convergence_20261030.md; complete per-fold figures and learning curves retained.
- Pod [pod-id-redacted] deleted after verification. Console GPU/vCPU/storage/endpoint0, spend$0/hour; balance$4.06; approximate new cost$0.24 withinUS$1. No active task/resource remains from this experiment.
- Next: training-only cross-channel target completeness/visible-structure audit before any new protocol. Do not use outer results to retroactively reselect checkpoints or thresholds; original sources/labels/models unchanged.
