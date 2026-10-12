# LD experiment continuation

Updated: 2026-10-12 UTC. This file records operational state; the frozen
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
