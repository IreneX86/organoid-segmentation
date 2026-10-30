# Expanded ROI paired grouped CV — prepared, no training started

## Hypothesis
More diverse existing paired crops may reduce the severe Fluo-only train/validation gap. This is a data-sampling experiment, not a change of architecture, loss or threshold.

## Frozen data
`outputs/er_diverse_roi_20261030/manifest.json` contains40immutable old anchors and114new320pxcandidates from the same five specimens. Original TIFF, old labels and checkpoints remain unchanged. New crops inherit existing ER-derived line positives and the approved binary crop convention; they are automatic candidate labels, NOT individually confirmed manual ground truth. Gray/outside-line pixels within these new crops are provisional background; this assumption may retain omissions in the teacher. No entire source image is declared negative.

Fixed sampling: eight spaced Z targets from8%–92%depth, snapped>=4slicesawayfromanchors/othernewZ; up tothree nonoverlappingXYwindows per plane selected near20th/50th/80thpercentile foreground density amongwindowswith>=2%positive. No student predictions or scores consulted. Missing eligible windows cause a recorded shortfall, not threshold relaxation. In-plane nonoverlap and Zspacing reduce redundancy but do not establish independent cells.

Historical held-out specimens outside these five are excluded. Each fold holds out one entire specimen; no crop from it may train either arm. The eight original anchors from that specimen are the SAME test set for both arms; its new crops are unused. Across five folds all40anchors are evaluated once. These specimens/anchors were previously inspected, so call this exploratory grouped CV, never independent validation.

## Matched prospective arms
- Control:32oldtraininganchors (four specimens).
- Expanded:those same32anchors plus new crops only from the four training specimens (122–125total crops dependingfold).
- InputsFluo only. Existingbase16UNet, same fresh random initialization per pairedfold, seed20261030+fold, deterministicTF, Adam0.0003,1000fixedupdates,batch4onecroppertrainingspecimen, existingBN/dropouttrainingbehavior,noaugmentation,per-cropp1–p99.8normalization,threshold0.5.
- Use same seeded per-specimen sampling algorithm in botharms; availablecrop pool is the factor. Finalcheckpoint only, noearlystopping/threshold/modelselection on held-outanchors.
- Fix all configs before starting. Complete all pairedfolds before interpretingaggregate. No fold rerun for poor metrics. No post-result parameter sweep. Keep all results.

## Reporting and interpretation
Report per-specimen means first, mean/stdacrossfive specimens, all40pairedimageDicechanges,IoU/precision/recall,lowforeground andworst-case descriptively, trainingfitgap, and identicalanchorcomparisons. Pixel/crop correlation precludes claiming40independent biological replicates. Teacher-derived evaluation tests agreement, not complete ER accuracy. Crops are positive-selected and cannot establish whole-image false-positive control. Any positive result remains a candidate requiring stronger independent verification; do not auto-upgrade production based on this exploratory OOF set.

## Status
Data preparation and fold integrity checks PASS; training not started, no GPU spending. Next execution would be10shortmatchedfits ononeGPU, preservingresultsbeforecleanup. Do not silently reuse the earlier completed-run US$1 budget as a new unrestricted budget.

## Execution authorization — 2026-10-30
User requested the next step after preparation. Execute the fixed paired fivefold study on one Secure Cloud GPU with a new US$1 ceiling. No more experiments or retries for metrics. Verify downloads then terminate; no production promotion. Dataset archive contains only cropped Fluo and binary labels, not ER intensities or entire raw stacks.

## Completion — 2026-10-30
All10fixedfits completed without reruns or protocol changes. Expanded arm rejected; no production promotion. See [results report](../reports/ER_diverse_cv_20261030.md). All checkpoints preserved locally and temporary Pod deleted after verification.
