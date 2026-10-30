# ER diversity expansion — 2026-10-30

Prepared114newpairedcrops plus40preservedanchors =154crops fromfiveexisting specimens. No training orGPUuse. These are additional views, not additional biological replicates.

| Specimen | New crops |
|---|---:|
| cell_20260814_23298_NZ_50 | 21 |
| cell_20260814_23299_NZ_200 | 24 |
| cell_20260904_23961_NZ_100 | 22 |
| cell_20260904_23962_NZ_170 | 24 |
| cell_20260904_23965_NZ_140 | 23 |

Fixed density-stratified sampling and full provenance are in `outputs/er_diverse_roi_20261030/manifest.json`. All154file hashes/shapes/binarylabels checked; no exact duplicateFluoarrays; allfivefolds exclude the whole held-outspecimen. Original40anchors are identical between prospectivearms and evaluatedonceacrossfolds. Sample counts differ because insufficient nonoverlappingeligiblewindows are skipped.

Candidate backgrounds inherit the user crop-level convention but are not newmanualannotations. Visualreview shows retainedteacheromissions/thicklines andsomeweakFluocrops; do notclaimlabelqualityimprovedbyexpansion. Selection uses ERmaskdensity and isbiasedtowardpositiveareas, so even a successfulcropCV would not prove whole-image specificity.

Per-specimen `*_review.png` pairs showFluo(left) and inheritedlabeloverlay(right). No masks were altered based on appearance or earlier student outputs.

Locked plannedcomparison: [protocol](../docs/er_diverse_roi_cv_protocol_20261030.md). Same1000updates,network,loss,normalizationandthreshold; changeonlytrainingcroppool. Tenfits(fivepairedfolds), not yetstarted. Do not report CVperformance until run.
