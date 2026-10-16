# Nuc validation by input TIFF modification-time cohort

This is a **read-only regrouping of the frozen 133/34 internal split and archived baseline per-image metrics**. The 167 local input TIFFs all match the split manifest SHA256 values. Cohorts use the input files’ **filesystem modification month in UTC**; these dates can reflect export or copying and do not establish biological source, acquisition date, or Z order.

| Input TIFF mtime | Total | Train | Validation | Nonempty validation | All-black controls (train / val) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oct 2025 | 10 | 9 | 1 | 1 | 0 / 0 |
| Nov 2025 | 148 | 116 | 32 | 31 | 3 / 1 |
| May 2026 | 9 | 8 | 1 | 1 | 0 / 0 |

The following are **arithmetic means of full-image, per-image scores on nonempty validation masks**. Oct 2025 is only `nuc_038.tif`; May 2026 is only `nuc_163.tif`.

| Mtime cohort | Nonempty n | Dice at fixed 0.50 | Precision at 0.50 | Recall at 0.50 | Dice at selected 0.45 | Precision at 0.45 | Recall at 0.45 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oct 2025 | 1 | 0.948750 | 0.931291 | 0.966876 | 0.947764 | 0.927463 | 0.968974 |
| Nov 2025 | 31 | 0.958377 | 0.965016 | 0.951960 | 0.958430 | 0.961647 | 0.955343 |
| May 2026 | 1 | 0.904463 | 0.916559 | 0.892683 | 0.905855 | 0.909968 | 0.901779 |

All four confirmed all-black controls (`nuc_002`–`nuc_005`) have Nov 2025 input mtimes. `nuc_002`–`004` are in training; the only validation control, `nuc_005`, had **2,744** false-positive pixels at 0.50 and **3,093** at 0.45. Empty-mask Dice, precision, and recall are excluded from the table.

These are **not cross-cohort tests**: all three mtime cohorts occur in both partitions. The Nov 2025 cohort accounts for 31 of 33 nonempty validation images and, in the [prior provenance audit](nuc_provenance_followup_20261016.md), exactly matches a 148-image high-similarity component. The apparent May-versus-Nov Dice difference rests on one May image and can reflect image content, annotation, or related views rather than a cohort effect. The model checkpoint and 0.45 threshold were selected on this same validation set. Independent source generalization remains unmeasured.

Sources: [frozen split](../experiments/results/nuc_internal_split_v1.json), archived `runpod_results/nuc_internal_20261015/full_results/model/validation_per_image.csv`, archived `local_verification.json` (passed), and current read-only input TIFF mtimes. [Machine-readable summary](assets/nuc_export_cohort_validation_20261016.json) records exact values, mtime ranges, and source hashes. No raw TIFFs, masks, split, checkpoint, or baseline metrics were changed.
