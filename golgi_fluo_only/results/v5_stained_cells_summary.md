# Golgi from Fluo — evaluation inside stained cells, and v5 per-cell labels (2026-11-06)

Labels: teacher PSEUDO-labels cleaned as in v4 (not manual GT); "manual" = 14 manual masks cleaned the same way.
Cells: Cellpose CPSAM (generalist, Fluo only) finds NUCLEI; cell territory = watershed from each nucleus over smoothed
Fluo, inside tissue, <= 50 px beyond the nucleus. Stained cell = territory with >= 100 px of Golgi label
(~10 per plane; 62% of label pixels fall inside them). All metrics below are computed ONLY inside stained cells,
on the same 36 held-out planes (z071, z103 of each spheroid) + 14 manual planes for every model.

| inside stained cells | v3 C7 | v4 | v5 per-cell | perinuclear ring baseline |
|---|---:|---:|---:|---:|
| Dice (best threshold), teacher planes | 0.226 | 0.205 | 0.205 | 0.054 |
| precision / recall at that threshold | 0.17 / 0.42 | 0.14 / 0.47 | 0.17 / 0.32 | — |
| Dice (best threshold), manual planes | 0.293 | 0.223 | 0.226 | 0.051 |
| per-cell Golgi centroid error, median px (thr 0.5) | 57 | 57 | 63 | — |
| per-cell IoU, median (thr 0.5) | 0.14 | 0.09 | 0.11 | — |
| AUC inside stained cells | 0.90 | 0.87 | 0.87 | — |

Conclusions
1. Scores are low even where the Golgi is known to be stained: most predicted pixels in a stained cell are not Golgi
   (precision ~15%). Incomplete staining is not the main reason for the low Dice.
2. Training only inside stained cells (v5) did not help; v3 C7 remains best.
3. The models are far better than a "ring around the nucleus" guess (0.22 vs 0.05), so Fluo does carry some Golgi
   information, but not enough for accurate shape (per-cell IoU ~0.1-0.15) or precise position (~60 px error).
Run: RTX 4090, Cellpose ~1 min, training 3 folds ~25 min, $1.73 (includes one aborted run with nucleus-only masks).
