# v8 = v7 + spatial augmentation (rotation 0-360, scale 0.85-1.15), 320 px patches, 24000 steps (2026-11-09)
Stopped after fold 0 (of 3) because it gave no gain over v7; this saved ~1.5 h of GPU (~$1.4). Fold-0 cost ~$1.0.
Fold 0 held-out spheroids (all 16 planes each), Dice within 40 px of teacher labels at threshold 0.5: v8 0.442, v7 0.444.
Same planes as the v3 comparison (fold-0 spheroids, z071/z103 + manual planes of 16471), threshold 0.5:
teacher labels v3 0.374 / v7 0.378 / v8 0.377; manual masks v3 0.380 / v7 0.407 / v8 0.391.
Conclusion: larger patches, spatial augmentation and longer training do not add to v7. v7 remains the best model.
