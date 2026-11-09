# v6: v3 setup with "clean" positives (2026-11-09)
Positives = old `Golgi label` folder mask AND cleaned teacher pseudo-label (34% of teacher pixels);
teacher-only / old-only Golgi pixels ignored; everything else as v3 C7 (Fluo z-3..z+3, same U-Net, 3 folds, 5000 steps).
RTX 4090, ~45 min, about $0.7. Labels are pseudo-labels; manual = 14 cleaned manual masks.

Same 36 held-out planes + 14 manual planes, best threshold for each model:
| | v3 | v6 |
|---|---:|---:|
| Dice near Golgi, teacher labels | 0.380 | 0.335 |
| Dice near Golgi, manual masks | 0.386 | 0.342 |
| Dice inside stained cells, manual | 0.290 | 0.231 |
| F1 with 6 px tolerance, manual | 0.634 | 0.577 |
| Precision near Golgi at best thr (teacher) | 0.30 | 0.31 |
Conclusion: cleaner positives did not raise precision; recall fell, so Dice fell. The false positives come from
Golgi-like Fluo texture, not from label noise. v3 remains the best model.
