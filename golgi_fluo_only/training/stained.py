"""Per-cell definitions shared by v5 training and by the local re-evaluation of v3/v4.
stained cell : a Cellpose cell containing >= MIN_PX pixels of the (cleaned) Golgi label.
S            : union of stained cells. Unstained cells are ignored (Golgi staining is incomplete).
Metrics inside S only: pixel Dice / precision / recall; and per stained cell, at threshold t:
  position = distance (px) between centroids of predicted and labelled Golgi in that cell,
  shape    = IoU of predicted vs labelled Golgi in that cell."""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_PX = 100


def tissue(f8):
    s = ndi.gaussian_filter(f8.astype(np.float32) / 255, 8); return s > threshold_otsu(s)


def stained(cells, y):
    cnt = np.bincount(cells[y].ravel(), minlength=int(cells.max()) + 1); cnt[0] = 0
    ids = np.nonzero(cnt >= MIN_PX)[0]; return ids, np.isin(cells, ids)


def weights(cells, y, rem, f8):
    """v5 training weights: stained cells (Golgi + rest of the cell), outside tissue, label pixels not in any cell."""
    ids, S = stained(cells, y); ext = ndi.binary_erosion(~tissue(f8), iterations=10)
    return ((S | ext | (y & (cells == 0))) & ~rem).astype(np.uint8), S, ids


def evaluate(cells, y, rem, pr, thr=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    ids, S = stained(cells, y); S = S & ~rem; yS = y & S; r = dict(n_stained_cells=int(len(ids)), pos=int(yS.sum()))
    if yS.any() and (S & ~y).any():
        r["auc_S"] = float(roc_auc_score(yS[S], pr[S])); r["ap_S"] = float(average_precision_score(yS[S], pr[S]))
    for t in thr:
        b = (pr >= t) & S; tp = int((b & yS).sum())
        r[f"dice_S_{t}"] = 2 * tp / max(b.sum() + yS.sum(), 1); r[f"prec_S_{t}"] = tp / max(b.sum(), 1); r[f"rec_S_{t}"] = tp / max(yS.sum(), 1)
        if t == 0.5 or t == 0.8:
            dist, iou, miss = [], [], 0
            for i in ids:
                C = cells == i; yc, bc = y & C & ~rem, b & C
                if not yc.any(): continue
                if not bc.any(): miss += 1; continue
                dist.append(float(np.hypot(*(np.array(ndi.center_of_mass(yc)) - np.array(ndi.center_of_mass(bc))))))
                iou.append(float((yc & bc).sum() / (yc | bc).sum()))
            r[f"cell_centroid_px_{t}"] = dist; r[f"cell_iou_{t}"] = iou; r[f"cell_missed_{t}"] = miss
    return r
