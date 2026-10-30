"""Analysis A: scale-resolved Fluo<->ER correspondence (label-free).

For 128x128 windows fully inside Fluo-defined tissue, compute band-limited
correlation (real cross-spectrum / sqrt(power product)) per spatial-period band.
Comparisons per specimen:
  F(z)~E(z)     main question: does Fluo co-vary with ER at this scale?
  E(z)~E(z+1)   positive reference: ER structure persistence across one Z step
  F(z)~F(z+1)   Fluo self-persistence
  F(z)~E_rot    null: ER window rotated 90 deg (no true correspondence)
Band-limited correlation in band b is the Pearson r of the two images after
band-pass filtering to b, pooled over windows. No masks or labels are used.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import BATCH, SPECIMENS, load_plane, sample_planes, tissue_mask

W = 128
BANDS = [(32, 1e9), (16, 32), (8, 16), (4, 8), (2.5, 4)]  # period in pixels
fy = np.fft.fftfreq(W)[:, None]
fx = np.fft.fftfreq(W)[None, :]
radius = np.sqrt(fy**2 + fx**2)
period = np.where(radius > 0, 1 / np.maximum(radius, 1e-9), np.inf)
band_masks = [(period >= lo) & (period < hi) & (radius > 0) for lo, hi in BANDS]
hann = np.outer(np.hanning(W), np.hanning(W)).astype(np.float32)


def spec(win):
    w = (win - win.mean()) / (win.std() + 1e-6)
    return np.fft.fft2(w * hann)


def windows(mask):
    out = []
    for y in range(0, mask.shape[0] - W + 1, W):
        for x in range(0, mask.shape[1] - W + 1, W):
            if mask[y:y + W, x:x + W].mean() >= 0.95:
                out.append((y, x))
    return out


def main(specs):
    results = {}
    for s in specs:
        acc = {k: np.zeros((len(BANDS), 3)) for k in ("FE", "EE1", "FF1", "FEnull")}
        nwin = 0
        for z in sample_planes(s):
            E, F = load_plane(s, z, "er"), load_plane(s, z, "fluo")
            E1, F1 = load_plane(s, z + 1, "er"), load_plane(s, z + 1, "fluo")
            m = tissue_mask(F)
            for y, x in windows(m):
                sl = (slice(y, y + W), slice(x, x + W))
                sF, sE, sE1, sF1 = spec(F[sl]), spec(E[sl]), spec(E1[sl]), spec(F1[sl])
                sEr = spec(np.rot90(E[sl]).copy())
                for key, a, b in (("FE", sF, sE), ("EE1", sE, sE1), ("FF1", sF, sF1), ("FEnull", sF, sEr)):
                    cross = (a * np.conj(b)).real
                    pa, pb = np.abs(a) ** 2, np.abs(b) ** 2
                    for i, bm in enumerate(band_masks):
                        acc[key][i] += (cross[bm].sum(), pa[bm].sum(), pb[bm].sum())
                nwin += 1
        r = {k: [float(v[i, 0] / np.sqrt(v[i, 1] * v[i, 2])) for i in range(len(BANDS))] for k, v in acc.items()}
        results[s] = {"batch": BATCH[s], "planes": sample_planes(s), "windows": nwin, "band_r": r}
        print(s, BATCH[s], "windows", nwin)
        for k in r:
            print("  %-7s" % k, " ".join("%+.3f" % v for v in r[k]))
        sys.stdout.flush()
    out = {"bands_period_px": BANDS, "results": results}
    json.dump(out, open("/home/claude/er_audit/out/a_spectral_coherence.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SPECIMENS))
