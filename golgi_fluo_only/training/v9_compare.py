"""Local comparison v3 / v7 / v9 on identical held-out planes (pooled Dice within 40 px of the cleaned labels),
plus Dice inside stained cells (stained.py) and 6-px tolerance F1."""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "/home/claude/golgi_v5/payload/code"); import stained as ST

B = "/home/claude"
SRC = {"v3": (f"{B}/golgi_v3/results/output", "C7"), "v7": (f"{B}/golgi_v7/results/output", "V7"), "v9": (f"{B}/golgi_v9/results/output", "V9")}
THR = (0.3, 0.4, 0.5, 0.6, 0.7)


def load(m, c, z, man):
    d, p = SRC[m]
    return (np.load(f"{d}/{p}_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{d}/{p}_prob_{c}.npz")[f"z{z:03d}"]) / 255.


def tolf1(y, p, tol=6):
    if not y.any() and not p.any(): return None
    dy = ndi.distance_transform_edt(~y) if y.any() else np.full(y.shape, 1e9); dp = ndi.distance_transform_edt(~p) if p.any() else np.full(y.shape, 1e9)
    return (p & (dy <= tol)).sum(), p.sum(), (y & (dp <= tol)).sum(), y.sum()


acc = {m: {s: {t: [0, 0] for t in THR} for s in ("teacher", "manual")} for m in SRC}
tol = {m: {s: [0, 0, 0, 0] for s in ("teacher", "manual")} for m in SRC}
stn = {m: {s: [] for s in ("teacher", "manual")} for m in SRC}
for i in range(18):
    c = f"cell_{16471 + i}_NZ_150"; d = np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H, W = [int(v) for v in d["shape"]]
    T = np.load(f"{B}/golgi_v5/cells_v2/cells/{c}.npz")
    for pre, key, sec in (("", "t", "teacher"), ("man_", "m", "manual")):
        if pre + "z" not in d.files: continue
        zs = [int(z) for z in d[pre + "z"]]; n = len(zs)
        up = lambda k: np.unpackbits(d[pre + k])[:n * H * W].reshape(n, H, W).astype(bool)
        Y, RM = up("y"), up("rem")
        for k, z in enumerate(zs):
            if sec == "teacher" and z not in (71, 103): continue
            near = (ndi.distance_transform_edt(~Y[k]) <= 40) & ~RM[k]; yv = Y[k] & near
            for m in SRC:
                pr = load(m, c, z, sec == "manual")
                for t in THR:
                    b = (pr >= t) & near; acc[m][sec][t][0] += 2 * (b & yv).sum(); acc[m][sec][t][1] += b.sum() + yv.sum()
                r = tolf1(yv, (pr >= 0.5) & near)
                if r: tol[m][sec] = [a + b for a, b in zip(tol[m][sec], r)]
                stn[m][sec].append(ST.evaluate(T[key][k], Y[k], RM[k], pr))
out = {}
for m in SRC:
    out[m] = {}
    for s in ("teacher", "manual"):
        a, pp, b, yy = tol[m][s]; P, R = a / max(pp, 1), b / max(yy, 1)
        out[m][s] = dict(dice_near={t: round(acc[m][s][t][0] / max(acc[m][s][t][1], 1), 4) for t in THR},
                         f1_6px_at0p5=round(2 * P * R / max(P + R, 1e-9), 4),
                         dice_stained={t: round(float(np.mean([r[f"dice_S_{t}"] for r in stn[m][s]])), 4) for t in (0.5, 0.6)})
json.dump(out, open("/home/claude/golgi_v9/compare.json", "w"), indent=1)
for m in out:
    for s in out[m]: print(m, s, out[m][s])
