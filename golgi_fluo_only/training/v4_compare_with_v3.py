"""Evaluate the v3 C7 held-out probability maps (z071, z103 per spheroid; 14 manual planes) against the v4 CLEANED labels,
with the same metrics as the v4 GPU script, so v3 and v4 can be compared on identical planes and labels."""
import glob, json, os, sys
import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score

V3 = os.environ.get("GOLGI_V3_OUT", "golgi_v3/results/output"); D4 = os.environ.get("GOLGI_V4_DATA", "golgi_v4/data")
PRED = sys.argv[1] if len(sys.argv) > 1 else "C7"          # "C7" (v3) or a v4 output dir containing Z12_* files
THR = [0.3, 0.4, 0.5, 0.6]


def dice(a, b): return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def evaluate(y, neg, rem, pr):
    near = ndi.distance_transform_edt(~y) <= 40 if y.any() else np.zeros_like(y); L = y | neg; r = dict(pos=int(y.sum()))
    if y.any() and (L & ~y).any():
        r["auc"] = float(roc_auc_score(y[L], pr[L])); r["ap"] = float(average_precision_score(y[L], pr[L]))
    for t in THR:
        b = pr >= t; r[f"dice_near_{t}"] = dice(b & near, y); r[f"dice_whole_{t}"] = dice(b & ~rem, y)
        r[f"fp_on_neg_{t}"] = float((b & neg).sum() / max(neg.sum(), 1))
    return r


def probs(c, z, manual):
    if PRED == "C7":
        f = f"{V3}/C7_manual_prob_{c}_z{z:03d}.npy" if manual else f"{V3}/C7_prob_{c}.npz"
        return (np.load(f) if manual else np.load(f)[f"z{z:03d}"]) / 255.
    f = f"{PRED}/Z12_manual_prob_{c}_z{z:03d}.npy" if manual else f"{PRED}/Z12_prob_{c}.npz"
    return (np.load(f) if manual else np.load(f)[f"z{z:03d}"]) / 255.


rows, man = [], []
for p in sorted(glob.glob(f"{D4}/cell_*_NZ_150.npz")):
    c = p.split("/")[-1][:-4]; d = np.load(p); H, W = [int(v) for v in d["shape"]]
    for pre, dest, zsel in (("", rows, (71, 103)), ("man_", man, None)):
        if pre + "z" not in d.files: continue
        zs = [int(z) for z in d[pre + "z"]]; n = len(zs)
        up = lambda k: np.unpackbits(d[pre + k])[:n * H * W].reshape(n, H, W).astype(bool)
        Y, NG, RM, YO = up("y"), up("neg"), up("rem"), up("y_orig")
        for k, z in enumerate(zs):
            if zsel and z not in zsel: continue
            pr = probs(c, z, pre == "man_"); r = evaluate(Y[k], NG[k], RM[k], pr); r.update(cell=c, z=z)
            r["dice_whole_orig_0.5"] = dice(pr >= 0.5, YO[k]); dest.append(r)
m = lambda R, k: float(np.mean([r[k] for r in R if k in r]))
keys = ["auc", "ap"] + [f"{a}_{t}" for a in ("dice_near", "dice_whole", "fp_on_neg") for t in THR] + ["dice_whole_orig_0.5"]
out = dict(pred=PRED, n_planes=len(rows), n_manual=len(man), teacher={k: m(rows, k) for k in keys}, manual={k: m(man, k) for k in keys},
           per_plane=rows, per_manual=man)
json.dump(out, open(f"eval_{'C7' if PRED == 'C7' else 'Z12'}_clean.json", "w"), indent=1)
print(PRED, len(rows), len(man)); print("teacher-planes", {k: round(v, 3) for k, v in out["teacher"].items()}); print("manual", {k: round(v, 3) for k, v in out["manual"].items()})
