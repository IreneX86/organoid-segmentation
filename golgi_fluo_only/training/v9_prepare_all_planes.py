"""v9 local data prep for one spheroid (CPU). Requested 2026-11-10: train on (nearly) all planes instead of 16.
Outputs golgi_v9/data/v9_<cell>.npz:
  f8    (150,H,W) uint8   every Fluo plane, normalised p1-p99.8 -> uint8 (same norm8 as v4/v5; asserted identical to the
                          v5 fluo8 inputs of the 16 teacher planes and the manual planes)
  zg    odd planes 1,3,...,149 (75 planes; includes the 16 original teacher planes 15,23,...,135)
  g     (75,H,W) uint16   raw Golgi channel of those planes (teacher input on the GPU)
  inB, big3d  packbits (75,H,W)  Golgi-channel 3D objects (same rule as v4 prep): pixel in an object / in an object that
                          reaches >=140 px in some plane  -> lets the GPU apply the 3D-noise cleaning rule to new planes
and lab_<cell>.npz: the existing cleaned labels of the 16 teacher planes + manual planes (copied from v5 data)."""
import glob, os, shutil, sys, tarfile, time
import numpy as np, tifffile
from scipy import ndimage as ndi

c = sys.argv[1]; C = f"cell_{c}_NZ_150"
X = "/mnt/user-data/uploads/Downloads/_golgi_xfer9"; W = "/home/claude/golgi_v9"; RAW = f"{W}/raw"
os.makedirs(f"{W}/data", exist_ok=True); os.makedirs(RAW, exist_ok=True); t0 = time.time()
parts = sorted(glob.glob(f"{X}/g{c}.pa??")); assert parts
with open(f"{RAW}/{c}.tgz", "wb") as o:
    for p in parts: shutil.copyfileobj(open(p, "rb"), o)
with tarfile.open(f"{RAW}/{c}.tgz", "r:gz") as t: t.extractall(RAW, filter="data")
os.remove(f"{RAW}/{c}.tgz")
NZ = 150
G = np.stack([tifffile.imread(f"{RAW}/{C}/Golgi/mask_{z:03d}.tif") for z in range(1, NZ + 1)])
F = np.stack([tifffile.imread(f"{RAW}/{C}/fluo/fluo_{z:03d}.tif") for z in range(1, NZ + 1)])
H, Wd = G.shape[1:]


def norm8(x):
    x = x.astype(np.float32); lo, hi = np.percentile(x, [1.0, 99.8])
    return (np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


F8 = np.stack([norm8(f) for f in F]); del F
d5 = np.load(f"/home/claude/golgi_v5/payload/data/{C}.npz")
g7 = np.load(f"/home/claude/golgi_v7/payload/data/golgi_{C}.npz")
for pre in ("", "man_"):
    if pre + "z" not in d5.files: continue
    for k, z in enumerate(int(v) for v in d5[pre + "z"]):
        idx = [min(max(z + o, 1), NZ) - 1 for o in range(-3, 4)]
        assert np.abs(d5[pre + "fluo8"][k].astype(np.int16) - F8[idx]).mean() < 0.05, (pre, z)
        assert np.array_equal(g7[pre + "golgi"][k], G[z - 1]), ("golgi", pre, z)
# 3D objects of the Golgi channel (v4 rule), plus the existing labels
B = np.zeros(G.shape, bool)
for k in range(NZ):
    g = G[k].astype(np.float32); B[k] = ndi.binary_opening(ndi.gaussian_filter(g, 1) > ndi.gaussian_filter(g, 15) * 1.5)
for pre in ("", "man_"):
    if pre + "z" not in d5.files: continue
    zs = [int(v) for v in d5[pre + "z"]]; n = len(zs)
    YO = np.unpackbits(d5[pre + "y_orig"])[:n * H * Wd].reshape(n, H, Wd).astype(bool)
    for k, z in enumerate(zs): B[z - 1] |= YO[k]
L3, n3 = ndi.label(B, structure=np.ones((3, 3, 3), bool))
maxA = np.zeros(n3 + 1, np.int64)
for k in range(NZ): maxA = np.maximum(maxA, np.bincount(L3[k].ravel(), minlength=n3 + 1))
big = maxA >= 140; big[0] = False
ZG = np.arange(1, NZ + 1, 2)
inB = np.stack([L3[z - 1] > 0 for z in ZG]); big3d = np.stack([big[L3[z - 1]] for z in ZG]); del L3, B
np.savez_compressed(f"{W}/data/v9_{C}.npz", f8=F8, zg=ZG, g=G[ZG - 1], inB=np.packbits(inB), big3d=np.packbits(big3d), shape=np.array([H, Wd]))
shutil.rmtree(f"{RAW}/{C}")
np.savez_compressed(f"{W}/data/lab_{C}.npz", **{k: d5[k] for k in d5.files if "fluo8" not in k})
print(c, "ok", f"{time.time()-t0:.0f}s", flush=True)
