"""v4 local data prep for one spheroid (runs on CPU in the container, no GPU needed).
Inputs : full Golgi + Fluo z-stacks of the spheroid (150 planes), v3 teacher pseudo-labels (16 planes), manual14.
Outputs: golgi_v4/data/cell_X.npz with
  fluo8  (16,7,H,W) uint8   Fluo planes z-12,-8,-4,0,+4,+8,+12 (clamped), each normalised p1-p99.8 -> uint8
  y      cleaned teacher label        y_orig  original teacher label      rem  pieces removed as artefact/noise
  neg    confident negatives (nuclei interior, outside tissue, 3-15 px ring around kept Golgi)
  (all masks np.packbits over (16,H,W)); plus the same for manual planes of this spheroid (man_*).
Cleaning rules (agreed with the user 2026-11-05):
  membrane : area>=300 px, solidity<0.40, max half-thickness<7 px           (thin curved arcs along cell membrane)
  diffuse  : area>=200 px, Golgi contrast vs 8-px ring <1.25                  (shapeless haze)
  noise 3D : the 3D object (Golgi channel, 26-connected over all planes) never exceeds 140 px in any plane
"""
import glob, json, os, sys, tarfile, gzip, shutil, time
import numpy as np, tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

c = sys.argv[1]; C = f"cell_{c}_NZ_150"
# Paths are configurable: GOLGI_XFER = folder with g<cell>.pa? archive parts (tar.gz of <cell>/Golgi and <cell>/fluo),
# GOLGI_WORK = v4 work folder, GOLGI_V3 = v3 folder (teacher_masks.npz, payload/data).
X = os.environ.get("GOLGI_XFER", "xfer"); W = os.environ.get("GOLGI_WORK", "golgi_v4"); V3 = os.environ.get("GOLGI_V3", "golgi_v3"); RAW = f"{W}/raw"
os.makedirs(f"{W}/data", exist_ok=True); os.makedirs(RAW, exist_ok=True)
t0 = time.time()
# ---- unpack
parts = sorted(glob.glob(f"{X}/g{c}.pa?"))
with open(f"{RAW}/{c}.tgz", "wb") as o:
    for p in parts:
        shutil.copyfileobj(open(p, "rb"), o)
with tarfile.open(f"{RAW}/{c}.tgz", "r:gz") as t:
    t.extractall(RAW, filter="data")
os.remove(f"{RAW}/{c}.tgz")
NZ = len(glob.glob(f"{RAW}/{C}/Golgi/mask_*.tif")); assert NZ == len(glob.glob(f"{RAW}/{C}/fluo/fluo_*.tif")) == 150, NZ
G = np.stack([tifffile.imread(f"{RAW}/{C}/Golgi/mask_{z:03d}.tif") for z in range(1, NZ + 1)])        # (150,H,W) uint16, index z-1
F = np.stack([tifffile.imread(f"{RAW}/{C}/fluo/fluo_{z:03d}.tif") for z in range(1, NZ + 1)])
H, Wd = G.shape[1:]
print(c, "loaded", G.shape, f"{time.time()-t0:.0f}s", flush=True)

# ---- labels to clean
T = np.load(f"{V3}/results/output/teacher_masks.npz")
d3 = np.load(f"{V3}/payload/data/{C}.npz"); ZT = [int(z) for z in d3["z"]]
assert np.array_equal(d3["golgi"][0], G[ZT[0] - 1]), "Golgi plane mismatch vs v3 payload"
unpack = lambda a: np.unpackbits(a)[:H * Wd].reshape(H, Wd).astype(bool)
labs = {("t", z): unpack(T[f"{C}_z{z:03d}"]) for z in ZT}
M = np.load(f"{V3}/payload/data/manual14.npz"); ZM = [int(M["z"][i]) for i in range(len(M["z"])) if str(M["cell"][i]) == C]
for i in range(len(M["z"])):
    if str(M["cell"][i]) == C:
        assert np.array_equal(M["golgi"][i], G[int(M["z"][i]) - 1]); labs[("m", int(M["z"][i]))] = M["manual"][i] > 0

# ---- Golgi-channel 3D objects
B = np.zeros(G.shape, bool)
for k in range(NZ):
    g = G[k].astype(np.float32)
    B[k] = ndi.binary_opening(ndi.gaussian_filter(g, 1) > ndi.gaussian_filter(g, 15) * 1.5)
for (_, z), Y in labs.items():
    B[z - 1] |= Y
L3, n3 = ndi.label(B, structure=np.ones((3, 3, 3), bool)); del B
maxA = np.zeros(n3 + 1, np.int64)
for k in range(NZ):
    maxA = np.maximum(maxA, np.bincount(L3[k].ravel(), minlength=n3 + 1))
print(c, "3D objects", n3, f"{time.time()-t0:.0f}s", flush=True)


def norm8(x):
    x = x.astype(np.float32); lo, hi = np.percentile(x, [1.0, 99.8])
    return (np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


OFF = [-12, -8, -4, 0, 4, 8, 12]
rows = []


def process(kind, z, Y):
    g = G[z - 1].astype(np.float32); f8 = np.stack([norm8(F[min(max(z + o, 1), NZ) - 1]) for o in OFF])
    fn = f8[3].astype(np.float32) / 255
    L = label(Y); DT = ndi.distance_transform_edt(Y); rem = np.zeros_like(Y)
    for p in regionprops(L):
        sl = tuple(slice(max(s.start - 10, 0), s.stop + 10) for s in p.slice); P = L[sl] == p.label
        ring = (ndi.distance_transform_edt(~P) <= 8) & ~Y[sl]
        gcon = float(g[sl][P].mean() / max(g[sl][ring].mean(), 1e-3)) if ring.any() else 9.0
        maxdt = float(DT[sl][P].max()); solid = float(p.solidity) if p.area >= 3 else 1.0
        ids = np.unique(L3[z - 1][sl][P]); ids = ids[ids > 0]; m3 = int(maxA[ids].max()) if len(ids) else int(p.area)
        mem = p.area >= 300 and solid < 0.40 and maxdt < 7.0
        dif = p.area >= 200 and gcon < 1.25
        small = m3 < 140
        if mem or dif or small:
            rem[sl] |= P
        rows.append(dict(cell=c, kind=kind, z=z, lab=int(p.label), area=int(p.area), solid=round(solid, 3), maxdt=round(maxdt, 2),
                         gcon=round(gcon, 3), max3d=m3, mem=bool(mem), dif=bool(dif), small=bool(small)))
    y = Y & ~rem
    s = ndi.gaussian_filter(fn, 8); tis = s > threshold_otsu(s)
    ext = ndi.binary_erosion(~tis, iterations=10)
    s4 = ndi.gaussian_filter(fn, 4); dark = (s4 < np.percentile(s4[tis], 25)) & tis
    dark = ndi.binary_opening(dark, iterations=6); Ld = label(dark); a = np.bincount(Ld.ravel()); a[0] = 0
    nuc = ndi.binary_erosion(ndi.binary_fill_holes(a[Ld] >= 2000), iterations=5)
    dy = ndi.distance_transform_edt(~y) if y.any() else np.full(Y.shape, 1e9)
    ring = (dy > 3) & (dy <= 15) & tis
    safe = ndi.distance_transform_edt(~Y) > 3            # keep away from any original label, kept or removed
    neg = (ring | nuc | ext) & safe & ~y
    return f8, y, rem, neg, nuc


out = {}
for kind, zs in (("t", ZT), ("m", ZM)):
    res = [process(kind, z, labs[(kind, z)]) for z in zs]
    if not res:
        continue
    pre = "" if kind == "t" else "man_"
    out[pre + "z"] = np.array(zs); out[pre + "fluo8"] = np.stack([r[0] for r in res])
    for j, nm in enumerate(["y", "rem", "neg", "nuc"], 1):
        out[pre + nm] = np.packbits(np.stack([r[j] for r in res]))
    out[pre + "y_orig"] = np.packbits(np.stack([labs[(kind, z)] for z in zs]))
out["shape"] = np.array([H, Wd])
np.savez_compressed(f"{W}/data/{C}.npz", **out)
json.dump(rows, open(f"{W}/data/{C}_pieces.json", "w"))
shutil.rmtree(f"{RAW}/{C}")
tp = [r for r in rows if r["kind"] == "t"]; tot = sum(r["area"] for r in tp)
f = lambda k: sum(r["area"] for r in tp if r[k])
print(c, f"done {time.time()-t0:.0f}s | teacher px {tot} removed: membrane {f('mem')} diffuse {f('dif')} noise3D {f('small')} "
      f"total {sum(r['area'] for r in tp if r['mem'] or r['dif'] or r['small'])} ({100*sum(r['area'] for r in tp if r['mem'] or r['dif'] or r['small'])/max(tot,1):.1f}%)", flush=True)
