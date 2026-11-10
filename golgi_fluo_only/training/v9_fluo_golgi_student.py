"""Fluo -> Golgi student U-Net, v9 (GPU, TensorFlow). Requested 2026-11-10: "use all planes" to raise Dice.
Identical to v7 (soft teacher targets, base-32 U-Net, Fluo z-3..z+3, 256 px patches, batch 16, 20000 steps cosine LR,
intensity augmentation, 8-fold TTA, same 3 folds by spheroid), except the TRAINING planes:
  v7: 16 planes per spheroid (z = 15, 23, ..., 135)
  v9: 75 planes per spheroid (every odd z = 1, 3, ..., 149; includes the 16)
For the 16 original planes the existing cleaned labels are used unchanged. For the 59 new planes the teacher mask
(prob >= 0.5) is cleaned with the same v4 rules (membrane: area>=300 & solidity<0.40 & max half-thickness<7;
diffuse: area>=200 & Golgi contrast vs 8-px ring <1.25; 3D noise: 3D Golgi object never >=140 px in any plane).
Evaluation is unchanged: the 16 original planes of each held-out spheroid (cleaned teacher labels) + 14 manual planes.
Labels remain pseudo-labels."""
import glob, hashlib, json, os, threading, queue, time
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from tensorflow.keras import Model, layers
import stained as ST

DATA = "payload/data"; OUT = "output"; os.makedirs(OUT, exist_ok=True)
STEPS, BATCH, LR, PATCH, SEED = int(os.environ.get("STEPS", 20000)), 16, 3e-4, 256, 20261110
THRS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TEACHER_SHA = "6f248ca44be752ccef3425e8cf7c2f953c2b1575f9eda7e719372ae83a263772"
NZ = 150


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(f"{OUT}/progress.txt", "a").write(s + "\n")


def starts(n, t, s):
    if n <= t: return [0]
    st = list(range(0, n - t + 1, s)); st += [n - t] if st[-1] != n - t else []
    return st


W1 = np.hanning(512).astype(np.float32); WB = np.outer(W1, W1); WB /= WB.max(); WB = np.maximum(WB, 1e-3)


def predict(model, x, tile=512, overlap=128):
    H, W = x.shape[:2]; P = np.zeros((H, W), np.float32); Wt = np.zeros((H, W), np.float32)
    co = [(y, z) for y in starts(H, tile, tile - overlap) for z in starts(W, tile, tile - overlap)]
    pr = np.asarray(model.predict(np.stack([x[y:y + tile, z:z + tile] for y, z in co]).astype(np.float32), batch_size=4, verbose=0), np.float32)[..., 0]
    for (y, z), p in zip(co, pr):
        P[y:y + tile, z:z + tile] += p * WB; Wt[y:y + tile, z:z + tile] += WB
    return np.clip(P / Wt, 0, 1)


def predict_tta(model, x):
    acc = np.zeros(x.shape[:2], np.float32)
    for k in range(4):
        for fl in (False, True):
            xi = np.rot90(x, k, axes=(0, 1)); xi = xi[:, ::-1] if fl else xi
            p = predict(model, np.ascontiguousarray(xi)); p = p[:, ::-1] if fl else p
            acc += np.rot90(p, -k, axes=(0, 1))
    return acc / 8


def norm01(a):
    a = a.astype(np.float32); lo, hi = np.percentile(a, [1, 99.8]); return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def block(x, f, d=0.0):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x); x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x); return layers.Dropout(d)(x) if d > 0 else x


def build_unet(cin, b=32):
    i = tf.keras.Input((None, None, cin))
    e1 = block(i, b); e2 = block(layers.MaxPooling2D()(e1), 2 * b); e3 = block(layers.MaxPooling2D()(e2), 4 * b)
    e4 = block(layers.MaxPooling2D()(e3), 8 * b); bt = block(layers.MaxPooling2D()(e4), 16 * b, 0.4)
    up = lambda x: layers.UpSampling2D(interpolation="bilinear")(x)
    d4 = block(layers.Concatenate()([up(bt), e4]), 8 * b, 0.3); d3 = block(layers.Concatenate()([up(d4), e3]), 4 * b, 0.2)
    d2 = block(layers.Concatenate()([up(d3), e2]), 2 * b, 0.1); d1 = block(layers.Concatenate()([up(d2), e1]), b)
    return Model(i, layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(d1))


def masked_loss(y, w, p):
    p = tf.clip_by_value(tf.cast(p, tf.float32), 1e-6, 1 - 1e-6); n = tf.reduce_sum(w) + 1e-6
    bce = tf.reduce_sum(w * -(y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))) / n
    dice = (2 * tf.reduce_sum(w * y * p) + 1) / (tf.reduce_sum(w * y) + tf.reduce_sum(w * p) + 1)
    return bce + 1 - dice


def zidx(z): return [min(max(z + o, 1), NZ) - 1 for o in range(-3, 4)]


def clean(args):
    """v4 cleaning rules applied to a binarised teacher mask Y of a new plane -> removed pieces."""
    Y, g, inB, big3d = args; g = g.astype(np.float32)
    L = label(Y); DT = ndi.distance_transform_edt(Y); rem = np.zeros_like(Y)
    for p in regionprops(L):
        sl = tuple(slice(max(s.start - 10, 0), s.stop + 10) for s in p.slice); P = L[sl] == p.label
        ring = (ndi.distance_transform_edt(~P) <= 8) & ~Y[sl]
        gcon = float(g[sl][P].mean() / max(g[sl][ring].mean(), 1e-3)) if ring.any() else 9.0
        maxdt = float(DT[sl][P].max()); solid = float(p.solidity) if p.area >= 3 else 1.0
        small = (not big3d[sl][P].any()) if inB[sl][P].any() else p.area < 140
        if (p.area >= 300 and solid < 0.40 and maxdt < 7.0) or (p.area >= 200 and gcon < 1.25) or small:
            rem[sl] |= P
    return rem


def weights(f_center, Y, RM, t):
    tis = ST.tissue(f_center); ext = ndi.binary_erosion(~tis, iterations=10); anyg = Y | (t >= 0.5)
    region = (ndi.distance_transform_edt(~anyg) <= 40) & tis if anyg.any() else np.zeros_like(tis)
    return ((region | ext) & ~RM).astype(np.uint8)


# ---------------- teacher soft targets on all 75 planes (GPU) + cleaning of the new planes (CPU pool)
tp = f"{DATA}/golgi_unet.keras"; assert hashlib.sha256(open(tp, "rb").read()).hexdigest() == TEACHER_SHA, "teacher SHA mismatch"
teacher = tf.keras.models.load_model(tp, compile=False)
cells = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f"{DATA}/lab_cell_*.npz")); assert os.environ.get("MOCK") or len(cells) == 18, cells
if os.environ.get("MOCK"): cells = cells[:int(os.environ.get("MOCK"))]
F8, D, EV, MAN, agree, stats = {}, {}, {}, [], [], []
pool_ = Pool(min(16, os.cpu_count() or 4))
for c in cells:
    v = np.load(f"{DATA}/v9_{c}.npz"); lab = np.load(f"{DATA}/lab_{c}.npz"); H, W = [int(x) for x in v["shape"]]
    F8[c] = v["f8"]; zg = [int(z) for z in v["zg"]]; G = v["g"]; n = len(zg)
    inB = np.unpackbits(v["inB"])[:n * H * W].reshape(n, H, W).astype(bool); big3d = np.unpackbits(v["big3d"])[:n * H * W].reshape(n, H, W).astype(bool)
    z16 = [int(z) for z in lab["z"]]; n16 = len(z16)
    up = lambda k, m: np.unpackbits(lab[k])[:m * H * W].reshape(m, H, W).astype(bool)
    Y16, RM16, YO16 = up("y", n16), up("rem", n16), up("y_orig", n16)
    T = [predict(teacher, norm01(G[k])[..., None]) for k in range(n)]
    new = [k for k, z in enumerate(zg) if z not in z16]
    rems = dict(zip(new, pool_.map(clean, [(T[k] >= 0.5, G[k], inB[k], big3d[k]) for k in new])))
    planes, ev = [], []
    for k, z in enumerate(zg):
        t = T[k]; fc = F8[c][z - 1]
        if z in z16:
            j = z16.index(z); Y, RM = Y16[j], RM16[j]
            agree.append(2 * ((t >= 0.5) & YO16[j]).sum() / max((t >= 0.5).sum() + YO16[j].sum(), 1))
            near = (ndi.distance_transform_edt(~Y) <= 40) & ~RM if Y.any() else np.zeros_like(Y)
            ev.append(dict(cell=c, z=z, y=Y, near=near))
        else:
            Y0 = t >= 0.5; RM = rems[k]; Y = Y0 & ~RM; stats.append((int(Y0.sum()), int(RM.sum())))
        w = weights(fc, Y, RM, t)
        planes.append(dict(cell=c, z=z, t=t.astype(np.float16), yi=np.nonzero(Y[::2, ::2]), wi=np.nonzero(w[::4, ::4]), w=w, hasy=bool(Y.any())))
    D[c] = planes; EV[c] = ev
    if "man_z" in lab.files:
        zm = [int(z) for z in lab["man_z"]]; m = len(zm); MY, MR = up("man_y", m), up("man_rem", m)
        for j, z in enumerate(zm):
            near = (ndi.distance_transform_edt(~MY[j]) <= 40) & ~MR[j] if MY[j].any() else np.zeros_like(MY[j])
            MAN.append(dict(cell=c, z=z, y=MY[j], near=near))
    del T, G, inB, big3d
    log("loaded", c, "planes", len(planes), "teacher-vs-saved-mask Dice (16 orig planes) %.3f" % np.mean(agree[-n16:]))
pool_.close(); del teacher
sY = sum(a for a, _ in stats); sR = sum(b for _, b in stats)
log("new planes", len(stats), "teacher px", sY, "removed by cleaning %.1f%%" % (100 * sR / max(sY, 1)), "| manual planes", len(MAN),
    "| mean agreement %.4f" % np.mean(agree))
assert os.environ.get("MOCK") or np.mean(agree) > 0.95, "teacher re-run does not reproduce the saved pseudo-labels"
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)


def full_input(c, z): return np.ascontiguousarray(np.moveaxis(F8[c][zidx(z)], 0, -1)).astype(np.float32) / 255


def make_batch(r):
    X, Y, Wt = [], [], []
    for _ in range(BATCH):
        p = TRAIN[r.integers(len(TRAIN))]; H, Wd = p["w"].shape
        if r.random() < 0.6 and p["hasy"]:
            ys, xs = p["yi"]; j = r.integers(len(ys)); cy, cx = ys[j] * 2, xs[j] * 2
        else:
            ys, xs = p["wi"]; j = r.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
        cy += r.integers(-PATCH // 3, PATCH // 3); cx += r.integers(-PATCH // 3, PATCH // 3)
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, Wd - PATCH)); s = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        x = np.moveaxis(F8[p["cell"]][zidx(p["z"])][:, s[0], s[1]], 0, -1).astype(np.float32) / 255
        x = np.clip(x * r.uniform(0.8, 1.25) + r.uniform(-0.1, 0.1), 0, 1) ** r.uniform(0.7, 1.5)
        if r.random() < 0.2: x = ndi.gaussian_filter(x, (r.uniform(0.3, 1.0),) * 2 + (0,))
        if r.random() < 0.2: x = np.clip(x + r.normal(0, r.uniform(0, 0.03), x.shape), 0, 1)
        a = [x, p["t"][s][..., None].astype(np.float32), p["w"][s][..., None].astype(np.float32)]
        k = r.integers(4); a = [np.rot90(v, k, axes=(0, 1)) for v in a]
        if r.random() < 0.5: a = [v[:, ::-1] for v in a]
        X.append(a[0]); Y.append(a[1]); Wt.append(a[2])
    return [np.ascontiguousarray(np.stack(v), dtype=np.float32) for v in (X, Y, Wt)]


def producer(qq, seed, stop):
    r = np.random.default_rng(seed)
    while not stop.is_set():
        try: b = make_batch(r)
        except Exception as e: log("PRODUCER_ERROR", repr(e)); raise
        while not stop.is_set():
            try: qq.put(b, timeout=0.5); break
            except queue.Full: pass


def pool(R, t): return 2 * sum(r[f"tp_{t}"] for r in R) / max(sum(r[f"pred_{t}"] + r["lab"] for r in R), 1)


def evaluate(p, pr):
    r = dict(cell=p["cell"], z=p["z"]); yv = p["y"] & p["near"]
    for t in THRS:
        b = (pr >= t) & p["near"]; r[f"tp_{t}"] = int((b & yv).sum()); r[f"pred_{t}"] = int(b.sum())
    r["lab"] = int(yv.sum()); return r


folds = [cells[k::3] for k in range(3)]; per_plane, man_res = [], []
for fi, held in enumerate(folds):
    TRAIN = [p for c in cells if c not in held for p in D[c]]; log(f"fold {fi} training planes", len(TRAIN))
    model = build_unet(7); opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(LR, STEPS, alpha=0.05))

    @tf.function
    def step(x, y, w):
        with tf.GradientTape() as t:
            loss = masked_loss(y, w, model(x, training=True))
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables)); return loss
    qq, stop = queue.Queue(maxsize=48), threading.Event()
    ths = [threading.Thread(target=producer, args=(qq, SEED + 1000 * fi + i, stop), daemon=True) for i in range(10)]
    for th in ths: th.start()
    t0 = time.time()
    for s in range(1, STEPS + 1):
        x, y, w = qq.get(timeout=600); loss = float(step(tf.constant(x), tf.constant(y), tf.constant(w)))
        if s % 500 == 0: log(f"fold {fi} step {s} loss {loss:.4f} {time.time()-t0:.0f}s")
    stop.set()
    model.save(f"{OUT}/V9_fold{fi}.keras")
    for c in held:
        probs = {}
        for p in EV[c]:
            pr = predict_tta(model, full_input(c, p["z"])); probs[f"z{p['z']:03d}"] = (pr * 255).astype(np.uint8)
            r = evaluate(p, pr); r["fold"] = fi; per_plane.append(r)
        np.savez_compressed(f"{OUT}/V9_prob_{c}.npz", **probs)
        rr = [r for r in per_plane if r["cell"] == c]
        log("held", c, " ".join("%.1f:%.3f" % (t, pool(rr, t)) for t in THRS))
    for p in MAN:
        if p["cell"] in held:
            pr = predict_tta(model, full_input(p["cell"], p["z"])); r = evaluate(p, pr); r["fold"] = fi; man_res.append(r)
            np.save(f"{OUT}/V9_manual_prob_{p['cell']}_z{p['z']:03d}.npy", (pr * 255).astype(np.uint8))
    log(f"fold {fi} done | held-out Dice near teacher:", " ".join("%.1f:%.3f" % (t, pool([r for r in per_plane if r['fold'] == fi], t)) for t in THRS),
        "| manual:", " ".join("%.1f:%.3f" % (t, pool([r for r in man_res if r['fold'] == fi], t)) for t in THRS))
    json.dump(dict(per_plane=per_plane, manual=man_res), open(f"{OUT}/results_partial.json", "w"), indent=1)
    del model
summ = {f"dice_near_teacher_{t}": pool(per_plane, t) for t in THRS}; summ.update({f"dice_near_manual_{t}": pool(man_res, t) for t in THRS})
log("SUMMARY", json.dumps(summ))
json.dump(dict(settings=dict(steps=STEPS, batch=BATCH, lr=LR, cosine=True, seed=SEED, patch=PATCH, z_offsets=[-3, -2, -1, 0, 1, 2, 3],
                             soft_teacher_targets=True, tta=8, train_planes="odd z 1..149 (75/spheroid)", folds=folds),
               summary=summ, per_plane=per_plane, manual=man_res), open(f"{OUT}/results.json", "w"), indent=1)
log("STUDENT_DONE")
