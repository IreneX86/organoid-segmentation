"""Fluo -> Golgi student U-Net, v5 (GPU, TensorFlow). 3 folds by spheroid (same folds as v1-v4).
Change vs v3 C7 (requested by the user 2026-11-06): PER-CELL labels.
  * input Fluo z-3..z+3 (same as v3 C7), 256 px patches, same U-Net and schedule;
  * labels = teacher pseudo-labels cleaned as in v4 (not manual GT);
  * cells from Cellpose (cellseg.py). In a cell with stained Golgi, Golgi = positive and the REST OF THAT CELL = negative;
    cells without stained Golgi are ignored; outside tissue = negative; removed artefact pieces ignored.
Evaluation on held-out spheroids and the 14 (cleaned) manual planes, INSIDE STAINED CELLS ONLY (stained.py):
pixel Dice/precision/recall, AUC/AP, per-cell Golgi centroid distance (position) and IoU (shape)."""
import glob, json, os, time
import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras import Model, layers
import stained as ST

DATA = "payload/data"; OUT = "output"; os.makedirs(OUT, exist_ok=True)
STEPS, BATCH, LR, PATCH, SEED = int(os.environ.get("STEPS", 5000)), 16, 3e-4, 256, 20261105


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


def dice(a, b): return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


cells = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATA}/cell_*.npz")); assert len(cells) == 18, cells
D, MAN = {}, []
for c in cells:
    d = np.load(f"{DATA}/{c}.npz"); H, W = [int(v) for v in d["shape"]]; CS = np.load(f"cells/{c}.npz")
    for pre, key, dest in (("", "t", None), ("man_", "m", MAN)):
        if pre + "z" not in d.files: continue
        zs = [int(z) for z in d[pre + "z"]]; n = len(zs); FL = d[pre + "fluo8"]; CL = CS[key]
        up = lambda k: np.unpackbits(d[pre + k])[:n * H * W].reshape(n, H, W).astype(bool)
        Y, RM = up("y"), up("rem")
        planes = []
        for k, z in enumerate(zs):
            p = dict(cell=c, z=z, f=np.ascontiguousarray(np.moveaxis(FL[k], 0, -1)), y=Y[k], rem=RM[k], cells=CL[k])
            p["w"], S, ids = ST.weights(p["cells"], p["y"], p["rem"], FL[k][3]); p["nstained"] = len(ids)
            p["yi"] = np.nonzero(p["y"][::2, ::2]); p["wi"] = np.nonzero(p["w"][::4, ::4]); planes.append(p)
        if dest is None: D[c] = planes
        else: MAN.extend(planes)
    log("loaded", c, "stained cells/plane", [p["nstained"] for p in D[c]][:6], "weight frac %.3f" % np.mean([p["w"].mean() for p in D[c]]))
log("manual planes", len(MAN))
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)


def sample_batch(train):
    X, Y, Wt = [], [], []
    for _ in range(BATCH):
        p = train[rng.integers(len(train))]; H, Wd = p["y"].shape
        if rng.random() < 0.6 and p["y"].any():
            ys, xs = p["yi"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 2, xs[j] * 2
        else:
            ys, xs = p["wi"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
        cy += rng.integers(-PATCH // 3, PATCH // 3); cx += rng.integers(-PATCH // 3, PATCH // 3)
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, Wd - PATCH)); s = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        a = [p["f"][s].astype(np.float32) / 255, p["y"][s][..., None].astype(np.float32), p["w"][s][..., None].astype(np.float32)]
        k = rng.integers(4); a = [np.rot90(v, k, axes=(0, 1)) for v in a]
        if rng.random() < 0.5: a = [v[:, ::-1] for v in a]
        X.append(a[0]); Y.append(a[1]); Wt.append(a[2])
    t = lambda v: tf.constant(np.ascontiguousarray(np.stack(v), dtype=np.float32))
    return t(X), t(Y), t(Wt)


def evaluate(p, pr):
    r = ST.evaluate(p["cells"], p["y"], p["rem"], pr); r.update(cell=p["cell"], z=p["z"]); return r


folds = [cells[k::3] for k in range(3)]; per_plane, man_res = [], []
for fi, held in enumerate(folds):
    train = [p for c in cells if c not in held for p in D[c]]
    model = build_unet(7); opt = tf.keras.optimizers.Adam(LR)

    @tf.function
    def step(x, y, w):
        with tf.GradientTape() as t:
            loss = masked_loss(y, w, model(x, training=True))
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables)); return loss
    t0 = time.time()
    for s in range(1, STEPS + 1):
        loss = float(step(*sample_batch(train)))
        if s % 250 == 0: log(f"fold {fi} step {s} loss {loss:.4f} {time.time()-t0:.0f}s")
    model.save(f"{OUT}/V5_fold{fi}.keras")
    for c in held:
        probs = {}
        for p in D[c]:
            pr = predict(model, p["f"].astype(np.float32) / 255); probs[f"z{p['z']:03d}"] = (pr * 255).astype(np.uint8)
            r = evaluate(p, pr); r["fold"] = fi; per_plane.append(r)
        np.savez_compressed(f"{OUT}/V5_prob_{c}.npz", **probs)
        rr = [r for r in per_plane if r["cell"] == c and "auc_S" in r]
        log("held", c, "auc_S %.3f ap_S %.3f dice_S@0.5 %.3f" % tuple(np.mean([r[k] for r in rr]) for k in ("auc_S", "ap_S", "dice_S_0.5")))
    for p in MAN:
        if p["cell"] in held:
            pr = predict(model, p["f"].astype(np.float32) / 255); r = evaluate(p, pr); r["fold"] = fi; man_res.append(r)
            np.save(f"{OUT}/V5_manual_prob_{p['cell']}_z{p['z']:03d}.npy", (pr * 255).astype(np.uint8))
    json.dump(dict(per_plane=per_plane, manual=man_res), open(f"{OUT}/results_partial.json", "w"), indent=1)
    del model
m = lambda R, k: float(np.mean([r[k] for r in R if k in r]))
mm = lambda R, k: float(np.median([v for r in R for v in r.get(k, [])])) if any(r.get(k) for r in R) else None
KEYS = ["auc_S", "ap_S"] + [f"{a}_{t}" for a in ("dice_S", "prec_S", "rec_S") for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
summ = {k: m(per_plane, k) for k in KEYS}; summ.update({"manual_" + k: m(man_res, k) for k in KEYS})
for t in (0.5, 0.8):
    summ[f"median_cell_centroid_px_{t}"] = mm(per_plane, f"cell_centroid_px_{t}"); summ[f"median_cell_iou_{t}"] = mm(per_plane, f"cell_iou_{t}")
log("SUMMARY", json.dumps(summ))
json.dump(dict(settings=dict(steps=STEPS, batch=BATCH, lr=LR, seed=SEED, patch=PATCH, z_offsets=[-3, -2, -1, 0, 1, 2, 3], min_px_stained=ST.MIN_PX, folds=folds),
               summary=summ, per_plane=per_plane, manual=man_res), open(f"{OUT}/results.json", "w"), indent=1)
log("STUDENT_DONE")
