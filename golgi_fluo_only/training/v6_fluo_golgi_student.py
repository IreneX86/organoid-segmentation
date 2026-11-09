"""Fluo -> Golgi student U-Net, v6 (GPU, TensorFlow). 3 folds by spheroid (same folds as v1-v5). Requested 2026-11-09.
Same as v3 C7 (Fluo z-3..z+3, 256 px patches, same U-Net/schedule/loss, region = 40 px around Golgi inside tissue +
outside tissue as negatives), except the POSITIVES:
  positive = old 'Golgi label' folder mask AND cleaned teacher pseudo-label (both agree: high precision cores);
  ignored  = teacher-only or old-only Golgi pixels, artefact pieces removed in v4 cleaning;
  planes without an old label (16478 z79-135): all Golgi pixels ignored (negatives still used).
Labels are pseudo-labels (not manual GT). Probability maps of every held-out plane and manual plane are saved; the
full comparison with v3 (plain Dice, tolerance F1, against teacher labels and manual masks) is done locally."""
import glob, json, os, time
import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras import Model, layers
import stained as ST
THRS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

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
    d = np.load(f"{DATA}/{c}.npz"); d6 = np.load(f"{DATA}/v6_{c}.npz"); H, W = [int(v) for v in d["shape"]]
    for pre, dest in (("", None), ("man_", MAN)):
        if pre + "z" not in d.files: continue
        zs = [int(z) for z in d[pre + "z"]]; n = len(zs); FL = d[pre + "fluo8"]
        up = lambda src, k: np.unpackbits(src[pre + k])[:n * H * W].reshape(n, H, W).astype(bool)
        Y, RM, POS, IGN = up(d, "y"), up(d, "rem"), up(d6, "pos"), up(d6, "ign6")
        planes = []
        for k, z in enumerate(zs):
            p = dict(cell=c, z=z, f=np.ascontiguousarray(np.moveaxis(FL[k], 0, -1)), y=Y[k], rem=RM[k], pos=POS[k])
            tis = ST.tissue(FL[k][3]); ext = ndi.binary_erosion(~tis, iterations=10)
            anyg = Y[k] | POS[k] | IGN[k]
            region = (ndi.distance_transform_edt(~anyg) <= 40) & tis if anyg.any() else np.zeros_like(tis)
            p["w"] = ((region | ext) & ~(RM[k] | IGN[k])).astype(np.uint8)
            p["near"] = (ndi.distance_transform_edt(~Y[k]) <= 40) & ~RM[k] if Y[k].any() else np.zeros_like(Y[k])
            p["yi"] = np.nonzero(p["pos"][::2, ::2]); p["wi"] = np.nonzero(p["w"][::4, ::4]); planes.append(p)
        if dest is None: D[c] = planes
        else: MAN.extend(planes)
    log("loaded", c, "pos px", [int(p["pos"].sum()) for p in D[c]][:6], "teacher px", [int(p["y"].sum()) for p in D[c]][:6])
log("manual planes", len(MAN))
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)


def sample_batch(train):
    X, Y, Wt = [], [], []
    for _ in range(BATCH):
        p = train[rng.integers(len(train))]; H, Wd = p["y"].shape
        if rng.random() < 0.6 and p["pos"].any():
            ys, xs = p["yi"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 2, xs[j] * 2
        else:
            ys, xs = p["wi"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
        cy += rng.integers(-PATCH // 3, PATCH // 3); cx += rng.integers(-PATCH // 3, PATCH // 3)
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, Wd - PATCH)); s = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        a = [p["f"][s].astype(np.float32) / 255, p["pos"][s][..., None].astype(np.float32), p["w"][s][..., None].astype(np.float32)]
        k = rng.integers(4); a = [np.rot90(v, k, axes=(0, 1)) for v in a]
        if rng.random() < 0.5: a = [v[:, ::-1] for v in a]
        X.append(a[0]); Y.append(a[1]); Wt.append(a[2])
    t = lambda v: tf.constant(np.ascontiguousarray(np.stack(v), dtype=np.float32))
    return t(X), t(Y), t(Wt)


def evaluate(p, pr):
    r = dict(cell=p["cell"], z=p["z"]); yv = p["y"] & p["near"]
    for t in THRS:
        b = (pr >= t) & p["near"]; r[f"tp_{t}"] = int((b & yv).sum()); r[f"pred_{t}"] = int(b.sum())
    r["lab"] = int(yv.sum()); return r


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
    model.save(f"{OUT}/V6_fold{fi}.keras")
    for c in held:
        probs = {}
        for p in D[c]:
            pr = predict(model, p["f"].astype(np.float32) / 255); probs[f"z{p['z']:03d}"] = (pr * 255).astype(np.uint8)
            r = evaluate(p, pr); r["fold"] = fi; per_plane.append(r)
        np.savez_compressed(f"{OUT}/V6_prob_{c}.npz", **probs)
        rr = [r for r in per_plane if r["cell"] == c]
        log("held", c, " ".join("dice@%.1f %.3f" % (t, 2 * sum(r[f"tp_{t}"] for r in rr) / max(sum(r[f"pred_{t}"] + r["lab"] for r in rr), 1)) for t in THRS))
    for p in MAN:
        if p["cell"] in held:
            pr = predict(model, p["f"].astype(np.float32) / 255); r = evaluate(p, pr); r["fold"] = fi; man_res.append(r)
            np.save(f"{OUT}/V6_manual_prob_{p['cell']}_z{p['z']:03d}.npy", (pr * 255).astype(np.uint8))
    json.dump(dict(per_plane=per_plane, manual=man_res), open(f"{OUT}/results_partial.json", "w"), indent=1)
    del model
pool = lambda R, t: 2 * sum(r[f"tp_{t}"] for r in R) / max(sum(r[f"pred_{t}"] + r["lab"] for r in R), 1)
summ = {f"dice_near_teacher_{t}": pool(per_plane, t) for t in THRS}; summ.update({f"dice_near_manual_{t}": pool(man_res, t) for t in THRS})
log("SUMMARY", json.dumps(summ))
json.dump(dict(settings=dict(steps=STEPS, batch=BATCH, lr=LR, seed=SEED, patch=PATCH, z_offsets=[-3, -2, -1, 0, 1, 2, 3], folds=folds),
               summary=summ, per_plane=per_plane, manual=man_res), open(f"{OUT}/results.json", "w"), indent=1)
log("STUDENT_DONE")
