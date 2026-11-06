"""Fluo -> Golgi student U-Net, trained on teacher pseudo-labels (GPU, TensorFlow).

Teacher: golgi_unet.keras (2026-10-26, Golgi-channel input, SHA256 6f248ca4...), its own inference
  settings: per-image p1-p99.8 -> [0,1], 512 tiles, 128 overlap, Hann blending, threshold 0.5.
  Teacher masks are PSEUDO-LABELS, not manual ground truth.
Student: same 2D U-Net architecture (base 32), input = Fluo only (per-image p1-p99.8 -> [0,1]).
Mosaic handling: the Golgi channel is bright only in some cells. Supervised pixels:
  * R = teacher mask dilated by 40 px, inside tissue (near cells that do show Golgi): 1 / 0 as teacher says;
  * exterior background (outside tissue, eroded 10 px): 0;
  * everything else (tissue far from any visible Golgi) is IGNORED: status unknown.
Split: 3 folds by whole spheroid (sorted list, fold k = spheroids[k::3]); fixed 5000 updates, final
  weights, threshold 0.5 -> no checkpoint selection on held-out data.
Held-out evaluation inside R: AUC, AP vs prevalence, Dice@0.5, 4-px shift Dice, Fluo-intensity AUC baseline.
Real GT check: 14 manual masks (16471 x11, 16472 x3), scored by the fold model that did not train on them.
Pre-registered decision (2026-11-02, before any result):
  GO if mean AUC >= 0.80 AND mean AP >= 2 x mean prevalence AND AUC >= 0.75 in >= 14/18 spheroids.
"""
import glob
import json
import os
import time

import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras import Model, layers

DATA = "payload/data"; OUT = "output"
os.makedirs(OUT, exist_ok=True)
STEPS, BATCH, PATCH, LR = 5000, 16, 256, 3e-4
SEED = 20261102
rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)
DISK40 = (np.hypot(*np.mgrid[-40:41, -40:41]) <= 40)


def near(Y, r=40):  # == binary_dilation(Y, disk(r)), via Euclidean distance transform (much faster)
    return ndi.distance_transform_edt(~Y) <= r if Y.any() else np.zeros_like(Y)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(f"{OUT}/progress.txt", "a").write(s + "\n")


def norm(x):
    x = x.astype(np.float32); lo, hi = np.percentile(x, [1.0, 99.8])
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def starts(n, t, s):
    if n <= t:
        return [0]
    st = list(range(0, n - t + 1, s))
    if st[-1] != n - t:
        st.append(n - t)
    return st


W1 = np.hanning(512).astype(np.float32); WB = np.outer(W1, W1); WB /= WB.max(); WB = np.maximum(WB, 1e-3)


def predict(model, x, tile=512, overlap=128):
    P = np.zeros(x.shape, np.float32); Wt = np.zeros(x.shape, np.float32)
    co = [(y, z) for y in starts(x.shape[0], tile, tile - overlap) for z in starts(x.shape[1], tile, tile - overlap)]
    batch = np.stack([x[y:y + tile, z:z + tile] for y, z in co])[..., None]
    pr = model.predict(batch, batch_size=4, verbose=0)[..., 0]
    for (y, z), p in zip(co, pr):
        P[y:y + tile, z:z + tile] += p * WB; Wt[y:y + tile, z:z + tile] += WB
    return np.clip(P / Wt, 0, 1)


def tissue(fn):
    s = ndi.gaussian_filter(fn, 8); return s > threshold_otsu(s)


def block(x, f, d=0.0):
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return layers.Dropout(d)(x) if d > 0 else x


def build_unet(b=32):  # identical to the teacher's build_unet_2d
    i = tf.keras.Input((None, None, 1))
    e1 = block(i, b); e2 = block(layers.MaxPooling2D()(e1), 2 * b); e3 = block(layers.MaxPooling2D()(e2), 4 * b)
    e4 = block(layers.MaxPooling2D()(e3), 8 * b); bt = block(layers.MaxPooling2D()(e4), 16 * b, 0.4)
    up = lambda x: layers.UpSampling2D(interpolation="bilinear")(x)
    d4 = block(layers.Concatenate()([up(bt), e4]), 8 * b, 0.3); d3 = block(layers.Concatenate()([up(d4), e3]), 4 * b, 0.2)
    d2 = block(layers.Concatenate()([up(d3), e2]), 2 * b, 0.1); d1 = block(layers.Concatenate()([up(d2), e1]), b)
    return Model(i, layers.Conv2D(1, 1, activation="sigmoid")(d1))


def masked_loss(y, w, p):
    p = tf.clip_by_value(p, 1e-6, 1 - 1e-6); n = tf.reduce_sum(w) + 1e-6
    bce = tf.reduce_sum(w * -(y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))) / n
    dice = (2 * tf.reduce_sum(w * y * p) + 1) / (tf.reduce_sum(w * y) + tf.reduce_sum(w * p) + 1)
    return bce + 1 - dice


# ---------------- 1. teacher pseudo-labels ----------------
teacher = tf.keras.models.load_model(f"{DATA}/golgi_unet.keras", compile=False)
cells = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATA}/cell_*.npz"))
D = {}
for c in cells:
    d = np.load(f"{DATA}/{c}.npz"); planes = []
    for k in range(len(d["z"])):
        fn = norm(d["fluo"][k]); T = tissue(fn)
        Y = predict(teacher, norm(d["golgi"][k])) >= 0.5
        R = near(Y) & (T | Y)
        ext = ndi.binary_erosion(~T, iterations=10)
        W = (R | ext).astype(np.float32)
        planes.append(dict(z=int(d["z"][k]), f=fn, y=Y, r=R, w=W))
    D[c] = planes
    log("teacher", c, [int(p["y"].sum()) for p in planes])
np.savez_compressed(f"{OUT}/teacher_masks.npz", **{f"{c}_z{p['z']:03d}": np.packbits(p["y"]) for c in cells for p in D[c]})

# manual GT planes
M = np.load(f"{DATA}/manual14.npz")
man = []
for k in range(len(M["z"])):
    fn = norm(M["fluo"][k]); T = tissue(fn); G = M["manual"][k] > 0
    tY = predict(teacher, norm(M["golgi"][k])) >= 0.5
    man.append(dict(cell=str(M["cell"][k]), z=int(M["z"][k]), f=fn, y=G, r=near(G) & (T | G), teacher=tY))
log("teacher dice vs manual", [round(2 * (m["y"] & m["teacher"]).sum() / (m["y"].sum() + m["teacher"].sum()), 3) for m in man])


# ---------------- 2. student training, 3 folds by spheroid ----------------
def sample_batch(train):
    X, Y, Wt = [], [], []
    for _ in range(BATCH):
        p = train[rng.integers(len(train))]; H, Wd = p["f"].shape
        if rng.random() < 0.7 and p["y"].any():
            ys, xs = np.nonzero(p["y"][::2, ::2]); j = rng.integers(len(ys)); cy, cx = ys[j] * 2, xs[j] * 2
        elif rng.random() < 0.6 and p["r"].any():
            ys, xs = np.nonzero(p["r"][::4, ::4]); j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
        else:
            cy, cx = rng.integers(H), rng.integers(Wd)
        cy += rng.integers(-PATCH // 3, PATCH // 3); cx += rng.integers(-PATCH // 3, PATCH // 3)
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, Wd - PATCH))
        a = [p["f"][y0:y0 + PATCH, x0:x0 + PATCH], p["y"][y0:y0 + PATCH, x0:x0 + PATCH].astype(np.float32), p["w"][y0:y0 + PATCH, x0:x0 + PATCH]]
        k = rng.integers(4); a = [np.rot90(v, k) for v in a]
        if rng.random() < 0.5:
            a = [v[:, ::-1] for v in a]
        X.append(a[0]); Y.append(a[1]); Wt.append(a[2])
    s = lambda v: np.ascontiguousarray(np.stack(v)[..., None], dtype=np.float32)
    return s(X), s(Y), s(Wt)


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


folds = [cells[k::3] for k in range(3)]
results, man_res, curves = {}, [], {}
for fi, held in enumerate(folds):
    train = [p for c in cells if c not in held for p in D[c]]
    model = build_unet(); opt = tf.keras.optimizers.Adam(LR)

    @tf.function
    def step(x, y, w):
        with tf.GradientTape() as t:
            loss = masked_loss(y, w, model(x, training=True))
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss
    t0 = time.time(); hist = []
    for s in range(1, STEPS + 1):
        loss = float(step(*[tf.constant(a) for a in sample_batch(train)]))
        if s % 250 == 0:
            hist.append((s, loss)); log(f"fold {fi} step {s} loss {loss:.4f} {time.time()-t0:.0f}s")
    curves[fi] = hist
    model.save(f"{OUT}/student_fold{fi}.keras")
    for c in held:
        P, Yv, Bv = [], [], []; dl, ds = [], []
        figs = {}
        for p in D[c]:
            pr = predict(model, p["f"]); R = p["r"]
            if p["y"].any():
                P.append(pr[R]); Yv.append(p["y"][R]); Bv.append(ndi.gaussian_filter(p["f"], 1)[R])
                b = (pr >= 0.5) & R; dl.append(dice(b, p["y"] & R)); ds.append(dice(np.roll(b, 4, 1), p["y"] & R))
            if p["z"] in (69, 107):
                figs[f"z{p['z']:03d}"] = (pr * 255).astype(np.uint8)
        np.savez_compressed(f"{OUT}/student_prob_{c}.npz", **figs)
        P, Yv, Bv = map(np.concatenate, (P, Yv, Bv))
        results[c] = dict(fold=fi, auc=float(roc_auc_score(Yv, P)), ap=float(average_precision_score(Yv, P)), prev=float(Yv.mean()),
                          auc_intensity=float(roc_auc_score(Yv, Bv)), dice05=float(np.mean(dl)), dice05_shift4=float(np.mean(ds)),
                          pred_frac_in_region=float((P >= 0.5).mean()))
        log(c, {k: round(v, 3) for k, v in results[c].items()})
    for m in man:
        if m["cell"] in held:
            pr = predict(model, m["f"]); R = m["r"]; b = pr >= 0.5
            man_res.append(dict(cell=m["cell"], z=m["z"], fold=fi, auc=float(roc_auc_score(m["y"][R], pr[R])),
                                ap=float(average_precision_score(m["y"][R], pr[R])), prev=float(m["y"][R].mean()),
                                dice_in_region=dice(b & R, m["y"]), dice_whole=dice(b, m["y"]),
                                teacher_dice_whole=dice(m["teacher"], m["y"])))
            np.save(f"{OUT}/manual_prob_{m['cell']}_z{m['z']:03d}.npy", (pr * 255).astype(np.uint8))
            log("manual", man_res[-1])
    json.dump(dict(per_spheroid=results, manual=man_res, curves=curves), open(f"{OUT}/results_partial.json", "w"), indent=1)

aucs = np.array([results[c]["auc"] for c in cells]); aps = np.array([results[c]["ap"] for c in cells]); prev = np.array([results[c]["prev"] for c in cells])
summ = dict(mean_auc=float(aucs.mean()), mean_ap=float(aps.mean()), mean_prev=float(prev.mean()), n_auc_ge_075=int((aucs >= 0.75).sum()),
            mean_auc_intensity=float(np.mean([results[c]["auc_intensity"] for c in cells])),
            mean_dice05=float(np.mean([results[c]["dice05"] for c in cells])), mean_dice05_shift4=float(np.mean([results[c]["dice05_shift4"] for c in cells])),
            manual_mean_auc=float(np.mean([m["auc"] for m in man_res])), manual_mean_ap=float(np.mean([m["ap"] for m in man_res])),
            manual_mean_dice_whole=float(np.mean([m["dice_whole"] for m in man_res])), manual_teacher_dice=float(np.mean([m["teacher_dice_whole"] for m in man_res])))
summ["GO"] = bool(summ["mean_auc"] >= 0.80 and summ["mean_ap"] >= 2 * summ["mean_prev"] and summ["n_auc_ge_075"] >= 14)
json.dump(dict(summary=summ, settings=dict(steps=STEPS, batch=BATCH, patch=PATCH, lr=LR, seed=SEED, folds=folds),
               per_spheroid=results, manual=man_res, curves=curves), open(f"{OUT}/results.json", "w"), indent=1)
log("SUMMARY", json.dumps(summ))
log("STUDENT_DONE")
