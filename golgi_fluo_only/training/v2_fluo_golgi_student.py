"""Fluo -> Golgi student U-Net, v2 (GPU, TensorFlow). Two variants on the same data and folds:
  B "2.5D": input = Fluo z-1, z, z+1 (3 channels, each plane normalised p1-p99.8 -> [0,1] on its own)
  A "2D":   input = Fluo z only (the middle channel)            -> isolates the effect of more data
Data: 18 spheroids x 16 centre planes (z = 15, 23, ..., 135); v1 used 8 planes, 2D only.
Labels: teacher golgi_unet.keras (Golgi channel, threshold 0.5) = PSEUDO-LABELS, not manual GT.
Supervision, region R, folds (spheroids[k::3]), 5000 updates, batch 16, 256 patches, Adam 3e-4, threshold 0.5,
metrics: identical to v1. Training uses mixed_bfloat16 for speed (float32 output layer and loss).
Real GT check: 14 manual masks (16471 x11, 16472 x3) scored by the fold model that did not train on them.

Pre-registered comparison (written 2026-11-02 before any v2 result). v1 baseline: region Dice 0.310,
mean AUC 0.827, manual whole-image Dice 0.197.
  * a variant counts as IMPROVED if mean region Dice >= 0.340 AND manual whole-image Dice >= 0.227
    AND mean AUC >= 0.817.
  * 2.5D counts as HELPING over 2D if B's region Dice exceeds A's by >= 0.02 AND B > A in >= 12/18 spheroids.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow.keras import Model, layers

DATA = "payload/data"; OUT = "output"
os.makedirs(OUT, exist_ok=True)
STEPS, BATCH, PATCH, LR = int(os.environ.get("STEPS", 5000)), 16, 256, 3e-4
SEED = 20261102
VARIANTS = os.environ.get("VARIANTS", "B25D,A2D").split(",")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); open(f"{OUT}/progress.txt", "a").write(s + "\n")


def norm(x):
    x = x.astype(np.float32); lo, hi = np.percentile(x, [1.0, 99.8])
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def near(Y, r=40):
    return ndi.distance_transform_edt(~Y) <= r if Y.any() else np.zeros_like(Y)


def starts(n, t, s):
    if n <= t:
        return [0]
    st = list(range(0, n - t + 1, s))
    if st[-1] != n - t:
        st.append(n - t)
    return st


W1 = np.hanning(512).astype(np.float32); WB = np.outer(W1, W1); WB /= WB.max(); WB = np.maximum(WB, 1e-3)


def predict(model, x, tile=512, overlap=128):
    if x.ndim == 2:
        x = x[..., None]
    H, W = x.shape[:2]; P = np.zeros((H, W), np.float32); Wt = np.zeros((H, W), np.float32)
    co = [(y, z) for y in starts(H, tile, tile - overlap) for z in starts(W, tile, tile - overlap)]
    batch = np.stack([x[y:y + tile, z:z + tile] for y, z in co]).astype(np.float32)
    pr = np.asarray(model.predict(batch, batch_size=4, verbose=0), np.float32)[..., 0]
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


def build_unet(cin, b=32):  # teacher architecture, cin input channels, float32 sigmoid output
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


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


# ---------------- 1. teacher pseudo-labels (float32) ----------------
teacher = tf.keras.models.load_model(f"{DATA}/golgi_unet.keras", compile=False)
cells = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATA}/cell_*.npz"))
D = {}
for c in cells:
    d = np.load(f"{DATA}/{c}.npz"); planes = []
    for k in range(len(d["z"])):
        f3 = np.stack([norm(d["fluo"][k, j]) for j in range(3)], -1)
        T = tissue(f3[..., 1]); Y = predict(teacher, norm(d["golgi"][k])) >= 0.5
        R = near(Y) & (T | Y); ext = ndi.binary_erosion(~T, iterations=10)
        planes.append(dict(z=int(d["z"][k]), f=f3.astype(np.float16), y=Y, r=R, w=(R | ext).astype(np.uint8)))  # fp16/uint8 to halve RAM
    for p in planes:  # precomputed sampling indices (faster batches)
        p["yi"] = np.nonzero(p["y"][::2, ::2]); p["ri"] = np.nonzero(p["r"][::4, ::4])
    D[c] = planes
    log("teacher", c, [int(p["y"].sum()) for p in planes])
np.savez_compressed(f"{OUT}/teacher_masks.npz", **{f"{c}_z{p['z']:03d}": np.packbits(p["y"]) for c in cells for p in D[c]})
M = np.load(f"{DATA}/manual14.npz"); man = []
for k in range(len(M["z"])):
    f3 = np.stack([norm(M["fluo"][k, j]) for j in range(3)], -1); T = tissue(f3[..., 1]); G = M["manual"][k] > 0
    man.append(dict(cell=str(M["cell"][k]), z=int(M["z"][k]), f=f3, y=G, r=near(G) & (T | G),
                    teacher=predict(teacher, norm(M["golgi"][k])) >= 0.5))
log("teacher dice vs manual", [round(dice(m["y"], m["teacher"]), 3) for m in man])
del teacher
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

# ---------------- 2. students ----------------
folds = [cells[k::3] for k in range(3)]
all_results = {}
for var in VARIANTS:
    ch = slice(0, 3) if var == "B25D" else slice(1, 2); cin = 3 if var == "B25D" else 1
    rng = np.random.default_rng(SEED); tf.random.set_seed(SEED)
    results, man_res = {}, []

    def sample_batch(train):
        X, Y, Wt = [], [], []
        for _ in range(BATCH):
            p = train[rng.integers(len(train))]; H, Wd = p["y"].shape
            if rng.random() < 0.7 and p["y"].any():
                ys, xs = p["yi"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 2, xs[j] * 2
            elif rng.random() < 0.6 and p["r"].any():
                ys, xs = p["ri"]; j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
            else:
                cy, cx = rng.integers(H), rng.integers(Wd)
            cy += rng.integers(-PATCH // 3, PATCH // 3); cx += rng.integers(-PATCH // 3, PATCH // 3)
            y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH)); x0 = int(np.clip(cx - PATCH // 2, 0, Wd - PATCH))
            a = [p["f"][y0:y0 + PATCH, x0:x0 + PATCH, ch], p["y"][y0:y0 + PATCH, x0:x0 + PATCH, None].astype(np.float32),
                 p["w"][y0:y0 + PATCH, x0:x0 + PATCH, None]]
            k = rng.integers(4); a = [np.rot90(v, k, axes=(0, 1)) for v in a]
            if rng.random() < 0.5:
                a = [v[:, ::-1] for v in a]
            X.append(a[0]); Y.append(a[1]); Wt.append(a[2])
        s = lambda v: tf.constant(np.ascontiguousarray(np.stack(v), dtype=np.float32))
        return s(X), s(Y), s(Wt)

    for fi, held in enumerate(folds):
        train = [p for c in cells if c not in held for p in D[c]]
        model = build_unet(cin); opt = tf.keras.optimizers.Adam(LR)

        @tf.function
        def step(x, y, w):
            with tf.GradientTape() as t:
                loss = masked_loss(y, w, model(x, training=True))
            opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        t0 = time.time()
        for s in range(1, STEPS + 1):
            loss = float(step(*sample_batch(train)))
            if s % 250 == 0:
                log(f"{var} fold {fi} step {s} loss {loss:.4f} {time.time()-t0:.0f}s")
        model.save(f"{OUT}/{var}_fold{fi}.keras")
        for c in held:
            P, Yv, Bv, dl, ds, figs = [], [], [], [], [], {}
            for p in D[c]:
                pr = predict(model, p["f"][..., ch]); R = p["r"]
                if p["y"].any():
                    P.append(pr[R]); Yv.append(p["y"][R]); Bv.append(ndi.gaussian_filter(p["f"][..., 1].astype(np.float32), 1)[R])
                    b = (pr >= 0.5) & R; dl.append(dice(b, p["y"] & R)); ds.append(dice(np.roll(b, 4, 1), p["y"] & R))
                if p["z"] in (71, 103):
                    figs[f"z{p['z']:03d}"] = (pr * 255).astype(np.uint8)
            np.savez_compressed(f"{OUT}/{var}_prob_{c}.npz", **figs)
            P, Yv, Bv = map(np.concatenate, (P, Yv, Bv))
            results[c] = dict(fold=fi, auc=float(roc_auc_score(Yv, P)), ap=float(average_precision_score(Yv, P)), prev=float(Yv.mean()),
                              auc_intensity=float(roc_auc_score(Yv, Bv)), dice05=float(np.mean(dl)), dice05_shift4=float(np.mean(ds)),
                              pred_frac_in_region=float((P >= 0.5).mean()))
            log(var, c, {k: round(v, 3) for k, v in results[c].items()})
        for m in man:
            if m["cell"] in held:
                pr = predict(model, m["f"][..., ch]); R = m["r"]; b = pr >= 0.5
                man_res.append(dict(cell=m["cell"], z=m["z"], fold=fi, auc=float(roc_auc_score(m["y"][R], pr[R])),
                                    ap=float(average_precision_score(m["y"][R], pr[R])), prev=float(m["y"][R].mean()),
                                    dice_in_region=dice(b & R, m["y"]), dice_whole=dice(b, m["y"]), teacher_dice_whole=dice(m["teacher"], m["y"])))
                np.save(f"{OUT}/{var}_manual_prob_{m['cell']}_z{m['z']:03d}.npy", (pr * 255).astype(np.uint8))
        json.dump(dict(per_spheroid=results, manual=man_res), open(f"{OUT}/{var}_results_partial.json", "w"), indent=1)
    a = lambda k: float(np.mean([results[c][k] for c in cells]))
    summ = dict(mean_auc=a("auc"), mean_ap=a("ap"), mean_prev=a("prev"), mean_auc_intensity=a("auc_intensity"),
                mean_dice05=a("dice05"), mean_dice05_shift4=a("dice05_shift4"), n_auc_ge_075=int(sum(results[c]["auc"] >= 0.75 for c in cells)),
                manual_mean_auc=float(np.mean([m["auc"] for m in man_res])), manual_mean_ap=float(np.mean([m["ap"] for m in man_res])),
                manual_mean_dice_whole=float(np.mean([m["dice_whole"] for m in man_res])),
                manual_teacher_dice=float(np.mean([m["teacher_dice_whole"] for m in man_res])))
    summ["IMPROVED_vs_v1"] = bool(summ["mean_dice05"] >= 0.340 and summ["manual_mean_dice_whole"] >= 0.227 and summ["mean_auc"] >= 0.817)
    all_results[var] = dict(summary=summ, per_spheroid=results, manual=man_res)
    log("SUMMARY", var, json.dumps(summ))
    json.dump(dict(settings=dict(steps=STEPS, batch=BATCH, patch=PATCH, lr=LR, seed=SEED, folds=folds, precision="mixed_bfloat16"),
                   variants=all_results), open(f"{OUT}/results.json", "w"), indent=1)
if set(VARIANTS) >= {"A2D", "B25D"}:
    rb, ra = all_results["B25D"]["per_spheroid"], all_results["A2D"]["per_spheroid"]
    diff = np.mean([rb[c]["dice05"] - ra[c]["dice05"] for c in cells]); nwin = int(sum(rb[c]["dice05"] > ra[c]["dice05"] for c in cells))
    cmp = dict(b_minus_a_region_dice=float(diff), b_wins=nwin, HELPS_25D=bool(diff >= 0.02 and nwin >= 12))
    log("COMPARE", json.dumps(cmp))
    json.dump(dict(settings=dict(steps=STEPS, batch=BATCH, patch=PATCH, lr=LR, seed=SEED, folds=folds, precision="mixed_bfloat16"),
                   variants=all_results, compare=cmp), open(f"{OUT}/results.json", "w"), indent=1)
log("STUDENT_DONE")
