#!/usr/bin/env python3
"""Golgi mask from Fluo only (experimental). Ensemble of 6 U-Nets, threshold 0.45.

What it is
  * Students trained to imitate a Golgi-channel U-Net (teacher pseudo-labels), with Fluo as the ONLY input.
  * Ensemble = 3 folds of "v2 2.5D" (Fluo z-1, z, z+1) + 3 folds of "v3 C7" (Fluo z-3 .. z+3); mean probability.
  * Ensemble members and threshold were chosen on held-out spheroids against teacher pseudo-labels only.
  * One-time check on 14 manual Golgi masks (2 spheroids): whole-image Dice 0.218, AUC 0.885 (near-Golgi region).
    The Golgi-channel teacher scores Dice ~0.78 on the same masks: this tool finds roughly WHERE Golgi is,
    not its exact shape. Do not use it to measure Golgi area or morphology.
  * The Golgi channel labelled only some cells; the students also mark Golgi-like regions in other cells,
    and those predictions could not be checked.

Input
  * a folder of single-plane TIFFs of ONE Z-stack (e.g. fluo_001.tif ... fluo_150.tif), sorted by name = Z order, or
  * one multi-page TIFF (pages = Z planes).
  Images should match the training data (1200x1200 MCF-7 spheroid Fluo, same microscope settings).
Output (in --out, never overwritten)
  * golgi_prob_<name>.tif  uint8, probability x 255
  * golgi_mask_<name>.tif  uint8, 0 / 255 at the threshold
  * manifest.json          settings, model and input SHA256

Usage
  python golgi_predict.py INPUT --out OUTDIR [--planes 30,40-45] [--backend auto|tf|numpy] [--threshold 0.45]
  The TensorFlow backend is used when TensorFlow is installed (fast); otherwise a pure NumPy backend runs
  (no extra install, about 1-2 min per model per plane on a laptop CPU).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TILE, OVERLAP = 512, 128


# ----------------------------------------------------------------- utilities
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


_w = np.hanning(TILE).astype(np.float32); HANN = np.outer(_w, _w); HANN /= HANN.max(); HANN = np.maximum(HANN, 1e-3)


def tiled_predict(fn, x):
    """x (H, W, C) float32 in [0,1]; fn maps a batch (N, TILE, TILE, C) -> (N, TILE, TILE)."""
    H, W, _ = x.shape
    ph, pw = max(0, TILE - H), max(0, TILE - W)
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    Hp, Wp, _ = x.shape
    P = np.zeros((Hp, Wp), np.float32); Wt = np.zeros((Hp, Wp), np.float32)
    co = [(y, z) for y in starts(Hp, TILE, TILE - OVERLAP) for z in starts(Wp, TILE, TILE - OVERLAP)]
    for i in range(0, len(co), 4):
        part = co[i:i + 4]
        pr = fn(np.stack([x[y:y + TILE, z:z + TILE] for y, z in part]))
        for (y, z), p in zip(part, pr):
            P[y:y + TILE, z:z + TILE] += p * HANN; Wt[y:y + TILE, z:z + TILE] += HANN
    return np.clip(P / Wt, 0, 1)[:H, :W]


# ----------------------------------------------------------------- weights
def load_weights(path):
    d = np.load(path)
    conv = [(d[f"conv{i}_w"].astype(np.float32), d[f"conv{i}_b"].astype(np.float32)) for i in range(19)]
    bn = [tuple(d[f"bn{i}_{k}"].astype(np.float32) for k in ("gamma", "beta", "mean", "var")) for i in range(9)]
    return dict(conv=conv, bn=bn, offsets=[int(v) for v in d["z_offsets"]])


# ----------------------------------------------------------------- NumPy backend
def _np_conv(x, wb):
    w, b = wb
    if w.shape[0] == 1:
        return x @ w[0, 0] + b
    from numpy.lib.stride_tricks import sliding_window_view
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(xp, (3, 3), axis=(0, 1))
    return np.einsum("hwcij,ijco->hwo", win, w, optimize=True) + b


def _np_block(x, W, i):
    x = np.maximum(_np_conv(x, W["conv"][2 * i]), 0); x = np.maximum(_np_conv(x, W["conv"][2 * i + 1]), 0)
    g, be, m, v = W["bn"][i]
    return ((x - m) / np.sqrt(v + 1e-3) * g + be).astype(np.float32)


def _np_pool(x):
    H, Wd, C = x.shape
    return x.reshape(H // 2, 2, Wd // 2, 2, C).max(axis=(1, 3))


def _np_up(x):
    def idx(n):
        s = np.clip((np.arange(2 * n) + 0.5) / 2 - 0.5, 0, n - 1); i0 = np.floor(s).astype(int)
        return i0, np.minimum(i0 + 1, n - 1), (s - i0).astype(np.float32)
    H, Wd, _ = x.shape
    a, b, f = idx(H); x = x[a] * (1 - f)[:, None, None] + x[b] * f[:, None, None]
    a, b, f = idx(Wd); return (x[:, a] * (1 - f)[None, :, None] + x[:, b] * f[None, :, None]).astype(np.float32)


def np_forward(W, img):
    e1 = _np_block(img, W, 0); e2 = _np_block(_np_pool(e1), W, 1); e3 = _np_block(_np_pool(e2), W, 2)
    e4 = _np_block(_np_pool(e3), W, 3); bt = _np_block(_np_pool(e4), W, 4)
    d4 = _np_block(np.concatenate([_np_up(bt), e4], -1), W, 5); d3 = _np_block(np.concatenate([_np_up(d4), e3], -1), W, 6)
    d2 = _np_block(np.concatenate([_np_up(d3), e2], -1), W, 7); d1 = _np_block(np.concatenate([_np_up(d2), e1], -1), W, 8)
    return 1 / (1 + np.exp(-np.clip(_np_conv(d1, W["conv"][18])[..., 0], -60, 60)))  # clip: no overflow warning


def make_numpy_model(W):
    return lambda batch: np.stack([np_forward(W, b) for b in batch])


# ----------------------------------------------------------------- TensorFlow backend
def make_tf_model(W):
    import tensorflow as tf
    from tensorflow.keras import layers
    cin = W["conv"][0][0].shape[2]; convs, bns = [], []

    def block(x, f, d=0.0):
        c1 = layers.Conv2D(f, 3, padding="same", activation="relu"); c2 = layers.Conv2D(f, 3, padding="same", activation="relu")
        b = layers.BatchNormalization(epsilon=1e-3); convs.extend([c1, c2]); bns.append(b)
        x = b(c2(c1(x)))
        return layers.Dropout(d)(x) if d > 0 else x
    i = tf.keras.Input((None, None, cin)); up = lambda x: layers.UpSampling2D(interpolation="bilinear")(x)
    e1 = block(i, 32); e2 = block(layers.MaxPooling2D()(e1), 64); e3 = block(layers.MaxPooling2D()(e2), 128)
    e4 = block(layers.MaxPooling2D()(e3), 256); bt = block(layers.MaxPooling2D()(e4), 512, 0.4)
    d4 = block(layers.Concatenate()([up(bt), e4]), 256, 0.3); d3 = block(layers.Concatenate()([up(d4), e3]), 128, 0.2)
    d2 = block(layers.Concatenate()([up(d3), e2]), 64, 0.1); d1 = block(layers.Concatenate()([up(d2), e1]), 32)
    head = layers.Conv2D(1, 1, activation="sigmoid"); convs.append(head)
    model = tf.keras.Model(i, head(d1))
    for layer, (w, b) in zip(convs, W["conv"]):
        layer.set_weights([w, b])
    for layer, p in zip(bns, W["bn"]):
        layer.set_weights(list(p))
    return lambda batch: model.predict(batch.astype(np.float32), batch_size=4, verbose=0)[..., 0]


# ----------------------------------------------------------------- main
def read_stack(src):
    if os.path.isdir(src):
        files = sorted(f for f in glob.glob(os.path.join(src, "*.tif*")) if not os.path.basename(f).startswith("._"))
        if not files:
            sys.exit(f"no TIFF files in {src}")
        return [os.path.splitext(os.path.basename(f))[0] for f in files], (lambda k: tifffile.imread(files[k])), files
    with tifffile.TiffFile(src) as t:
        n = len(t.pages)
    base = os.path.splitext(os.path.basename(src))[0]
    return [f"{base}_z{k + 1:03d}" for k in range(n)], (lambda k: tifffile.imread(src, key=k)), [src]


def parse_planes(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a) - 1, int(b or a)))
    return [k for k in out if 0 <= k < n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("--out", required=True)
    ap.add_argument("--planes", default="", help="1-based plane numbers, e.g. 30,40-45 (default: all)")
    ap.add_argument("--backend", default="auto", choices=["auto", "tf", "numpy"])
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"refusing to write into non-empty {a.out}")
    os.makedirs(a.out, exist_ok=True)
    expected = json.load(open(os.path.join(a.models, "MODELS.json")))
    for fn, meta in expected.items():
        if sha256(os.path.join(a.models, fn)) != meta["sha256"]:
            sys.exit(f"model file {fn} does not match MODELS.json (corrupted?)")
    backend = a.backend
    if backend == "auto":
        try:
            import tensorflow  # noqa: F401
            backend = "tf"
        except Exception:
            backend = "numpy"
    Ws = [load_weights(os.path.join(a.models, fn)) for fn in sorted(expected)]
    models = [(W["offsets"], make_tf_model(W) if backend == "tf" else make_numpy_model(W)) for W in Ws]
    names, read, files = read_stack(a.input)
    n = len(names); todo = parse_planes(a.planes, n); cache = {}

    def plane(k):
        k = min(max(k, 0), n - 1)                     # edge planes: repeat the nearest plane
        if k not in cache:
            cache[k] = norm(read(k))
        return cache[k]
    print(f"backend={backend}  models={len(models)}  planes={len(todo)}/{n}", flush=True)
    records = []
    for k in todo:
        t0 = time.time()
        probs = [tiled_predict(fn, np.stack([plane(k + o) for o in offs], -1)) for offs, fn in models]
        P = np.mean(probs, 0); M = P >= a.threshold
        for kk in [c for c in cache if c < k - 3]:
            del cache[kk]
        tifffile.imwrite(os.path.join(a.out, f"golgi_prob_{names[k]}.tif"), np.round(P * 255).astype(np.uint8), compression="zlib")
        tifffile.imwrite(os.path.join(a.out, f"golgi_mask_{names[k]}.tif"), (M * 255).astype(np.uint8), compression="zlib")
        records.append(dict(plane=k + 1, name=names[k], golgi_pixels=int(M.sum()), seconds=round(time.time() - t0, 1)))
        print(f"plane {k + 1}/{n} {names[k]}: {int(M.sum())} px, {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(tool="golgi_predict.py (Fluo-only Golgi, experimental)", backend=backend, threshold=a.threshold,
                   tile=TILE, overlap=OVERLAP, normalisation="per-plane p1-p99.8 -> [0,1]", edge_planes="nearest plane repeated",
                   models=expected, inputs=[dict(file=os.path.basename(f), sha256=sha256(f)) for f in files], planes=records),
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
