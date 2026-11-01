"""NumPy re-implementation of the PyTorch UNet in train_positive_only.py (inference only).

Loads the saved state_dict (.pt zip) without torch, runs eval-mode forward
(BatchNorm with running statistics). Verified by reproducing held-out metrics.
"""
from __future__ import annotations

import pickle
import zipfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DT = {"FloatStorage": np.float32, "LongStorage": np.int64, "DoubleStorage": np.float64}


def load_state_dict(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class _S:  # storage marker
        def __init__(self, name):
            self.name = name

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *a):
                    arr = storage
                    item = arr.itemsize
                    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                                           strides=tuple(s * item for s in stride)).copy()
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return _S(name)
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            raw = z.read(f"{root}/data/{key}")
            return np.frombuffer(raw, dtype=DT[stype.name])[:numel]

    return U(z.open(f"{root}/data.pkl")).load()


def conv3(x, w, b):  # x (C,H,W), w (O,C,3,3), padding 1
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))  # C,H,W,3,3
    out = np.einsum("chwij,ocij->ohw", win, w, optimize=True)
    return out + b[:, None, None]


def bn(x, sd, p, eps=1e-5):
    g, b, m, v = sd[p + ".weight"], sd[p + ".bias"], sd[p + ".running_mean"], sd[p + ".running_var"]
    return (x - m[:, None, None]) / np.sqrt(v[:, None, None] + eps) * g[:, None, None] + b[:, None, None]


def block(x, sd, p):
    x = np.maximum(bn(conv3(x, sd[p + ".0.weight"], sd[p + ".0.bias"]), sd, p + ".1"), 0)
    return np.maximum(bn(conv3(x, sd[p + ".3.weight"], sd[p + ".3.bias"]), sd, p + ".4"), 0)


def pool(x):
    C, H, W = x.shape
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def upconv(x, w, b):  # ConvTranspose2d k2 s2: w (Cin,Cout,2,2)
    C, H, W = x.shape
    out = np.einsum("chw,coij->ohiwj", x, w, optimize=True).reshape(w.shape[1], H * 2, W * 2)
    return out + b[:, None, None]


def forward(sd, img):
    x = img[None].astype(np.float32)
    skips = []
    for i in range(5):
        x = block(x if i == 0 else pool(x), sd, f"enc.{i}")
        skips.append(x)
    x = skips.pop()
    for i in reversed(range(4)):
        x = block(np.concatenate([upconv(x, sd[f"up.{i}.weight"], sd[f"up.{i}.bias"]), skips[i]], 0), sd, f"dec.{i}")
    logit = np.einsum("chw,oc->ohw", x, sd["head.weight"][:, :, 0, 0]) + sd["head.bias"][:, None, None]
    return 1 / (1 + np.exp(-logit[0]))


def predict(sd, f, tile=512, ov=64):
    """Same tiling as train_positive_only.predict (simple average of overlapping tiles)."""
    H, W = f.shape
    out = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    step = tile - ov
    ys = sorted(set(list(range(0, H - tile, step)) + [H - tile]))
    xs = sorted(set(list(range(0, W - tile, step)) + [W - tile]))
    for y in ys:
        for x in xs:
            out[y:y + tile, x:x + tile] += forward(sd, f[y:y + tile, x:x + tile])
            cnt[y:y + tile, x:x + tile] += 1
    return out / cnt
