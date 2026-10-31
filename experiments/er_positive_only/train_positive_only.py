"""Fluo-only ER U-Net trained ONLY inside ER-reporter-positive cells (leave-one-specimen-out).

Data: dataset_v2/<specimen>.npz with arrays
  fluo   (P,H,W) uint16  model input (the ONLY input)
  mask   (P,H,W) uint8   ER_mask_v2 pseudo-label (from ER.tif; NOT manual GT)
  region (P,H,W) uint8   ER_positive_region; loss and evaluation use only these pixels.
                         Outside it ER status is unknown -> ignored, never background.
Protocol (fixed before running; no selection on held-out specimen):
  * fold = one held-out specimen; train on the rest; final weights after a fixed number
    of updates (no early stopping, no checkpoint choice, threshold fixed 0.5).
  * base-16 4-level U-Net, BCE + soft Dice, both masked by region; Adam 3e-4; batch 8;
    256-px patches centred in region with p=0.8; flips / rot90 only.
  * per-plane Fluo normalisation p1-p99.8 (same rule at inference).
Held-out metrics (inside region only): AUC, average precision (vs prevalence), Dice@0.5,
and F1 with 2-px tolerance. Compare against h_baseline_v2.json (gradient boosting).
Primary metrics: AUC and average precision (AP) inside region, versus prevalence (~0.26).
Dice and 2-px-tolerant F1 are reported but are NOT decision metrics: on this dense network
an all-region prediction already scores Dice ~0.42 and tolerant F1 ~0.60 (checked 2026-10-31).
CPU reference (h_baseline_v2.json, gradient boosting on Fluo features): mean AUC 0.704, AP 0.415.
Stop rule (fixed before any U-Net run): continue the Fluo-only fine-ER route only if the
U-Net reaches mean held-out AP >= 0.515 (baseline + 0.10) AND AUC >= 0.80 AND beats the
baseline AP in >= 5 of 6 held-out specimens. Otherwise stop this route; no re-tuning.
Usage: python train_positive_only.py --data dataset_v2 --out runs_v2 [--steps 3000]
       [--folds 23298,23300] [--device auto]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score, roc_auc_score

SPECIMENS = ["23298", "23300", "23961", "23963", "23965", "23966"]


def norm(f):
    lo, hi = np.percentile(f, [1, 99.8])
    return np.clip((f.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1.5)


class Block(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, b=16):
        super().__init__()
        c = [b, 2 * b, 4 * b, 8 * b, 16 * b]
        self.enc = nn.ModuleList([Block(1, c[0])] + [Block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([Block(2 * c[i], c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else Fn.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(4)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


def load(data, s):
    """Reads <s>.npz or the split files <s>_part*.npz (same arrays, concatenated by plane)."""
    files = sorted(Path(data).glob(f"{s}_part*.npz"), key=lambda p: int(p.stem.split("part")[1])) or [Path(data) / f"{s}.npz"]
    ds = [np.load(f) for f in files]
    cat = lambda k: np.concatenate([d[k] for d in ds])
    return [norm(f) for f in cat("fluo")], cat("mask") > 0, cat("region") > 0


def sample(planes, rng, n, size=256):
    X, Y, W = [], [], []
    for _ in range(n):
        F, M, R = planes[rng.integers(len(planes))]
        H, Wd = F.shape
        if rng.random() < 0.8 and R.any():
            ys, xs = np.nonzero(R[::4, ::4])
            j = rng.integers(len(ys)); cy, cx = ys[j] * 4, xs[j] * 4
        else:
            cy, cx = rng.integers(H), rng.integers(Wd)
        y0 = int(np.clip(cy - size // 2, 0, H - size)); x0 = int(np.clip(cx - size // 2, 0, Wd - size))
        f, m, r = (a[y0:y0 + size, x0:x0 + size] for a in (F, M, R))
        k = rng.integers(4); fl = rng.random() < 0.5
        f, m, r = (np.rot90(a, k) for a in (f, m, r))
        if fl:
            f, m, r = (a[:, ::-1] for a in (f, m, r))
        X.append(f.copy()); Y.append(m.copy()); W.append(r.copy())
    t = lambda a: torch.from_numpy(np.stack(a)[:, None].astype(np.float32))
    return t(X), t(Y), t(W)


def masked_loss(logit, y, w):
    p = torch.sigmoid(logit)
    bce = (Fn.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    inter = (p * y * w).sum(); den = ((p + y) * w).sum()
    return bce + 1 - (2 * inter + 1) / (den + 1)


@torch.no_grad()
def predict(model, f, dev, tile=512, ov=64):
    H, W = f.shape; out = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    step = tile - ov
    ys = sorted(set(list(range(0, H - tile, step)) + [H - tile])); xs = sorted(set(list(range(0, W - tile, step)) + [W - tile]))
    for y in ys:
        for x in xs:
            t = torch.from_numpy(f[y:y + tile, x:x + tile][None, None]).to(dev)
            out[y:y + tile, x:x + tile] += torch.sigmoid(model(t))[0, 0].float().cpu().numpy(); cnt[y:y + tile, x:x + tile] += 1
    return out / cnt


def evaluate(model, planes, dev):
    model.eval(); P, Y, f1, dice = [], [], [], []
    disk2 = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    for F, M, R in planes:
        p = predict(model, F, dev); P.append(p[R]); Y.append(M[R])
        B = (p >= 0.5) & R; G = M & R
        prec = (B & binary_dilation(G, disk2)).sum() / max(B.sum(), 1); rec = (G & binary_dilation(B, disk2)).sum() / max(G.sum(), 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-9)); dice.append(2 * (B & G).sum() / max(B.sum() + G.sum(), 1))
    P, Y = np.concatenate(P), np.concatenate(Y)
    return dict(auc=float(roc_auc_score(Y, P)), ap=float(average_precision_score(Y, P)), prev=float(Y.mean()),
                ap_norm=float((average_precision_score(Y, P) - Y.mean()) / (1 - Y.mean())),
                dice05=float(np.mean(dice)), f1_tol2=float(np.mean(f1)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="dataset_v2"); a.add_argument("--out", default="runs_v2")
    a.add_argument("--steps", type=int, default=3000); a.add_argument("--batch", type=int, default=8)
    a.add_argument("--folds", default=",".join(SPECIMENS)); a.add_argument("--device", default="auto")
    a.add_argument("--seed", type=int, default=20261031)
    args = a.parse_args()
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                       "mps" if args.device == "auto" and torch.backends.mps.is_available() else
                       ("cpu" if args.device == "auto" else args.device))
    data = {s: list(zip(*load(args.data, s))) for s in SPECIMENS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); results = {}
    for held in args.folds.split(","):
        torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        train = [p for s in SPECIMENS if s != held for p in data[s]]
        model = UNet().to(dev); opt = torch.optim.Adam(model.parameters(), 3e-4); t0 = time.time(); hist = []
        for step in range(1, args.steps + 1):
            model.train(); x, y, w = (t.to(dev) for t in sample(train, rng, args.batch))
            loss = masked_loss(model(x), y, w); opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                hist.append((step, float(loss))); print(f"fold {held} step {step} loss {float(loss):.4f} {time.time()-t0:.0f}s", flush=True)
        torch.save(model.state_dict(), out / f"unet_heldout_{held}.pt")
        results[held] = dict(heldout=evaluate(model, data[held], dev), train_seconds=time.time() - t0, loss_history=hist)
        print(held, {k: round(v, 3) for k, v in results[held]["heldout"].items()}, flush=True)
        json.dump(dict(args=vars(args), device=str(dev), results=results), open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
