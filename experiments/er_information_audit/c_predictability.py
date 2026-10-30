"""Analysis C: nonlinear, leave-one-specimen-out predictability (CPU only).

Question: given ONLY local Fluo appearance, how much of the ER signal can a
flexible pixel model recover on an unseen specimen, at coarse vs fine scale?

Inputs: multiscale filter-bank features computed from one channel.
Targets (all measured or algorithmic, none are manual ground truth):
  T_fine   ER band-pass detail (Gauss s=1 minus s=4, divided by local std) -> network detail
  T_coarse ER Gaussian s=8 (normalized)                                     -> ER density layout
  T_pl     ER_mask_line pseudo-label (binary)                               -> teacher imitation
Feature sources (arms):
  fluo         Fluo(z) all scales                        <- the deploy-time question
  fluo_coarse  Fluo(z) scales >= 6 only                  <- how much is just cell layout
  er_adj       ER(z+1) all scales                        <- positive reference (adjacent plane)
  fluo_flip    Fluo(z) mirrored left-right               <- null: no pixel correspondence
Model: sklearn HistGradientBoosting, fixed hyperparameters, no tuning.
Evaluation: leave-one-specimen-out (whole specimen held out) and, separately,
within-specimen (train on alternate planes) to quantify cross-specimen shift.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from common import BATCH, SPECIMENS, load_plane, robust_norm, sample_planes, tissue_mask

RNG = np.random.default_rng(20261030)
N_PER_PLANE = 15000
FINE = [0.7, 1.5, 3.0]
COARSE = [6.0, 12.0, 24.0]


def features(img, mask, coarse_only=False):
    x = robust_norm(img, mask)
    feats = []
    sig = COARSE if coarse_only else FINE + COARSE
    g = {s: gaussian_filter(x, s) for s in sig}
    feats += [g[s] for s in sig]
    if not coarse_only:
        feats += [g[0.7] - g[3.0], g[1.5] - g[6.0], g[3.0] - g[12.0]]
        for s in (1.5, 3.0):
            H = hessian_matrix(x, sigma=s, order="rc", use_gaussian_derivatives=False)
            e1, e2 = hessian_matrix_eigvals(H)
            feats += [e1, e2]
        feats.append(gaussian_gradient_magnitude(x, 1.5))
        feats.append(np.sqrt(np.maximum(gaussian_filter(x**2, 4) - gaussian_filter(x, 4) ** 2, 0)))
    else:
        feats += [g[6.0] - g[24.0], gaussian_gradient_magnitude(x, 6.0)]
    return np.stack(feats, -1).astype(np.float32)


def targets(E, PL, mask):
    e = robust_norm(E, mask)
    band = gaussian_filter(e, 1.0) - gaussian_filter(e, 4.0)
    local = np.sqrt(gaussian_filter(band**2, 8)) + 1e-3
    return band / local, gaussian_filter(e, 8.0), (PL > 0).astype(np.uint8)


def build(spec):
    rows = {k: [] for k in ("fluo", "fluo_coarse", "er_adj", "fluo_flip", "t_fine", "t_coarse", "t_pl", "plane")}
    for z in sample_planes(spec):
        E, F, E1, PL = (load_plane(spec, z, "er"), load_plane(spec, z, "fluo"),
                        load_plane(spec, z + 1, "er"), load_plane(spec, z, "pl"))
        m = tissue_mask(F)
        m[:24], m[-24:], m[:, :24], m[:, -24:] = False, False, False, False
        ys, xs = np.nonzero(m)
        pick = RNG.choice(len(ys), size=min(N_PER_PLANE, len(ys)), replace=False)
        ys, xs = ys[pick], xs[pick]
        tf, tc, tp = targets(E, PL, m)
        Ff = np.ascontiguousarray(F[:, ::-1])
        for key, arr in (("fluo", features(F, m)), ("fluo_coarse", features(F, m, True)),
                         ("er_adj", features(E1, m)), ("fluo_flip", features(Ff, m[:, ::-1]))):
            rows[key].append(arr[ys, xs])
        rows["t_fine"].append(tf[ys, xs]); rows["t_coarse"].append(tc[ys, xs]); rows["t_pl"].append(tp[ys, xs])
        rows["plane"].append(np.full(len(ys), z))
    return {k: np.concatenate(v) for k, v in rows.items()}


def fit_eval(Xtr, ytr, Xte, yte, kind):
    if kind == "bin":
        if yte.min() == yte.max():
            return {"auc": None, "ap": None, "prev": float(yte.mean())}
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        return {"auc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p)), "prev": float(yte.mean())}
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"r": float(np.corrcoef(p, yte)[0, 1]), "r2": float(1 - np.mean((p - yte) ** 2) / np.var(yte))}


def main():
    specs = list(SPECIMENS)
    data = {s: build(s) for s in specs}
    print("built", {s: len(d["t_pl"]) for s, d in data.items()}); sys.stdout.flush()
    arms = ("fluo", "fluo_coarse", "er_adj", "fluo_flip")
    tgts = (("t_fine", "reg"), ("t_coarse", "reg"), ("t_pl", "bin"))
    res = {"loso": {}, "within": {}}
    for s in specs:
        res["loso"][s], res["within"][s] = {}, {}
        tr = [t for t in specs if t != s]
        planes = np.unique(data[s]["plane"])
        a_pl, b_pl = planes[0::2], planes[1::2]
        ia, ib = np.isin(data[s]["plane"], a_pl), np.isin(data[s]["plane"], b_pl)
        for arm in arms:
            Xtr = np.concatenate([data[t][arm] for t in tr])
            for tg, kind in tgts:
                ytr = np.concatenate([data[t][tg] for t in tr])
                res["loso"][s][f"{arm}|{tg}"] = fit_eval(Xtr, ytr, data[s][arm], data[s][tg], kind)
                res["within"][s][f"{arm}|{tg}"] = fit_eval(data[s][arm][ia], data[s][tg][ia],
                                                           data[s][arm][ib], data[s][tg][ib], kind)
        print(s, json.dumps({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()}
                             for k, d in res["loso"][s].items()}))
        sys.stdout.flush()
    json.dump(res, open("/home/claude/er_audit/out/c_predictability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
