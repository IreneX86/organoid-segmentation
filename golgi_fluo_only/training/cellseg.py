"""Cell instance segmentation of the Fluo centre plane of every labelled plane with Cellpose (CPSAM, generalist model,
no fine-tuning). Images are downscaled 2x (cells ~150 px -> ~75 px) and masks upscaled back (nearest).
Output: cells/<cell>.npz  t=(16,H,W) uint16 for teacher planes, m=(n,H,W) uint16 for manual planes (0 = no cell)."""
import glob, os, time
import numpy as np
import torch
from importlib.metadata import version
from cellpose import models
version_str = version("cellpose")
import cv2

os.makedirs("cells", exist_ok=True)
print("cellpose", version_str, "cuda", torch.cuda.is_available(), flush=True)
model = models.CellposeModel(gpu=True)


def run(planes):
    small = [cv2.resize(p.astype(np.float32) / 255, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA) for p in planes]
    masks = model.eval(small, batch_size=16)[0]
    H, W = planes[0].shape
    return np.stack([cv2.resize(m.astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST) for m in masks])


t0 = time.time()
for p in sorted(glob.glob("payload/data/cell_*.npz")):
    c = os.path.basename(p)[:-4]; d = np.load(p); out = dict(t=run(list(d["fluo8"][:, 3])))
    if "man_z" in d.files:
        out["m"] = run(list(d["man_fluo8"][:, 3]))
    np.savez_compressed(f"cells/{c}.npz", **out)
    print(c, "cells per plane", [int(m.max()) for m in out["t"]][:6], f"{time.time()-t0:.0f}s", flush=True)
print("CELLSEG_DONE", flush=True)
