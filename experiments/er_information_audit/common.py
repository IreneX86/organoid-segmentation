"""Shared loaders for the read-only ER/Fluo information audit (CPU only).

Ground-truth vocabulary used throughout:
  * ER.tif intensity  -> measured ER-channel signal (not a mask, not manual GT)
  * ER_mask_line.tif  -> algorithmic pseudo-label (NOT ground truth)
No manual ER ground truth exists for this project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

ROOT = Path("/mnt/user-data/uploads/Desktop--Organoid/ER")
PL_ROOT = Path("/mnt/user-data/uploads/ChatGPT--Organoid/outputs/er_coherent_fiber_full_20261028")
SPECIMENS = {
    "23298": ("MCF-7_20260814", "cell_20260814_23298_NZ_50"),
    "23300": ("MCF-7_20260814", "cell_20260814_23300_NZ_100"),
    "23961": ("WT-ER-20260904 ", "cell_20260904_23961_NZ_100"),
    "23963": ("WT-ER-20260904 ", "cell_20260904_23963_NZ_40"),
    "23966": ("WT-ER-20260904 ", "cell_20260904_23966_NZ_100"),
}
SPECIMENS_ALL = dict(SPECIMENS, **{"23965": ("WT-ER-20260904 ", "cell_20260904_23965_NZ_140")})
BATCH = {"23298": "MCF7", "23300": "MCF7", "23961": "WT", "23963": "WT", "23966": "WT", "23965": "WT"}


def paths(spec: str):
    batch, cell = SPECIMENS_ALL[spec]
    d = ROOT / batch / cell
    return d / "ER.tif", d / "fluo.tif", PL_ROOT / batch / cell / "ER_mask_line.tif"


def n_planes(spec: str) -> int:
    with tifffile.TiffFile(paths(spec)[0]) as t:
        return len(t.pages)


def load_plane(spec: str, z: int, which: str) -> np.ndarray:
    er, fl, pl = paths(spec)
    path = {"er": er, "fluo": fl, "pl": pl}[which]
    return tifffile.imread(path, key=z).astype(np.float32)


def sample_planes(spec: str, k: int = 6) -> list[int]:
    """Evenly spaced planes in the central 20-80% of the stack (fixed rule, no selection)."""
    n = n_planes(spec)
    return [int(round(v)) for v in np.linspace(0.2 * (n - 1), 0.8 * (n - 2), k)]


def tissue_mask(fluo: np.ndarray) -> np.ndarray:
    """Broad tissue support from Fluo only (deploy-time available)."""
    s = gaussian_filter(fluo, 8)
    return s > threshold_otsu(s)


def robust_norm(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img[mask], [1, 99.8])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1.5)
