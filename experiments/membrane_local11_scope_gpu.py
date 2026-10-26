"""Frozen Cellpose v5 50-slice inference for an 11-layer local-Z scope test.

This script reads only the already authorized z280–329 raw extract and a
precomputed global-Z intensity profile; it never reads human labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cellpose
import numpy as np
import torch
from cellpose import models


ROOT = Path(os.environ.get("MEMBRANE_SCOPE_ROOT", "/workspace/membrane_local_scope"))
MODEL_SHA256 = "914f6951034c9297dd9a0a4f687d7256bb1311385a2c11a8c81d884c6e0b91c5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def per_z_bounds(profile: dict, first: int, stop: int) -> np.ndarray:
    if profile["window_radius_z"] != 5 or profile["xy_sample_stride"] != 5:
        raise RuntimeError("Normalization protocol changed")
    values = np.stack([
        np.asarray(profile["channels"][channel]["window11_p1_p99"][first:stop],
                   dtype=np.float32)
        for channel in ("Memb", "fluo")
    ], axis=1)
    if values.shape != (stop - first, 2, 2):
        raise RuntimeError(f"Unexpected bounds shape: {values.shape}")
    if np.any(values[..., 1] <= values[..., 0] + 1e-3):
        raise RuntimeError("Unsafe per-Z intensity range")
    return values


def normalize_at_global_z(raw: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Apply precomputed global-Z bounds, independent of the inference batch."""
    if raw.ndim != 4 or raw.shape[-1] != 2 or bounds.shape != (raw.shape[0], 2, 2):
        raise ValueError((raw.shape, bounds.shape))
    image = raw.astype(np.float32)
    low = bounds[:, None, None, :, 0]
    spread = bounds[:, None, None, :, 1] - low
    np.subtract(image, low, out=image)
    np.divide(image, spread, out=image)
    return image


def main() -> None:
    raw_path = ROOT / "raw_z280_330_yx_full.npy"
    model_path = ROOT / "cpsam_finetuned_v5"
    profile_path = ROOT / "z_intensity_profile.json"
    manifest = json.loads((ROOT / "gpu_input_manifest.json").read_text())
    profile = json.loads(profile_path.read_text())
    if sha256(raw_path) != manifest["npy_sha256"]:
        raise RuntimeError("Raw extract checksum mismatch")
    if sha256(model_path) != MODEL_SHA256:
        raise RuntimeError("v5 checkpoint checksum mismatch")
    for channel in ("Memb", "fluo"):
        if profile["channels"][channel]["source_sha256"] != manifest["source_sha256"][channel]:
            raise RuntimeError(f"Source TIFF provenance mismatch: {channel}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one allocated GPU")
    raw = np.load(raw_path, mmap_mode="r")
    if raw.shape != (50, 1200, 1200, 2) or raw.dtype != np.uint16:
        raise RuntimeError((raw.shape, raw.dtype))
    bounds = per_z_bounds(profile, 280, 330)
    normalized = normalize_at_global_z(raw, bounds)
    output = ROOT / "results"
    output.mkdir(exist_ok=False)
    kwargs = dict(channel_axis=-1, z_axis=0, diameter=None,
                  flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                  stitch_threshold=0.25)
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    start = time.monotonic()
    masks, _, _ = model.eval(normalized, normalize={"normalize": False}, **kwargs)
    masks = np.asarray(masks)
    if masks.shape != (50, 1200, 1200):
        raise RuntimeError(("Wrong output shape", masks.shape))
    path = output / "window11_full50_masks.npz"
    np.savez_compressed(path, masks=masks)
    config = {
        "global_z": [280, 330],
        "comparison_global_z": [300, 310],
        "input_sha256": sha256(raw_path),
        "source_tiff_sha256": manifest["source_sha256"],
        "model_sha256": sha256(model_path),
        "profile_sha256": sha256(profile_path),
        "output_sha256": sha256(path),
        "output_shape": list(masks.shape),
        "output_foreground_pixels": int(np.count_nonzero(masks)),
        "cellpose": cellpose.version,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "normalize": {"normalize": False},
        "eval_kwargs": kwargs,
        "elapsed_seconds": time.monotonic() - start,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("MEMBRANE_LOCAL11_SCOPE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
