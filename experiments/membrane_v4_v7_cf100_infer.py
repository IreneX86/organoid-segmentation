"""Frozen two-checkpoint, 2D-chunk/global-3D-stitch inference for CF 100-Z.

Same local11 normalization and Cellpose eval kwargs as the completed MCF-7
v4/v7 paired runs. This runs only after one-GPU preflight; it never trains.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import cellpose
import numpy as np
import tifffile
import torch
from cellpose import models, utils
from membrane_local11_scope_gpu import normalize_at_global_z, per_z_bounds, sha256


ROOT = Path(os.environ.get("MEMBRANE_CF_ROOT", "/workspace/membrane_v4_v7_cf100_20261023"))
Z_COUNT, YX = 100, (1200, 1200)
KWARGS = dict(channel_axis=-1, z_axis=0, diameter=None, flow_threshold=.4,
              cellprob_threshold=0., do_3D=False, stitch_threshold=.25)


def dense_plane(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    if ids[0] != 0:
        ids = np.r_[0, ids]
    return np.searchsorted(ids, mask).astype(np.uint32)


def main() -> None:
    manifest = json.loads((ROOT / "input_manifest.json").read_text())
    if manifest["shape_zyx"] != [Z_COUNT, *YX]:
        raise ValueError("Source manifest shape changed")
    for name, info in manifest["files"].items():
        if sha256(ROOT / name) != info["sha256"]:
            raise RuntimeError(f"Input SHA256 mismatch: {name}")
    if cellpose.version != "4.2.1.1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require Cellpose 4.2.1.1 and exactly one CUDA GPU")
    matmul = torch.ones((16, 16), device="cuda:0") @ torch.ones((16, 16), device="cuda:0")
    if matmul.device.type != "cuda" or float(matmul[0, 0]) != 16:
        raise RuntimeError("GPU matrix preflight failed")
    print("GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0), flush=True)
    profile = json.loads((ROOT / "z_intensity_profile.json").read_text())
    for channel in ("Memb", "fluo"):
        if profile["channels"][channel]["source_sha256"] != manifest["files"][f"{channel}.tif"]["sha256"]:
            raise RuntimeError("Normalization profile/source mismatch")
    bounds = per_z_bounds(profile, 0, Z_COUNT)
    out = ROOT / "results"
    out.mkdir(exist_ok=False)
    config = {"cell": manifest["cell"], "source_sha256": {k: manifest["files"][k]["sha256"]
              for k in ("Memb.tif", "fluo.tif")},
              "model_sha256": {}, "cellpose": cellpose.version,
              "python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "visible_gpu_count": torch.cuda.device_count(),
              "profile_sha256": sha256(ROOT / "z_intensity_profile.json"),
              "normalization": manifest["normalization"], "eval_kwargs": KWARGS,
              "runs": {}}
    for version in ("v4", "v7"):
        model_key = f"cpsam_finetuned_{version}"
        model = models.CellposeModel(gpu=True, pretrained_model=str(ROOT / model_key))
        planes = np.zeros((Z_COUNT, *YX), dtype=np.uint32)
        chunks = []
        start = time.monotonic()
        with tifffile.TiffFile(ROOT / "Memb.tif") as memb, \
                tifffile.TiffFile(ROOT / "fluo.tif") as fluo:
            for z0 in range(0, Z_COUNT, 10):
                z1 = min(Z_COUNT, z0 + 10)
                raw = np.stack([np.stack((memb.pages[z].asarray(), fluo.pages[z].asarray()), axis=-1)
                                for z in range(z0, z1)])
                normalized = normalize_at_global_z(raw, bounds[z0:z1])
                t0 = time.monotonic()
                masks, _, _ = model.eval(normalized, normalize={"normalize": False}, **KWARGS)
                masks = np.asarray(masks)
                if masks.shape != (z1-z0, *YX):
                    raise RuntimeError(f"Wrong {version} output shape: {masks.shape}")
                for j, z in enumerate(range(z0, z1)):
                    planes[z] = dense_plane(masks[j])
                chunks.append({"z0": z0, "z1": z1, "seconds": time.monotonic()-t0})
                print(f"{version.upper()}_CHUNK_COMPLETE {z0}:{z1}", flush=True)
        stitched = np.asarray(utils.stitch3D(planes, stitch_threshold=.25))
        if stitched.max() > 65535:
            raise RuntimeError("uint16 cannot represent the stitched instance IDs")
        output = out / f"memb_instances_{version}_local11.tif"
        tifffile.imwrite(output, stitched.astype(np.uint16), photometric="minisblack",
                         imagej=True, compression="zlib", compressionargs={"level": 1},
                         metadata={"axes": "ZYX", "Info": f"{version}; CF; local11; global stitch .25"})
        config["model_sha256"][version] = manifest["files"][model_key]["sha256"]
        config["runs"][version] = {"chunks": chunks, "elapsed_seconds": time.monotonic()-start,
                                   "3d_ids": int(len(np.unique(stitched))-1),
                                   "output": output.name, "output_sha256": sha256(output)}
        del model, planes, stitched
        torch.cuda.empty_cache()
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("MEMBRANE_CF100_V4_V7_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
