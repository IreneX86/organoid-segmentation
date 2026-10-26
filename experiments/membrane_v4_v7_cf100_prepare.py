"""Prepare a label-free local11 profile/manifest for the CF 100-Z demo.

Only reads the original TIFFs and frozen checkpoints. Outputs small JSON under
the ignored runpod_results directory; no annotation or historical prediction is
read or changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
CELL = Path.home() / "Desktop/Organoid/Membrane/CF/cell_21422_NZ_100"
MODELS = Path.home() / "Desktop/Organoid/Membrane/cellpose_seg/models"
OUT = ROOT / "runpod_results/membrane_v4_v7_cf100_20261023"
Z_COUNT = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bounds(array: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(array.astype(np.float32), (1, 99))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile_path = OUT / "z_intensity_profile.json"
    manifest_path = OUT / "input_manifest.json"
    if profile_path.exists() or manifest_path.exists():
        raise FileExistsError("Prepared profile/manifest already exist")
    files = {name: path for name, path in (
        ("Memb.tif", CELL / "Memb.tif"), ("fluo.tif", CELL / "fluo.tif"),
        ("cpsam_finetuned_v4", MODELS / "cpsam_finetuned_v4"),
        ("cpsam_finetuned_v7", MODELS / "cpsam_finetuned_v7"),
        ("membrane_local11_scope_gpu.py", ROOT / "experiments/membrane_local11_scope_gpu.py"),
        ("infer_cf100.py", ROOT / "experiments/membrane_v4_v7_cf100_infer.py"))}
    manifest = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in files.items()}
    profile = {"scope": "raw pixels only; no labels/predictions",
               "xy_sample_stride": 5, "window_radius_z": 5,
               "global_z_definition": "center z uses max(0,z-5):min(100,z+6), independent of inference batches",
               "channels": {}}
    for channel in ("Memb", "fluo"):
        with tifffile.TiffFile(CELL / f"{channel}.tif") as pages:
            if pages.series[0].shape != (Z_COUNT, 1200, 1200):
                raise ValueError(f"Unexpected {channel} stack shape")
            sampled = np.stack([pages.pages[z].asarray()[::5, ::5] for z in range(Z_COUNT)])
        local = [bounds(sampled[max(0, z - 5):min(Z_COUNT, z + 6)])
                 for z in range(Z_COUNT)]
        profile["channels"][channel] = {
            "source_sha256": manifest[f"{channel}.tif"]["sha256"],
            "window11_p1_p99": local,
        }
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    manifest.update({"z_intensity_profile.json": {"sha256": sha256(profile_path),
                                                   "bytes": profile_path.stat().st_size}})
    record = {"purpose": "out-of-development CF full-stack v4/v7 application demo",
              "cell": str(CELL), "shape_zyx": [Z_COUNT, 1200, 1200],
              "channel_order": ["Memb", "fluo"], "cellpose": "4.2.1.1",
              "normalization": "global-Z-indexed local11 p1/p99, no clipping",
              "chunk_size_z": 10, "stitch_threshold": .25,
              "human_labels_in_payload": False, "files": manifest}
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path),
                      "input_bytes": sum(item["bytes"] for item in manifest.values()),
                      "model_sha256": {key: manifest[key]["sha256"]
                                       for key in ("cpsam_finetuned_v4", "cpsam_finetuned_v7")}}))


if __name__ == "__main__":
    main()
