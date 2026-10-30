"""Generate paired-channel, Z-supported missing-ER *review candidates*.

These outputs are not labels. ER and Fluo must both support a tubular signal,
and a neighbor Z slice must support its location. Existing positives are never
removed. No validation/test organoid is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import meijering
import tifffile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99.8))
    return np.clip((image.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1)


def line_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    im = normalize(image)
    ridge = meijering(gaussian_filter(im, 0.7), sigmas=(1, 2, 3),
                      black_ridges=False).astype(np.float32)
    highpass = np.maximum(gaussian_filter(im, 0.7) - gaussian_filter(im, 7), 0)
    return ridge, highpass


def seed_gate(feature: np.ndarray, seed: np.ndarray, floor: float) -> float:
    """Calibrate to existing positive examples in this *training* crop only."""
    values = feature[seed]
    return max(floor, float(np.percentile(values, 20))) if values.size else floor


def keep_components(mask: np.ndarray, minimum: int = 12) -> np.ndarray:
    components, number = label(mask)
    if not number:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return (components > 0) & (sizes[components] >= minimum)


def display(image: np.ndarray) -> np.ndarray:
    return np.uint8(normalize(image) * 255)


def run(crops: Path, source_manifests: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = {p.parent.name: json.loads(p.read_text())
              for p in source_manifests.rglob("manifest.json")}
    crop_manifest = json.loads((crops / "manifest.json").read_text())
    if crop_manifest.get("training_ready") is not False or crop_manifest.get("scope") != \
            "six training organoids only; validation and test unopened":
        raise ValueError("Crop manifest is not the locked training-only review set")
    records = crop_manifest["records"]
    if len(records) != 18 or len({r["cell"] for r in records}) != 6:
        raise ValueError("Expected the 18 locked training-only crops")
    output.mkdir(parents=True)
    results = []
    panels = []
    for record in records:
        cell = record["cell"]
        full = source[cell]
        if record["split"] != "train" or full["source_z_start_zero_based"] != 0:
            raise ValueError(f"Non-training or partial source: {cell}")
        directory = crops / record["directory"]
        names = {"ER": "ER_reference", "Fluo": "Fluo_input",
                 "seed": "observed_ER_positive_seed_NOT_COMPLETE_LABEL"}
        for key, stem in names.items():
            if sha256(directory / f"{stem}.tif") != record["crop_sha256"][stem]:
                raise ValueError(f"Crop hash changed: {directory / stem}")
        er = tifffile.imread(directory / "ER_reference.tif")
        fluo = tifffile.imread(directory / "Fluo_input.tif")
        seed = tifffile.imread(directory / "observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif") > 0
        z = record["z_zero_based"]
        y, x = record["y0"], record["x0"]
        sl = np.s_[y:y + 384, x:x + 384]
        er_path, fluo_path = Path(full["source_er"]), Path(full["source_fluo"])
        if sha256(er_path) != record["source_sha256"]["ER"] or \
                sha256(fluo_path) != record["source_sha256"]["Fluo"]:
            raise ValueError(f"Source hash changed for {cell}")
        er_r, er_h = line_features(er)
        fl_r, fl_h = line_features(fluo)
        er_r_min, er_h_min = seed_gate(er_r, seed, 0.025), seed_gate(er_h, seed, 0.004)
        fl_r_min, fl_h_min = seed_gate(fl_r, seed, 0.025), seed_gate(fl_h, seed, 0.004)
        current = ((er_r >= er_r_min) & (er_h >= er_h_min) &
                   (fl_r >= fl_r_min) & (fl_h >= fl_h_min))
        support = np.zeros_like(seed)
        with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fl_file:
            for neighbor in (z - 1, z + 1):
                if neighbor < 0 or neighbor >= full["source_shape"][0]:
                    continue
                er_n_r, er_n_h = line_features(er_file.pages[neighbor].asarray()[sl])
                fl_n_r, fl_n_h = line_features(fl_file.pages[neighbor].asarray()[sl])
                supported = ((er_n_r >= 0.8 * er_r_min) & (er_n_h >= 0.8 * er_h_min) &
                             (fl_n_r >= 0.8 * fl_r_min) & (fl_n_h >= 0.8 * fl_h_min))
                support |= binary_dilation(supported, iterations=2)
        new = keep_components(current & support & ~seed)
        # Separate the recommendation from the old mask: neither is asserted GT.
        dest = output / record["directory"]
        dest.mkdir()
        new_path = dest / "candidate_additions_NOT_LABEL.tif"
        tifffile.imwrite(new_path, new.astype(np.uint8) * 255)
        rgb = np.repeat(display(fluo)[..., None], 3, axis=2)
        rgb[seed] = (250, 215, 0)
        rgb[new] = (0, 230, 235)
        panel = np.concatenate((np.repeat(display(er)[..., None], 3, axis=2),
                                np.repeat(display(fluo)[..., None], 3, axis=2), rgb), axis=1)
        Image.fromarray(panel).save(dest / "review_ER_Fluo_old_yellow_new_cyan.png")
        panels.append(Image.fromarray(panel).resize((576, 192)))
        results.append({"directory": record["directory"], "cell": cell,
                        "z_zero_based": z, "old_candidate_pixels": int(seed.sum()),
                        "new_candidate_pixels": int(new.sum()),
                        "thresholds": {"ER_ridge": er_r_min, "ER_highpass": er_h_min,
                                       "Fluo_ridge": fl_r_min, "Fluo_highpass": fl_h_min},
                        "candidate_sha256": sha256(new_path),
                        "status": "automatic review candidate, not a label"})
    sheet = Image.new("RGB", (576, 192 * len(panels)))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, 192 * i))
    sheet.save(output / "all_18_review.png")
    manifest = {"source_crop_manifest_sha256": sha256(crops / "manifest.json"),
                "method": "paired ER/Fluo ridge and local contrast calibrated to old training positives; ±1 Z support; min 12 px components",
                "semantics": "yellow old positives, cyan suggested additions; both need review; zeros are unknown",
                "training_ready": False, "records": results}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--source-manifests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.crops, a.source_manifests, a.output)
    print(json.dumps({"crops": len(result["records"]),
                      "suggested_additional_pixels": sum(r["new_candidate_pixels"] for r in result["records"]),
                      "output": str(a.output)}))
