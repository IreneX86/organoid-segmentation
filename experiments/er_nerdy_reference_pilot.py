"""Compare public nERdy+ ER-channel predictions on three TRAIN-only crops.

The external AGPL model/code are loaded from a separate, user-specified clone
and are not copied into this project. Outputs are review candidates, not GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile


SELECTED = (
    "cell_20260814_23298_NZ_50_z012_y272_x544",
    "cell_20260904_23965_NZ_140_z070_y272_x272",
    "cell_20260904_23966_NZ_100_z050_y272_x272",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99.8))
    return np.clip((image.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0, 1).astype(np.float32)


def overlay(image: np.ndarray, old: np.ndarray, external: np.ndarray) -> Image.Image:
    rgb = np.repeat((normalize(image) * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[old] = (255, 220, 0)
    rgb[external & ~old] = (0, 225, 235)
    return Image.fromarray(rgb)


def main(crops: Path, clone: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((crops / "manifest.json").read_text())
    index = {r["directory"]: r for r in manifest["records"]}
    if manifest["training_ready"] is not False or not set(SELECTED).issubset(index):
        raise ValueError("Expected the locked training-only crop set")
    for name in SELECTED:
        if index[name]["split"] != "train":
            raise ValueError(f"Non-training crop: {name}")
    model_path = clone / "nERdy+" / "NNet_groupy_p4m_v2_VecAdam.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    # Compatibility aliases are local to this process; external files stay unchanged.
    np.int = int  # type: ignore[attr-defined]
    np.float = float  # type: ignore[attr-defined]
    np.complex = complex  # type: ignore[attr-defined]
    sys.path.insert(0, str(clone / "nERdy+"))
    import torch
    from model import D4nERdy  # type: ignore[import-not-found]
    from postprocessing import postprocessing  # type: ignore[import-not-found]

    torch.set_num_threads(4)
    model = D4nERdy(1, 1)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    commit = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
    output.mkdir(parents=True)
    rows = []
    contact = Image.new("RGB", (384 * 4, 410 * len(SELECTED)), "white")
    for i, name in enumerate(SELECTED):
        src = crops / name
        row = index[name]
        stems = ("ER_reference", "Fluo_input", "observed_ER_positive_seed_NOT_COMPLETE_LABEL")
        for stem in stems:
            if sha(src / f"{stem}.tif") != row["crop_sha256"][stem]:
                raise ValueError(f"Crop changed: {src / stem}")
        er = tifffile.imread(src / "ER_reference.tif")
        fluo = tifffile.imread(src / "Fluo_input.tif")
        old = tifffile.imread(src / "observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif") > 0
        with torch.no_grad():
            tensor = torch.from_numpy(normalize(er)[None, None, :, :])
            probability = torch.sigmoid(model(tensor)).squeeze().numpy()
        span = float(probability.max() - probability.min())
        scaled = (probability - probability.min()) / max(span, 1e-8)
        external = postprocessing(scaled) > 0
        tif_path = output / f"{name}_nERdy_external_CANDIDATE.tif"
        tifffile.imwrite(tif_path, external.astype(np.uint8) * 255)
        for j, frame in enumerate((Image.fromarray((normalize(er) * 255).astype(np.uint8)).convert("RGB"),
                                   Image.fromarray((normalize(fluo) * 255).astype(np.uint8)).convert("RGB"),
                                   overlay(er, old, external), overlay(fluo, old, external))):
            contact.paste(frame, (j * 384, i * 410 + 26))
        ImageDraw.Draw(contact).text((0, i * 410 + 4),
                                     f"{name} | ER / Fluo / ER+old(yellow)+nERdy-only(cyan) / Fluo+overlay",
                                     fill="black")
        rows.append({"crop": name, "cell": row["cell"], "split": "train",
                     "old_candidate_pixels": int(old.sum()),
                     "nERdy_candidate_pixels": int(external.sum()),
                     "nERdy_only_pixels": int((external & ~old).sum()),
                     "old_only_pixels": int((old & ~external).sum()),
                     "candidate_sha256": sha(tif_path),
                     "status": "external ER-channel candidate, NOT ground truth"})
    contact_path = output / "three_train_crops_review.png"
    contact.save(contact_path)
    report = {"model": "nERdy+ public pretrained ER-channel model",
              "external_repo": "https://github.com/NanoscopyAI/nERdy",
              "external_commit": commit, "external_weights_sha256": sha(model_path),
              "input_preprocessing": "each ER crop scaled at its own 1st-99.8th percentiles; exploratory adaptation",
              "postprocessing": "authors' rolling-ball background subtraction then Otsu",
              "warning": "different imaging domain; no manually corrected GT; Fluo-only inference NOT tested",
              "training_ready": False, "records": rows}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--clone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    result = main(a.crops, a.clone, a.output)
    print(json.dumps({"crops": len(result["records"]), "output": str(a.output)}))
