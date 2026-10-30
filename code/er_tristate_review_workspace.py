"""Prepare an editable, explicitly UNREVIEWED three-state ER label workspace.

Codes: 0 = definite non-ER, 1 = candidate ER positive, 255 = unknown/ignore.
The initial draft has old candidate positives and NO asserted negatives.
Nothing here is training-ready until each ROI receives a documented review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(crops: Path, candidates: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    crop_manifest_path, cand_manifest_path = crops / "manifest.json", candidates / "manifest.json"
    crop_manifest = json.loads(crop_manifest_path.read_text())
    cand_manifest = json.loads(cand_manifest_path.read_text())
    if crop_manifest.get("training_ready") is not False or cand_manifest.get("training_ready") is not False:
        raise ValueError("Review inputs must not be training-ready")
    candidates_by_dir = {r["directory"]: r for r in cand_manifest["records"]}
    if len(crop_manifest["records"]) != 18 or len(candidates_by_dir) != 18:
        raise ValueError("Expected 18 locked training-only crops and candidates")
    output.mkdir(parents=True)
    records = []
    for row in crop_manifest["records"]:
        if row["split"] != "train" or row["directory"] not in candidates_by_dir:
            raise ValueError(f"Unexpected split or candidate: {row['directory']}")
        name = row["directory"]
        src = crops / name / "observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif"
        if sha(src) != row["crop_sha256"]["observed_ER_positive_seed_NOT_COMPLETE_LABEL"]:
            raise ValueError(f"Old seed hash mismatch: {src}")
        seed = tifffile.imread(src)
        if seed.shape != (384, 384) or seed.dtype != np.uint8 or not np.isin(seed, (0, 255)).all():
            raise ValueError(f"Invalid seed: {src}")
        proposal = np.full(seed.shape, 255, dtype=np.uint8)
        proposal[seed > 0] = 1
        target = output / name
        target.mkdir()
        label_path = target / "tri_state_INITIAL_UNREVIEWED.tif"
        tifffile.imwrite(label_path, proposal)
        records.append({"directory": name, "cell": row["cell"], "split": "train",
                        "z_zero_based": row["z_zero_based"], "y0": row["y0"], "x0": row["x0"],
                        "label_path": str(label_path.relative_to(output)),
                        "label_sha256": sha(label_path),
                        "candidate_positive_pixels": int((proposal == 1).sum()),
                        "definite_negative_pixels": 0,
                        "unknown_pixels": int((proposal == 255).sum()),
                        "review_status": "UNREVIEWED; never train from this draft"})
    manifest = {"source_crop_manifest_sha256": sha(crop_manifest_path),
                "source_candidate_manifest_sha256": sha(cand_manifest_path),
                "codes": {"0": "definite non-ER", "1": "candidate ER positive",
                          "255": "unknown/ignore"},
                "training_ready": False, "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# ER three-state review workspace (UNREVIEWED)\n\n"
        "Each crop's TIFF is aligned with the source ER/Fluo crop of the same "
        "directory name. Load `Fluo_input.tif`, `ER_reference.tif`, the initial "
        "tri-state TIFF, and the cyan additions proposal in napari as layers. "
        "Initial 1 pixels copy old candidate positives; every other pixel is "
        "255 (unknown), and there are no confirmed negatives. Add missing "
        "Fluo-visible ER-like lines after checking paired ER and adjacent Z; "
        "mark only confidently non-ER as 0; leave ambiguous areas 255. "
        "Save a NEW reviewed TIFF and reviewer notes. Do not overwrite this "
        "initial draft, train from it, or call automatic candidates ground truth. "
        "Validation/test organoids are absent.\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.crops, args.candidates, args.output)
    print(json.dumps({"output": str(args.output), "crops": len(result["records"]),
                      "training_ready": result["training_ready"]}))
