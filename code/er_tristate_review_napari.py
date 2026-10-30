"""Open one ER/Fluo crop for human three-state review in napari.

This editor saves immutable *drafts*, never training-ready labels. The old
pseudo-mask is only an editable starting proposal, not approved ground truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(crop_root: Path, proposal_root: Path, draft_root: Path, crop_name: str) -> dict:
    record = next((r for r in json.loads((crop_root / "manifest.json").read_text())["records"]
                   if r["directory"] == crop_name), None)
    if record is None or record["split"] != "train":
        raise ValueError("Select a crop from the locked training-only manifest")
    source = crop_root / crop_name
    draft = draft_root / crop_name / "tri_state_INITIAL_UNREVIEWED.tif"
    er_path, fluo_path = source / "ER_reference.tif", source / "Fluo_input.tif"
    old_path = source / "observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif"
    add_path = proposal_root / crop_name / "candidate_additions_NOT_LABEL.tif"
    for path, stem in ((er_path, "ER_reference"), (fluo_path, "Fluo_input"),
                       (old_path, "observed_ER_positive_seed_NOT_COMPLETE_LABEL")):
        if sha(path) != record["crop_sha256"][stem]:
            raise ValueError(f"Source changed: {path}")
    entry = next(r for r in json.loads((proposal_root / "manifest.json").read_text())["records"]
                 if r["directory"] == crop_name)
    if sha(add_path) != entry["candidate_sha256"]:
        raise ValueError("Addition candidate changed")
    draft_entry = next(r for r in json.loads((draft_root / "manifest.json").read_text())["records"]
                       if r["directory"] == crop_name)
    if sha(draft) != draft_entry["label_sha256"]:
        raise ValueError("Initial draft changed")
    er, fluo, old, addition, tri = (tifffile.imread(path) for path in
                                    (er_path, fluo_path, old_path, add_path, draft))
    if any(a.shape != (384, 384) for a in (er, fluo, old, addition, tri)):
        raise ValueError("Crop shape mismatch")
    if not np.isin(tri, (1, 255)).all():
        raise ValueError("Initial draft has unexpected state")
    return {"record": record, "er": er, "fluo": fluo, "old": old,
            "addition": addition, "editable": np.where(tri == 1, 1, 0).astype(np.uint8),
            "initial_draft_sha256": sha(draft)}


def save_draft(editable: np.ndarray, source: dict, target_root: Path, crop_name: str) -> Path:
    if editable.shape != (384, 384) or not np.isin(editable, (0, 1, 2)).all():
        raise ValueError("Editable labels must use 0 unknown, 1 positive, 2 negative")
    target_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = target_root / f"{crop_name}_{timestamp}_DRAFT.tif"
    tri = np.full(editable.shape, 255, dtype=np.uint8)
    tri[editable == 1] = 1
    tri[editable == 2] = 0
    tifffile.imwrite(dest, tri)
    metadata = {"crop": crop_name, "created_utc": timestamp,
                "source_initial_draft_sha256": source["initial_draft_sha256"],
                "new_draft_sha256": sha(dest), "positive_pixels": int((tri == 1).sum()),
                "negative_pixels": int((tri == 0).sum()),
                "unknown_pixels": int((tri == 255).sum()),
                "review_status": "DRAFT; not independently checked; training_ready=false"}
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = load(args.crops, args.candidates, args.initial, args.crop)
    if args.dry_run:
        print(json.dumps({"crop": args.crop, "shape": list(bundle["editable"].shape),
                          "old_positive_pixels": int((bundle["editable"] == 1).sum()),
                          "new_candidate_pixels": int((bundle["addition"] > 0).sum())}))
        return
    import napari

    viewer = napari.Viewer(title=f"ER tri-state review: {args.crop}")
    viewer.add_image(bundle["fluo"], name="Fluo: deployment input", colormap="gray")
    viewer.add_image(bundle["er"], name="ER: registered reference", colormap="gray", visible=False)
    viewer.add_labels((bundle["addition"] > 0).astype(np.uint8),
                      name="auto additions: proposal only", visible=False, opacity=0.45)
    viewer.add_labels((bundle["old"] > 0).astype(np.uint8),
                      name="old seed: proposal only", visible=False, opacity=0.45)
    editable_layer = viewer.add_labels(bundle["editable"],
                                       name="EDIT: 0 unknown, 1 ER, 2 definite non-ER", opacity=0.55)
    editable_layer.selected_label = 1

    @viewer.bind_key("s")
    def save_current(_viewer) -> None:
        saved = save_draft(np.asarray(editable_layer.data), bundle, args.save_root, args.crop)
        print(f"Saved UNREVIEWED DRAFT {saved}", flush=True)

    print("Paint label 1 for ER, 2 for definite non-ER; erase to 0 (unknown). Press S to save an immutable DRAFT.")
    napari.run()


if __name__ == "__main__":
    main()
