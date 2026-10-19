"""Package the verified ten-cell Nuc predictions as mask-only delivery ZIPs.

This script reads the locked batch plan and the already published Desktop
predictions. It verifies all 1,330 source/mask pairs before writing anything to
``deliverables/``. ZIPs contain generated masks and portable provenance only:
never raw fluorescence images, the Keras checkpoint, remote/local paths, or the
full prediction manifests. Existing differing deliverables are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runpod_results/nuc_bulk_20261019"
CELL_COUNTS = {
    "cell_22375_NZ_100": 100,
    "cell_22376_NZ_100": 100,
    "cell_22377_NZ_120": 120,
    "cell_22378_NZ_150": 150,
    "cell_22379_NZ_150": 150,
    "cell_22380_150": 150,
    "cell_22381_140": 140,
    "cell_22382_140": 140,
    "cell_22383_140": 140,
    "cell_22385_140": 140,
}
OUTPUT_NAME = "Nuc_label"
ARCHIVE_NAME = "Nuc_label.zip"
METHOD = "nuc_review_2d_low003_core045_min500_close4_curved_split_v1"
SCOPE = "automatic masks for expert review, not manual ground truth"
MODEL_SHA256 = "c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e"
SETTINGS = {
    "low_threshold": 0.03,
    "core_threshold": 0.45,
    "normalization_percentiles": [1.0, 99.8],
    "tile_size": 512,
    "overlap": 128,
    "prediction_batch_size": 4,
    "foreground_value": 255,
}
SOURCE_NAMES = {
    "code/nuc_predict_review.py",
    "code/nuc_postprocess.py",
    "code/sliding_window.py",
    "code/tiff_io.py",
    "reference/nuc_unet.keras",
    "batch/bulk.py",
    "batch/remote_run.sh",
    "batch/setup_env.sh",
    "batch/start_runner.sh",
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
NAME_RE = re.compile(r"fluo_[0-9]{3}\.tif\Z")
ZIP_TIMESTAMP = (2026, 10, 19, 0, 0, 0)
MAX_GIT_BLOB_BYTES = 100_000_000


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_stream(source: object) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked file: {path}")


def regular_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Missing or symlinked directory: {path}")


def source_path(name: str, run_root: Path) -> Path:
    if name.startswith("code/"):
        return PROJECT_ROOT / name
    if name == "reference/nuc_unet.keras":
        return PROJECT_ROOT / "runpod_results/nuc_internal_20261015/full_results/model/nuc_unet.keras"
    return run_root / name.removeprefix("batch/")


def checked_tiff(path: Path, *, mask: bool) -> np.ndarray:
    regular_file(path)
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != 1 or len(tiff.series) != 1:
            raise ValueError(f"Expected one independent 2D TIFF: {path}")
        page = tiff.pages[0]
        if mask and page.photometric != tifffile.PHOTOMETRIC.MINISBLACK:
            raise ValueError(f"Mask is not MINISBLACK: {path}")
        image = page.asarray()
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D image: {path}")
    if mask and (image.dtype != np.uint8 or not np.isin(image, (0, 255)).all()):
        raise ValueError(f"Expected uint8 mask with only 0/255: {path}")
    return image


def verify_plan(plan: dict, plan_path: Path, data_root: Path) -> tuple[list[dict], dict]:
    """Reject any changed batch scope, science, source code or model bytes."""
    if (plan.get("schema_version") != 1 or plan.get("output_name") != OUTPUT_NAME
            or plan.get("method") != METHOD or plan.get("scope") != SCOPE
            or plan.get("settings") != SETTINGS
            or Path(str(plan.get("data_root", ""))).expanduser().resolve() != data_root):
        raise ValueError("Locked Nuc batch plan or data root differs")
    hashes = plan.get("source_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != SOURCE_NAMES
            or hashes.get("reference/nuc_unet.keras") != MODEL_SHA256
            or any(not isinstance(h, str) or SHA_RE.fullmatch(h) is None
                   for h in hashes.values())):
        raise ValueError("Invalid model/code hashes in locked plan")
    for name, expected in sorted(hashes.items()):
        path = source_path(name, plan_path.parent)
        regular_file(path)
        if digest_file(path) != expected:
            raise ValueError(f"Locked code/model source changed: {name}")
    entries = plan.get("entries")
    if (not isinstance(entries, list) or len(entries) != len(CELL_COUNTS)
            or [e.get("cell") for e in entries if isinstance(e, dict)] != sorted(CELL_COUNTS)):
        raise ValueError("Plan must contain exactly the ten locked cells")
    for entry in entries:
        cell = entry["cell"]
        records = entry.get("files")
        if not isinstance(records, list) or len(records) != CELL_COUNTS[cell]:
            raise ValueError(f"Wrong planned image count: {cell}")
        names = [r.get("filename") for r in records if isinstance(r, dict)]
        if (len(names) != len(records) or names != sorted(names)
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or NAME_RE.fullmatch(name) is None
                       for name in names)):
            raise ValueError(f"Invalid filenames in plan: {cell}")
        for record in records:
            if (record.get("shape") != [1200, 1200]
                    or not isinstance(record.get("size_bytes"), int)
                    or record["size_bytes"] <= 0
                    or not isinstance(record.get("sha256"), str)
                    or SHA_RE.fullmatch(record["sha256"]) is None):
                raise ValueError(f"Invalid planned source record: {cell}")
    return entries, hashes


def verify_cell(entry: dict, data_root: Path, plan_hash: str, hashes: dict) -> tuple[dict, list[tuple[str, Path, str]]]:
    """Read and hash every input and mask; keep only redacted provenance."""
    cell = entry["cell"]
    input_dir = data_root / cell / "fluo"
    mask_dir = data_root / cell / OUTPUT_NAME
    regular_dir(input_dir)
    regular_dir(mask_dir)
    records = entry["files"]
    names = [record["filename"] for record in records]
    input_tiffs = {p.name for p in input_dir.iterdir()
                   if p.suffix.lower() in {".tif", ".tiff"} and not p.name.startswith("._")}
    if input_tiffs != set(names):
        raise ValueError(f"Input TIFF set differs from plan: {cell}")
    if {p.name for p in mask_dir.iterdir()} != set(names) | {"prediction_manifest.json"}:
        raise ValueError(f"Unexpected or missing Nuc mask files: {cell}")
    manifest_path = mask_dir / "prediction_manifest.json"
    regular_file(manifest_path)
    manifest = load_object(manifest_path)
    expected = {
        "status": "complete", "method": METHOD, "scope": SCOPE,
        "model_sha256": MODEL_SHA256, **SETTINGS,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Published Nuc method/model/settings changed: {cell}")
    if (not str(manifest.get("input_dir", "")).endswith(f"/inputs/{cell}/fluo")
            or not str(manifest.get("output_dir", "")).endswith(
                f"/outputs/{cell}/{OUTPUT_NAME}")):
        raise ValueError(f"Manifest source/output batch paths differ: {cell}")
    rows = manifest.get("files")
    if (not isinstance(rows, list) or len(rows) != len(records)
            or [r.get("filename") for r in rows if isinstance(r, dict)] != names):
        raise ValueError(f"Manifest image count/order differs from plan: {cell}")
    portable_rows = []
    mask_files = []
    for source_record, row in zip(records, rows):
        name = source_record["filename"]
        input_path = input_dir / name
        mask_path = mask_dir / name
        input_image = checked_tiff(input_path, mask=False)
        mask_image = checked_tiff(mask_path, mask=True)
        height, width = source_record["shape"]
        mask_hash = digest_file(mask_path)
        foreground = int(np.count_nonzero(mask_image))
        postprocessing = row.get("postprocessing")
        if (list(input_image.shape) != [height, width]
                or list(mask_image.shape) != [height, width]
                or input_path.stat().st_size != source_record["size_bytes"]
                or digest_file(input_path) != source_record["sha256"]
                or row.get("input_sha256") != source_record["sha256"]
                or row.get("output_sha256") != mask_hash
                or row.get("height") != height or row.get("width") != width
                or row.get("foreground_pixels") != foreground
                or not isinstance(postprocessing, dict)
                or postprocessing.get("shape") != [height, width]
                or postprocessing.get("final_foreground_pixels") != foreground):
            raise ValueError(f"Source/mask/manifest mismatch: {cell}/{name}")
        portable_rows.append({
            "filename": name,
            "height": height,
            "width": width,
            "foreground_pixels": foreground,
            "input_sha256": source_record["sha256"],
            "mask_sha256": mask_hash,
        })
        mask_files.append((name, mask_path, mask_hash))
    portable = {
        "schema_version": 1,
        "cell": cell,
        "target": "nuc",
        "status": "automatic_prediction_for_human_review_not_ground_truth",
        "method": METHOD,
        "settings": SETTINGS,
        "model_sha256": MODEL_SHA256,
        "code_sha256": {key: hashes[key] for key in sorted(hashes) if key.startswith("code/")},
        "batch_plan_sha256": plan_hash,
        "source_prediction_manifest_sha256": digest_file(manifest_path),
        "mask_count": len(mask_files),
        "files": portable_rows,
    }
    return portable, mask_files


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_archive(path: Path, portable: dict, masks: list[tuple[str, Path, str]]) -> dict[str, bytes]:
    """Write an ordered ZIP with fixed metadata, then check every member."""
    manifest_bytes = json_bytes(portable)
    sums = "".join(f"{digest}  {name}\n" for name, _, digest in masks).encode("ascii")
    explanation = (
        "Nuc 2D model predictions for expert review; these are not manual labels.\n"
        "Unzip into a new location to avoid replacing existing results.\n"
        "Each mask matches its fluo source filename and 1200x1200 pixel grid.\n"
        "Single-channel uint8: background 0 (black), nucleus 255 (white).\n"
        "Run `shasum -a 256 -c SHA256SUMS.txt` from this Nuc_label folder.\n"
    ).encode("ascii")
    metadata = {
        f"{OUTPUT_NAME}/PORTABLE_MANIFEST.json": manifest_bytes,
        f"{OUTPUT_NAME}/SHA256SUMS.txt": sums,
        f"{OUTPUT_NAME}/AUTO_LABELS_README.txt": explanation,
    }
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as archive:
        for name, source, _ in masks:
            with source.open("rb") as src, archive.open(zip_info(f"{OUTPUT_NAME}/{name}"), "w") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        for name, content in metadata.items():
            archive.writestr(zip_info(name), content, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    if path.stat().st_size >= MAX_GIT_BLOB_BYTES:
        raise ValueError(f"ZIP exceeds GitHub's 100 MB blob limit: {path}")
    with zipfile.ZipFile(path) as archive:
        expected_names = [f"{OUTPUT_NAME}/{name}" for name, _, _ in masks] + list(metadata)
        if archive.namelist() != expected_names or archive.testzip() is not None:
            raise ValueError(f"ZIP member order, set, or CRC differs: {path}")
        for info in archive.infolist():
            if info.is_dir() or info.file_size >= MAX_GIT_BLOB_BYTES:
                raise ValueError(f"Unexpected ZIP member: {info.filename}")
        for name, _, digest in masks:
            with archive.open(f"{OUTPUT_NAME}/{name}") as member:
                if digest_stream(member) != digest:
                    raise ValueError(f"ZIP mask SHA256 differs: {path}/{name}")
        for name, expected in metadata.items():
            if archive.read(name) != expected:
                raise ValueError(f"ZIP metadata differs: {path}/{name}")
    return metadata


def existing_tree_ok(staged: Path, destination: Path) -> None:
    """Preflight the complete delivery before installing any new file."""
    if not os.path.lexists(destination):
        return
    regular_dir(destination)
    expected_top = {p.name for p in staged.iterdir()}
    unexpected = {p.name for p in destination.iterdir()} - expected_top - {"README.md"}
    if unexpected:
        raise ValueError(f"Unexpected existing Nuc deliverables: {sorted(unexpected)}")
    for staged_cell in (path for path in staged.iterdir() if path.is_dir()):
        delivered_cell = destination / staged_cell.name
        if not os.path.lexists(delivered_cell):
            continue
        regular_dir(delivered_cell)
        unexpected = {p.name for p in delivered_cell.iterdir()} - {
            p.name for p in staged_cell.iterdir()
        }
        if unexpected:
            raise ValueError(f"Unexpected files in {delivered_cell}: {sorted(unexpected)}")
    for path in staged.rglob("*"):
        target = destination / path.relative_to(staged)
        if not os.path.lexists(target):
            continue
        if path.is_dir():
            regular_dir(target)
        elif target.is_symlink() or not target.is_file() or digest_file(target) != digest_file(path):
            raise FileExistsError(f"Existing Nuc deliverable differs; not replacing: {target}")


def install_tree(staged: Path, destination: Path) -> None:
    """Create only missing files with no-clobber hard links on the same volume."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(staged.rglob("*")):
        target = destination / path.relative_to(staged)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif not os.path.lexists(target):
            os.link(path, target)
        elif target.is_symlink() or digest_file(target) != digest_file(path):
            raise FileExistsError(f"Existing Nuc deliverable differs; not replacing: {target}")
    existing_tree_ok(staged, destination)


def package(plan_path: Path, data_root: Path, destination: Path) -> None:
    plan_path = plan_path.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    destination = destination.expanduser().absolute()
    regular_file(plan_path)
    regular_dir(data_root)
    if destination.is_symlink():
        raise ValueError(f"Symlinked output directory: {destination}")
    plan = load_object(plan_path)
    entries, hashes = verify_plan(plan, plan_path, data_root)
    plan_hash = digest_file(plan_path)
    # Complete all source and mask preflight before writing the first ZIP.
    checked = [verify_cell(entry, data_root, plan_hash, hashes) for entry in entries]
    if sum(len(masks) for _, masks in checked) != 1330:
        raise ValueError("The ten-cell Nuc delivery must contain exactly 1,330 masks")

    regular_dir(destination.parent)
    with tempfile.TemporaryDirectory(prefix=".nuc_auto_labels_stage_",
                                     dir=destination.parent) as temporary:
        staged = Path(temporary)
        index_rows = []
        top_checksums = []
        for entry, (portable, masks) in zip(entries, checked):
            cell = entry["cell"]
            folder = staged / cell
            folder.mkdir()
            archive = folder / ARCHIVE_NAME
            build_archive(archive, portable, masks)
            digest = digest_file(archive)
            (folder / f"{ARCHIVE_NAME}.sha256").write_text(
                f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii")
            top_checksums.append(f"{digest}  {cell}/{ARCHIVE_NAME}\n")
            index_rows.append({
                "cell": cell,
                "mask_count": len(masks),
                "archive": f"{cell}/{ARCHIVE_NAME}",
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": digest,
                "portable_manifest_sha256": hashlib.sha256(json_bytes(portable)).hexdigest(),
            })
        index = {
            "schema_version": 1,
            "target": "nuc",
            "delivery_status": "automatic_predictions_for_expert_review_not_ground_truth",
            "method": METHOD,
            "model_sha256": MODEL_SHA256,
            "batch_plan_sha256": plan_hash,
            "cells": index_rows,
            "total_masks": sum(row["mask_count"] for row in index_rows),
        }
        (staged / "PACKAGE_INDEX.json").write_bytes(json_bytes(index))
        (staged / "SHA256SUMS.txt").write_text("".join(top_checksums), encoding="ascii")
        existing_tree_ok(staged, destination)
        install_tree(staged, destination)
    print(f"NUC_DELIVERY_READY cells=10 masks=1330 path={destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_RUN_ROOT / "batch_plan.json")
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "Desktop/Organoid/organoid_data")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "deliverables/nuc_auto_labels")
    args = parser.parse_args()
    package(args.plan, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
