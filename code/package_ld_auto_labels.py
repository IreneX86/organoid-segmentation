"""Package the ten reviewed LD prediction folders for private Git delivery.

Only generated ``fluo_label/*.tif`` masks are archived. The original fluo
images, old ``fluo_label_previous`` output, absolute source paths, full
prediction manifests, checkpoints, and local RunPod logs never enter ZIPs.
Each archive receives a portable manifest and per-mask SHA256 list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import tifffile


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
METHOD = "five_fold_binary_vote_at_locked_validation_thresholds_majority_3_of_5"
THRESHOLDS = (0.65, 0.80, 0.60, 0.35, 0.50)
ZIP_TIMESTAMP = (2026, 10, 15, 0, 0, 0)
MASK_NAME = re.compile(r"fluo_\d{3}\.tif", re.IGNORECASE)
MAX_GIT_BLOB_BYTES = 100_000_000


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_entry(name: str, content: bytes, archive: zipfile.ZipFile) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED,
                     compresslevel=6)


def verify_source(cell: str, data_root: Path) -> tuple[list[dict], dict, str]:
    folder = data_root / cell / "fluo_label"
    input_folder = data_root / cell / "fluo"
    if (folder.is_symlink() or input_folder.is_symlink()
            or not folder.is_dir() or not input_folder.is_dir()):
        raise ValueError(f"Missing or symlinked LD folder: {cell}")
    manifest_path = folder / "prediction_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Missing prediction manifest: {cell}")
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (original.get("status") != "complete" or original.get("target") != "ld"
            or original.get("method") != METHOD
            or original.get("image_count") != CELL_COUNTS[cell]
            or original.get("foreground_value") != 255
            or original.get("not_validated_as_ensemble") is not True):
        raise ValueError(f"Not the expected completed LD focal vote: {cell}")
    folds = original.get("folds", [])
    if (len(folds) != 5 or [f.get("fold") for f in folds] != list(range(5))
            or any(abs(float(f.get("threshold", -1)) - THRESHOLDS[i]) > 1e-8
                   for i, f in enumerate(folds))):
        raise ValueError(f"Fold thresholds do not match the locked review method: {cell}")
    source_files = sorted(p.name for p in input_folder.iterdir()
                          if p.suffix.lower() == ".tif" and not p.name.startswith("._"))
    records = original.get("files", [])
    if not isinstance(records, list) or len(records) != CELL_COUNTS[cell]:
        raise ValueError(f"Incomplete per-image manifest: {cell}")
    by_name = {record.get("filename"): record for record in records
               if isinstance(record, dict)}
    expected = sorted(by_name)
    if len(by_name) != CELL_COUNTS[cell] or expected != source_files:
        raise ValueError(f"Input/output names differ: {cell}")
    if any(not MASK_NAME.fullmatch(name) for name in expected):
        raise ValueError(f"Unexpected TIFF filename: {cell}")
    actual_entries = {p.name for p in folder.iterdir()}
    if actual_entries - set(expected) - {"prediction_manifest.json", "README.txt"}:
        raise ValueError(f"Unexpected output files: {cell}")
    if set(expected) - actual_entries:
        raise ValueError(f"Missing masks: {cell}")

    sanitized = []
    for name in expected:
        mask_path = folder / name
        if mask_path.is_symlink() or not mask_path.is_file():
            raise ValueError(f"Missing or symlinked mask: {cell}/{name}")
        record = by_name[name]
        mask = tifffile.imread(mask_path)
        if (mask.shape != (1200, 1200) or mask.dtype != np.uint8
                or not np.isin(mask, (0, 255)).all()
                or record.get("height") != 1200 or record.get("width") != 1200
                or record.get("foreground_pixels") != int(np.count_nonzero(mask))):
            raise ValueError(f"Mask geometry or values differ: {cell}/{name}")
        digest = sha256_file(mask_path)
        if digest != record.get("mask_sha256"):
            raise ValueError(f"Mask SHA256 differs from prediction manifest: {cell}/{name}")
        sanitized.append({
            "filename": name,
            "sha256": digest,
            "height": 1200,
            "width": 1200,
            "foreground_pixels": int(np.count_nonzero(mask)),
        })
    common_folds = [{
        "fold": fold["fold"],
        "threshold": round(float(fold["threshold"]), 2),
        "checkpoint_sha256": fold["checkpoint_sha256"],
        "selection_lock_sha256": fold["selection_lock_sha256"],
    } for fold in folds]
    return sanitized, {"folds": common_folds}, sha256_file(manifest_path)


def verify_archive(path: Path, records: list[dict]) -> None:
    expected = {f"fluo_label/{record['filename']}": record["sha256"]
                for record in records}
    expected.update({
        "fluo_label/PORTABLE_MANIFEST.json": None,
        "fluo_label/AUTO_LABELS_README.txt": None,
        "fluo_label/SHA256SUMS.txt": None,
    })
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f"Archive members differ from the mask-only plan: {path}")
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC failed: {path}")
        for name, digest in expected.items():
            if digest is not None and sha256_bytes(archive.read(name)) != digest:
                raise ValueError(f"Archived mask SHA256 differs: {name}")


def package_cell(cell: str, data_root: Path, output_root: Path) -> dict:
    records, model, source_manifest_hash = verify_source(cell, data_root)
    archive_dir = output_root / cell
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "fluo_label.zip"
    digest_path = archive_dir / "fluo_label.zip.sha256"
    portable = {
        "schema_version": 1,
        "cell": cell,
        "input_folder": f"organoid_data/{cell}/fluo",
        "output_folder": f"organoid_data/{cell}/fluo_label",
        "method": METHOD,
        "purpose": "automated LD predictions for human review; not manual ground truth",
        "independent_external_accuracy_known": False,
        "mask_count": len(records),
        "mask_dtype": "uint8",
        "background_value": 0,
        "foreground_value": 255,
        "normalization_percentiles": [1.0, 99.8],
        "patch_size": 512,
        "overlap": 128,
        "majority_votes_required": 3,
        "folds": model["folds"],
        "local_prediction_manifest_sha256": source_manifest_hash,
        "files": records,
    }
    manifest_bytes = (json.dumps(portable, indent=2, sort_keys=True) + "\n").encode()
    sha_lines = "".join(f"{r['sha256']}  fluo_label/{r['filename']}\n"
                        for r in records).encode()
    note = (
        "These TIFFs are automated lipid-droplet predictions for expert review, "
        "not manually labelled ground truth. The five-fold voting ensemble "
        "has no independent external accuracy estimate. TIFF masks are 1200x1200 "
        "uint8 with background=0 and foreground=255. The archive contains no "
        "source fluorescence images, checkpoints, or absolute computer paths.\n"
    ).encode()

    if archive_path.exists() or digest_path.exists():
        if not archive_path.is_file() or not digest_path.is_file():
            raise ValueError(f"Incomplete existing delivery: {cell}")
        line = digest_path.read_text(encoding="ascii").strip().split()
        if len(line) != 2 or line[1] != archive_path.name or sha256_file(archive_path) != line[0]:
            raise ValueError(f"Existing delivery checksum differs: {cell}")
        verify_archive(archive_path, records)
        with zipfile.ZipFile(archive_path) as archive:
            if (archive.read("fluo_label/PORTABLE_MANIFEST.json") != manifest_bytes
                    or archive.read("fluo_label/AUTO_LABELS_README.txt") != note
                    or archive.read("fluo_label/SHA256SUMS.txt") != sha_lines):
                raise ValueError(f"Existing portable metadata differs: {cell}")
    else:
        partial = archive_path.with_suffix(".zip.partial")
        if partial.exists():
            raise FileExistsError(f"Retained incomplete archive needs inspection: {partial}")
        try:
            with zipfile.ZipFile(partial, "w", allowZip64=True) as archive:
                for record in records:
                    name = record["filename"]
                    zip_entry(f"fluo_label/{name}",
                              (data_root / cell / "fluo_label" / name).read_bytes(), archive)
                zip_entry("fluo_label/PORTABLE_MANIFEST.json", manifest_bytes, archive)
                zip_entry("fluo_label/AUTO_LABELS_README.txt", note, archive)
                zip_entry("fluo_label/SHA256SUMS.txt", sha_lines, archive)
            verify_archive(partial, records)
            if partial.stat().st_size >= MAX_GIT_BLOB_BYTES:
                raise ValueError(f"ZIP exceeds GitHub single-file limit: {cell}")
            os.rename(partial, archive_path)
            with digest_path.open("x", encoding="ascii") as digest_out:
                digest_out.write(f"{sha256_file(archive_path)}  {archive_path.name}\n")
        except Exception:
            # Keep a partial ZIP for diagnosis; never replace the source masks.
            raise
    return {
        "cell": cell,
        "masks": len(records),
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "Desktop/Organoid/organoid_data")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "deliverables/ld_auto_labels")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.is_relative_to(data_root):
        raise ValueError("Archives must not be written into the source data root")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    results = [package_cell(cell, data_root, output_root)
               for cell in CELL_COUNTS]
    if sum(result["masks"] for result in results) != 1330:
        raise ValueError("Expected exactly 1,330 LD masks")
    print(json.dumps({"status": "PASS", "cells": len(results),
                      "masks": 1330, "archives": results}, indent=2))


if __name__ == "__main__":
    main()
