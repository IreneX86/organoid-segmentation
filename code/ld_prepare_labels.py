"""Create traceable binary LD labels from antialiased manual TIFF masks.

This is a one-time label-definition step, performed before model evaluation.
It never edits the source TIFFs and never chooses a threshold from model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from tiff_io import pair_tiff_directories, read_tiff_2d


RULE_ID = "ld_antialias_midpoint_v1"


def sha256_file(path: Path) -> str:
    """Hash source and derived files without reading them all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binarize_antialiased_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep pixels with at least 50% of that file's observed white level.

    Photoshop masks approach 65535, whereas three ImageJ masks use 0/255
    despite being stored as uint16. A fixed 16-bit cutoff would erase the
    latter. The per-file midpoint is a declared coverage convention, not a
    threshold optimized against model predictions or held-out Dice.
    """

    if mask.ndim != 2 or mask.dtype != np.uint16:
        raise ValueError(f"Expected a 2D uint16 LD mask, got {mask.shape} {mask.dtype}")
    if int(mask.min()) != 0:
        raise ValueError("LD mask background must contain value 0")
    maximum = int(mask.max())
    if maximum != 255 and not 60000 <= maximum <= 65535:
        raise ValueError(f"Unexpected LD mask white level {maximum}; review its encoding")
    # ceil(max / 2) includes pixels covering at least half the annotated area.
    threshold = (maximum + 1) // 2
    return (mask >= threshold).astype(np.uint8), threshold


def prepare_labels(
    input_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    expected_pairs: int = 36,
) -> dict[str, object]:
    """Write new binary TIFFs and a source-to-output provenance manifest."""

    input_dir = input_dir.resolve()
    mask_dir = mask_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing derived labels: {output_dir}")
    if output_dir.is_relative_to(input_dir) or output_dir.is_relative_to(mask_dir):
        raise ValueError("Derived labels must be outside the source TIFF directories")

    pairs = pair_tiff_directories(input_dir, mask_dir)
    if len(pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} LD pairs, found {len(pairs)}")

    # Check every source before creating the output directory. This avoids
    # producing a partial derived dataset for a malformed or mismatched pair.
    records: list[tuple[Path, np.ndarray, dict[str, object]]] = []
    for pair in pairs:
        image = read_tiff_2d(pair.image_path)
        source_mask = read_tiff_2d(pair.label_path)
        if image.shape != source_mask.shape or image.dtype != np.uint16:
            raise ValueError(f"Input/mask geometry or input dtype mismatch: {pair.filename}")
        binary, threshold = binarize_antialiased_mask(source_mask)
        maximum = int(source_mask.max())
        item: dict[str, object] = {
            "filename": pair.filename,
            "shape": list(image.shape),
            "source_image_sha256": sha256_file(pair.image_path),
            "source_mask_sha256": sha256_file(pair.label_path),
            "source_mask_max": maximum,
            "source_mask_unique_values": int(np.unique(source_mask).size),
            "threshold_inclusive": threshold,
            "foreground_pixels_nonzero": int(np.count_nonzero(source_mask)),
            "foreground_pixels_25pct": int(np.count_nonzero(source_mask >= (maximum + 3) // 4)),
            "foreground_pixels_50pct": int(binary.sum()),
            "foreground_pixels_75pct": int(np.count_nonzero(source_mask >= (3 * maximum + 3) // 4)),
        }
        records.append((pair.label_path, binary, item))

    output_dir.mkdir(parents=True, exist_ok=False)
    for _, binary, item in records:
        output_path = output_dir / str(item["filename"])
        tifffile.imwrite(output_path, binary, photometric="minisblack", metadata=None)
        saved = read_tiff_2d(output_path)
        if saved.dtype != np.uint8 or not np.array_equal(saved, binary):
            raise RuntimeError(f"Derived TIFF failed read-back verification: {output_path}")
        item["derived_mask_sha256"] = sha256_file(output_path)

    manifest: dict[str, object] = {
        "rule_id": RULE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_input_dir": str(input_dir),
        "source_mask_dir": str(mask_dir),
        "derived_mask_dir": str(output_dir),
        "label_semantics": {"background": 0, "foreground": 1},
        "pixel_rule": "mask_value >= ceil(source_mask_max / 2)",
        "files": [item for _, _, item in records],
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-pairs", type=int, default=36)
    args = parser.parse_args()
    manifest = prepare_labels(
        args.input_dir, args.mask_dir, args.output_dir, args.expected_pairs
    )
    print(f"Prepared {len(manifest['files'])} LD labels using {RULE_ID}")


if __name__ == "__main__":
    main()
