"""Check every ER batch mask against its source, manifest and encoded values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, default=8)
    args = parser.parse_args()
    root = args.results_root.resolve()
    manifests = sorted(root.glob("cell_*/ER_mask_manifest.json"))
    if len(manifests) != args.expected_cells:
        raise RuntimeError(f"Expected {args.expected_cells} complete cells, found {len(manifests)}")
    rows = []
    for manifest_path in manifests:
        info = json.loads(manifest_path.read_text())
        mask_path = manifest_path.with_name("ER_mask.tif")
        er_path = Path(info["source_er"])
        fluo_path = Path(info["source_fluo"])
        if sha256(mask_path) != info["output_sha256"]:
            raise RuntimeError(f"Mask SHA256 mismatch: {mask_path}")
        with (
            tifffile.TiffFile(er_path) as er,
            tifffile.TiffFile(fluo_path) as fluo,
            tifffile.TiffFile(mask_path) as mask,
        ):
            shape = tuple(info["output_shape"])
            if (
                mask.series[0].shape != shape
                or er.series[0].shape != shape
                or fluo.series[0].shape != shape
            ):
                raise RuntimeError(f"Source/output shape mismatch: {manifest_path.parent.name}")
            if len(mask.pages) != shape[0] or mask.pages[0].dtype != np.uint8:
                raise RuntimeError(f"Wrong page count/dtype: {mask_path}")
            counts = []
            for z, page in enumerate(mask.pages):
                array = page.asarray()
                if not np.all((array == 0) | (array == 255)):
                    raise RuntimeError(f"Nonbinary mask: {mask_path}, Z={z + 1}")
                count = int(np.count_nonzero(array))
                if count != info["slices"][z]["final_pixels"]:
                    raise RuntimeError(f"Per-slice audit mismatch: {mask_path}, Z={z + 1}")
                counts.append(count)
        rows.append({
            "cell": manifest_path.parent.name,
            "source_er": str(er_path),
            "mask": str(mask_path),
            "sha256": info["output_sha256"],
            "shape": list(shape),
            "foreground_fraction_mean": float(np.mean(counts) / (shape[1] * shape[2])),
            "foreground_pixels_min": min(counts),
            "foreground_pixels_max": max(counts),
            "artifact_temporal_fallbacks": info.get("artifact_temporal_fallbacks", []),
        })
        print(f"{manifest_path.parent.name}: PASS", flush=True)
    summary = {
        "status": "PASS",
        "cells": len(rows),
        "slices": sum(row["shape"][0] for row in rows),
        "results": rows,
    }
    (root / "batch_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Batch verified: {summary['cells']} cells, {summary['slices']} slices")


if __name__ == "__main__":
    main()
