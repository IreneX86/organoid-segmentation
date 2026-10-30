"""Read-only verification of a completed ER coherent-fiber batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from er_bright_line_pipeline import sha256
from er_ridge_batch_predict import pairs


def verify(input_root: Path, output_root: Path) -> dict:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    tasks = pairs(input_root)
    expected = {relative: (er, fluo) for er, fluo, relative in tasks}
    batch = json.loads((output_root / "batch_manifest.json").read_text())
    if batch["cell_count"] != len(expected) or len(batch["completed"]) != len(expected):
        raise ValueError("Batch manifest does not cover all input cells")
    expected_outputs = {str(output_root / rel / "ER_mask_line.tif") for rel in expected}
    if set(batch["completed"]) != expected_outputs:
        raise ValueError("Batch output paths do not match input cells")
    records = []
    for relative, (er, fluo) in expected.items():
        folder = output_root / relative
        manifest = json.loads((folder / "manifest.json").read_text())
        if Path(manifest["source_er"]).resolve() != er.resolve() or \
                Path(manifest["source_fluo"]).resolve() != fluo.resolve():
            raise ValueError(f"Source mismatch: {relative}")
        with tifffile.TiffFile(er) as source:
            source_shape = source.series[0].shape
        start = manifest["source_z_start_zero_based"]
        stop = manifest["source_z_stop_exclusive"]
        if manifest["source_shape"] != list(source_shape) or not 0 <= start < stop <= source_shape[0]:
            raise ValueError(f"Source shape/Z mismatch: {relative}")
        output_shape = source_shape[1:] if stop - start == 1 else (stop - start, *source_shape[1:])
        if [row["source_z_zero_based"] for row in manifest["slice_audits"]] != list(range(start, stop)):
            raise ValueError(f"Incomplete Z audit: {relative}")
        if sha256(er) != manifest["source_er_sha256"] or sha256(fluo) != manifest["source_fluo_sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {relative}")
        base = Path(manifest["base_mask"])
        if sha256(base) != manifest["base_mask_sha256"]:
            raise ValueError(f"Base mask SHA256 mismatch: {relative}")
        for name in ("ER_mask_bright", "ER_mask_line_legacy", "ER_mask_line"):
            record = manifest["outputs"][name]
            path = folder / f"{name}.tif"
            if Path(record["path"]).resolve() != path.resolve() or sha256(path) != record["sha256"]:
                raise ValueError(f"Output SHA256/path mismatch: {path}")
            with tifffile.TiffFile(path) as mask_file:
                if mask_file.series[0].shape != output_shape or record["shape"] != list(output_shape):
                    raise ValueError(f"Mask shape mismatch: {path}")
                if mask_file.series[0].dtype != np.dtype("uint8"):
                    raise ValueError(f"Mask dtype is not uint8: {path}")
                for page in mask_file.pages:
                    if not np.isin(page.asarray(), (0, 255)).all():
                        raise ValueError(f"Mask is not binary 0/255: {path}")
        records.append({"cell": er.parent.name, "z_slices": stop - start,
                        "shape": list(output_shape), "result": "PASS"})
    return {"result": "PASS", "cell_count": len(records),
            "z_slices": sum(row["z_slices"] for row in records), "cells": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input_root, args.output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
