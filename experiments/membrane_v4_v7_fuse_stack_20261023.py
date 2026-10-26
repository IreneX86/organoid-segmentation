"""Apply the unchanged conservative v4-base/v7-addition rule to another stack.

Imports the already-tested candidate screen and fail-closed 3D ID assignment;
only their data paths and Z length change. Human labels never enter selection.
Historical v4/v7 predictions and the 330-Z fusion remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

import membrane_v4_base_v7_additions_20261023 as screen
import membrane_v4_base_v7_constrained_stitch_20261023 as assign
from membrane_v4_v7_fusion_3d_qc_20261023 import run_qc


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cell", "v4", "v7", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--z-count", required=True, type=int)
    args = parser.parse_args()
    if args.z_count < 3 or args.z_count > 1000:
        raise ValueError("Unexpected Z count")
    for path in (args.cell / "Memb.tif", args.cell / "fluo.tif", args.v4, args.v7):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(args.out)
    with tifffile.TiffFile(args.v4) as a, tifffile.TiffFile(args.v7) as b:
        if len(a.pages) != args.z_count or len(b.pages) != args.z_count:
            raise RuntimeError("Prediction Z count mismatch")
        if a.pages[0].shape != (1200, 1200) or b.pages[0].shape != (1200, 1200):
            raise RuntimeError("Prediction XY shape mismatch")
    # All scientific constants (area, overlap, signal and anchoring gates) are
    # imported unchanged from the 330-Z pilot. Patch only stack locations.
    screen.CELL, screen.V4, screen.V7 = args.cell, args.v4, args.v7
    screen.N_PLANES, screen.OUT = args.z_count, args.out
    assign.CELL, assign.V4, assign.V7 = args.cell, args.v4, args.v7
    assign.N_PLANES, assign.OUT = args.z_count, args.out
    assign.OUTPUT = args.out / "memb_instances_v4base_v7add_safe.tif"
    args.out.mkdir(parents=True)
    events, _ = screen.scan_candidates()
    runs = assign.determine_assignments(assign.accepted_runs(events))
    result = assign.write_output(runs)
    (args.out / "candidate_events.json").write_text(json.dumps(events, indent=2) + "\n")
    (args.out / "safe_runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    summary = {"status": "out_of_development_visual_candidate",
               "cell": str(args.cell), "z_count": args.z_count,
               "selection_uses_human_labels": False,
               "method": "frozen v4-base/v7-addition screen and fail-closed v4 anchored IDs",
               "criteria": {"min_area": screen.MIN_AREA,
                            "max_v4_overlap_fraction": screen.MAX_V4_OVERLAP_FRACTION,
                            "min_consecutive_planes": screen.MIN_CONSECUTIVE_PLANES,
                            "min_adjacent_iou": screen.MIN_ADJACENT_IOU,
                            "min_membrane_ratio": screen.MIN_MEMBRANE_RATIO,
                            "min_dark_core_px": screen.MIN_DARK_CORE_PX,
                            "min_dark_support_planes": screen.MIN_DARK_SUPPORT_PLANES,
                            "min_anchor_iou": assign.MIN_ANCHOR_IOU},
               "v4_sha256": screen.sha256(args.v4), "v7_sha256": screen.sha256(args.v7),
               "screened_slice_events": len(events),
               "accepted_slice_events": sum(len(r["layers"]) for r in runs if r["global_id"] is not None),
               "accepted_runs": sum(r["global_id"] is not None for r in runs),
               "deferred_conflicting_runs": sum(r["global_id"] is None for r in runs),
               **result}
    # Verify source v4 labels are bitwise retained in the fused TIFF.
    added = 0
    with tifffile.TiffFile(args.v4) as a, tifffile.TiffFile(assign.OUTPUT) as fused:
        for z in range(args.z_count):
            base = a.pages[z].asarray()
            output = fused.pages[z].asarray()
            if not np.array_equal(output[base > 0], base[base > 0]):
                raise AssertionError(f"v4 label changed at z{z}")
            added += int(np.count_nonzero((output > 0) & (base == 0)))
    summary["added_voxels"] = added
    (args.out / "safe_config.json").write_text(json.dumps(summary, indent=2) + "\n")
    qc_rows = []
    with tifffile.TiffFile(args.v4) as a, tifffile.TiffFile(args.v7) as b, \
            tifffile.TiffFile(assign.OUTPUT) as fused, \
            tifffile.TiffFile(args.cell / "fluo.tif") as fluo:
        for run in runs:
            qc_rows.append(run_qc(run, a, b, fused, fluo))
    if any(row["qc_status"] in ("FAIL_v4_identity_changed", "review_fragmented_3d_candidate")
           for row in qc_rows if row["global_id"] is not None):
        raise RuntimeError("An accepted 3D addition failed continuity/identity QC")
    (args.out / "qc_runs.json").write_text(json.dumps(qc_rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
