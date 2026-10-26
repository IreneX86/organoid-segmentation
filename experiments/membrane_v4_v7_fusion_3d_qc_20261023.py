"""Read-only per-run 3D continuity and fluo-dark-core audit of the safe fusion.

Dark fluo regions are *nuclear candidates*, never authoritative nuclei.
This script does not change predictions or choose fusion parameters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure, label

from membrane_v4_base_v7_additions_20261023 import CELL, OUT, V4, V7, sha256


SAFE = OUT / "memb_instances_v4base_v7add_safe.tif"
STRUCTURE_3D = generate_binary_structure(3, 1)


def persistent_components(mask: np.ndarray, min_volume: int = 300, min_layers: int = 2) -> list[dict]:
    labels, count = label(mask, structure=STRUCTURE_3D)
    results = []
    for ident in range(1, count + 1):
        points = np.where(labels == ident)
        volume = len(points[0])
        layers = np.unique(points[0])
        if volume >= min_volume and len(layers) >= min_layers:
            results.append({"voxels": volume, "layers": len(layers),
                            "z_first_local": int(layers.min()), "z_last_local": int(layers.max()),
                            "centroid_y": float(np.mean(points[1])),
                            "centroid_x": float(np.mean(points[2]))})
    return sorted(results, key=lambda row: -row["voxels"])


def run_qc(run: dict, v4_t: tifffile.TiffFile, v7_t: tifffile.TiffFile,
           fused_t: tifffile.TiffFile, fluo_t: tifffile.TiffFile) -> dict:
    layers = run["layers"]
    vid = run["v7_id"]
    bounds = [1200, 0, 1200, 0]
    for z in layers:
        yy, xx = np.where(v7_t.pages[z].asarray() == vid)
        bounds[0] = min(bounds[0], int(yy.min()))
        bounds[1] = max(bounds[1], int(yy.max()) + 1)
        bounds[2] = min(bounds[2], int(xx.min()))
        bounds[3] = max(bounds[3], int(xx.max()) + 1)
    y0, y1, x0, x1 = bounds
    crop = np.s_[max(y0 - 6, 0):min(y1 + 6, 1200), max(x0 - 6, 0):min(x1 + 6, 1200)]
    added_planes = []
    dark_planes = []
    v4_identity_unchanged = True
    for z in layers:
        v4 = v4_t.pages[z].asarray()[crop]
        source = v7_t.pages[z].asarray()[crop]
        fused = fused_t.pages[z].asarray()[crop]
        if not np.array_equal(fused[v4 > 0], v4[v4 > 0]):
            v4_identity_unchanged = False
        proposed = (source == vid) & (v4 == 0)
        if run["global_id"] is None:
            if np.any(fused[proposed] != 0):
                raise AssertionError(f"A deferred run was added at z{z}")
        elif not np.all(fused[proposed] == run["global_id"]):
            raise AssertionError(f"Wrong 3D identity at z{z}")
        added_planes.append(proposed)
        nucleus_core = binary_erosion(proposed, iterations=4)
        rim = proposed & ~nucleus_core
        fluo = gaussian_filter(fluo_t.pages[z].asarray()[crop].astype(np.float32), 3)
        if nucleus_core.sum() >= 100 and rim.sum() >= 30:
            dark_planes.append(nucleus_core & (fluo < .85 * np.median(fluo[rim])))
        else:
            dark_planes.append(np.zeros_like(proposed))
    added = np.stack(added_planes)
    dark = np.stack(dark_planes)
    candidate_components = persistent_components(added, min_volume=300, min_layers=2)
    dark_components = persistent_components(dark)
    if run["global_id"] is None:
        status = "deferred_v4_anchor_conflict"
    elif not v4_identity_unchanged:
        status = "FAIL_v4_identity_changed"
    elif len(candidate_components) != 1:
        status = "review_fragmented_3d_candidate"
    elif not dark_components:
        status = "review_no_persistent_dark_core"
    elif len(dark_components) > 1:
        status = "review_multiple_dark_core_candidates"
    else:
        status = "one_persistent_dark_core_candidate_not_proven_nucleus"
    return {"run_index": run["run_index"], "v7_id": vid,
            "z_first": run["z_first"], "z_last": run["z_last"],
            "layers": len(layers), "global_id": run["global_id"],
            "decision": run["decision"], "boundary_anchors": run["boundary_anchors"],
            "added_voxels": int(added.sum()),
            "added_3d_components_persistent": len(candidate_components),
            "dark_core_3d_candidates_persistent": len(dark_components),
            "dark_core_components": dark_components[:5],
            "v4_identity_unchanged": v4_identity_unchanged,
            "qc_status": status}


def main() -> None:
    output = OUT / "qc_19_runs.json"
    if output.exists():
        raise FileExistsError(output)
    safe = json.loads((OUT / "safe_config.json").read_text())
    if sha256(SAFE) != safe["output_sha256"]:
        raise RuntimeError("Safe TIFF hash mismatch")
    runs = json.loads((OUT / "safe_runs.json").read_text())
    rows = []
    with tifffile.TiffFile(V4) as v4, tifffile.TiffFile(V7) as v7, \
            tifffile.TiffFile(SAFE) as fused, tifffile.TiffFile(CELL / "fluo.tif") as fluo:
        for run in runs:
            rows.append(run_qc(run, v4, v7, fused, fluo))
    if sum(row["global_id"] is not None for row in rows) != 19:
        raise RuntimeError("Expected 19 inserted runs")
    if not all(row["v4_identity_unchanged"] for row in rows):
        raise RuntimeError("A v4 3D ID changed")
    with (OUT / "qc_19_runs.csv").open("w", newline="") as stream:
        columns = [key for key in rows[0] if key not in ("dark_core_components", "boundary_anchors")]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    summary = {"scope": "19 accepted runs plus one deferred conflict; dark areas are candidates, not manual nucleus labels",
               "safe_output_sha256": safe["output_sha256"],
               "status_counts": {status: sum(row["qc_status"] == status for row in rows)
                                 for status in sorted({row["qc_status"] for row in rows})},
               "runs": rows}
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["status_counts"], indent=2))


if __name__ == "__main__":
    main()
