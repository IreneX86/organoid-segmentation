"""Globally assign 3D IDs to accepted v7 runs without merging v4 tracks.

The first unconstrained IoU restitch of this pilot linked two *different* v4
tracks through an added v7 run. This fail-closed pass treats existing v4 3D
IDs as immutable anchors. A contiguous v7 run may join only one supported v4
anchor (adjacent-slice IoU >=0.25), and only when that v4 cell is absent in
every slice of the run. Conflicting/no anchors get a fresh global 3D ID.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import tifffile

from membrane_v4_base_v7_additions_20261023 import (
    CELL, MAX_SOURCE, N_PLANES, OUT, SHAPE, V4, V7, sha256,
)


OUTPUT = OUT / "memb_instances_v4base_v7add_safe.tif"
MIN_ANCHOR_IOU = .25


def accepted_runs(events: list[dict]) -> list[dict]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for row in events:
        if row["decision"] == "add":
            by_id[row["v7_id"]].append(row["z"])
    runs = []
    for v7_id, layers in sorted(by_id.items()):
        current: list[int] = []
        for z in layers:
            if current and z != current[-1] + 1:
                runs.append({"v7_id": v7_id, "layers": current})
                current = []
            current.append(z)
        if current:
            runs.append({"v7_id": v7_id, "layers": current})
    return sorted(runs, key=lambda row: (row["layers"][0], row["v7_id"]))


def boundary_anchors(v7_mask: np.ndarray, neighboring_v4: np.ndarray) -> list[dict]:
    ids, counts = np.unique(neighboring_v4[v7_mask], return_counts=True)
    results = []
    source_area = int(v7_mask.sum())
    for ident, intersection in zip(ids, counts):
        if ident == 0:
            continue
        target_area = int(np.count_nonzero(neighboring_v4 == ident))
        iou = int(intersection) / max(source_area + target_area - int(intersection), 1)
        if iou >= MIN_ANCHOR_IOU:
            results.append({"v4_id": int(ident), "iou": iou})
    return results


def determine_assignments(runs: list[dict]) -> list[dict]:
    with tifffile.TiffFile(V4) as a, tifffile.TiffFile(V7) as b:
        if len(a.pages) != N_PLANES or len(b.pages) != N_PLANES:
            raise RuntimeError("Unaligned predictions")
        for index, run in enumerate(runs, 1):
            layers, v7_id = run["layers"], run["v7_id"]
            anchors = []
            for edge, adjacent in ((layers[0], layers[0] - 1), (layers[-1], layers[-1] + 1)):
                if 0 <= adjacent < N_PLANES:
                    target = b.pages[edge].asarray() == v7_id
                    for match in boundary_anchors(target, a.pages[adjacent].asarray()):
                        anchors.append({"neighbor_z": adjacent, **match})
            candidate_ids = {item["v4_id"] for item in anchors}
            if len(candidate_ids) == 1:
                anchor = candidate_ids.pop()
                present_within_run = any(np.any(a.pages[z].asarray() == anchor) for z in layers)
                if present_within_run:
                    decision = "new_id_anchor_present_inside_run"
                    global_id = MAX_SOURCE + index
                else:
                    decision = "join_unique_v4_anchor"
                    global_id = anchor
            elif anchors:
                # A v7 run bridging two different v4 3D tracks could silently
                # reintroduce the very merge the user wants removed. Keep the
                # v4-only pixels here and put the run on the review list.
                decision = "review_conflicting_v4_anchors_not_added"
                global_id = None
            else:
                decision = "new_id_no_reliable_v4_anchor"
                global_id = MAX_SOURCE + index
            run.update({"run_index": index, "z_first": layers[0], "z_last": layers[-1],
                        "boundary_anchors": anchors, "decision": decision,
                        "global_id": global_id})
    return runs


def write_output(runs: list[dict]) -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    assignments = {(z, run["v7_id"]): run["global_id"] for run in runs
                   if run["global_id"] is not None for z in run["layers"]}
    by_z: dict[int, list[int]] = defaultdict(list)
    for z, v7_id in assignments:
        by_z[z].append(v7_id)
    working = OUTPUT.with_suffix(".working.tif")
    if working.exists():
        raise FileExistsError(working)
    base_tracks: set[int] = set()
    all_tracks: set[int] = set()
    try:
        with tifffile.TiffFile(V4) as a, tifffile.TiffFile(V7) as b, \
                tifffile.TiffWriter(working, imagej=True) as writer:
            for z in range(N_PLANES):
                base = a.pages[z].asarray()
                source = b.pages[z].asarray()
                output = base.astype(np.uint16, copy=True)
                base_tracks.update(map(int, np.unique(base[base > 0])))
                for v7_id in by_z[z]:
                    added = (source == v7_id) & (base == 0)
                    output[added] = assignments[(z, v7_id)]
                all_tracks.update(map(int, np.unique(output[output > 0])))
                writer.write(output, contiguous=True, photometric="minisblack")
        stacked = tifffile.memmap(working)
        if stacked.shape != (N_PLANES, *SHAPE):
            raise RuntimeError(f"Unexpected stack shape {stacked.shape}")
        tifffile.imwrite(OUTPUT, stacked, imagej=True, photometric="minisblack",
                         compression="zlib", compressionargs={"level": 1},
                         metadata={"axes": "ZYX",
                                   "Info": "v4 immutable 3D IDs; isolated v7 runs linked by unique boundary anchor"})
        del stacked
        working.unlink()
    except Exception:
        working.unlink(missing_ok=True)
        OUTPUT.unlink(missing_ok=True)
        raise
    return {"output_sha256": sha256(OUTPUT), "v4_3d_ids": len(base_tracks),
            "fused_3d_ids": len(all_tracks), "new_3d_ids": len(all_tracks - base_tracks),
            "new_id_values": sorted(all_tracks - base_tracks)}


def main() -> None:
    original = json.loads((OUT / "run_config.json").read_text())
    if sha256(V4) != original["v4_sha256"] or sha256(V7) != original["v7_sha256"]:
        raise RuntimeError("Source prediction hash mismatch")
    events = json.loads((OUT / "candidate_events.json").read_text())
    runs = determine_assignments(accepted_runs(events))
    result = write_output(runs)
    (OUT / "safe_runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    config = {"status": "exploratory_candidate_not_production", "global_stitch": "v4 immutable anchors; v7 contiguous runs; unique >=.25 boundary IoU or new ID",
              "v4_sha256": original["v4_sha256"], "v7_sha256": original["v7_sha256"],
              "selection_uses_human_labels": False,
              "run_decisions": {name: sum(run["decision"] == name for run in runs)
                                for name in sorted({run["decision"] for run in runs})},
              "screened_runs": len(runs),
              "accepted_runs": sum(run["global_id"] is not None for run in runs),
              "accepted_slice_events": sum(len(run["layers"]) for run in runs if run["global_id"] is not None),
              **result}
    (OUT / "safe_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
