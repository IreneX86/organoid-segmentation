"""Conservative v4-base/v7-addition pilot for the 330-plane Membrane stack.

Candidate selection uses only the two predictions and raw Memb/fluo signals.
Human labels are deliberately absent from this module. Existing v4 pixels are
never changed; v7 objects that overlap v4 substantially are never substituted.
After selection, all per-plane objects receive fresh, one-to-one global Z IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "runpod_results/membrane_v4_local11_20261023/results/memb_instances_v4_local11.tif"
V7 = ROOT / "runpod_results/membrane_v7_fullstack_20261022/v7/memb_instances_v7_local11.tif"
CELL = Path.home() / "Desktop/Organoid/Membrane/MCF-7/cell_21941_NZ_330"
OUT = ROOT / "runpod_results/membrane_v4_base_v7_additions_20261023"

EXPECTED_V4 = "ab16f5d603801aa3e575f8c2af1ce479382dc8578ad3223a95795f21a4c44d9e"
EXPECTED_V7 = "ee7edf835e6759acc03ea360e6a9120df8dbf69feaeb501545653f730c45eb0e"
N_PLANES = 330
SHAPE = (1200, 1200)
SOURCE_OFFSET = 512
MAX_SOURCE = 1024

# Fixed before consulting any corrected human labels. In particular, a v7
# object with meaningful overlap with a v4 cell is deferred, not split.
MIN_AREA = 1000
MAX_V4_OVERLAP_FRACTION = 0.02
MIN_CONSECUTIVE_PLANES = 3
MIN_ADJACENT_IOU = 0.25
MIN_MEMBRANE_RATIO = 1.05
MIN_DARK_CORE_PX = 100
MIN_DARK_SUPPORT_PLANES = 2
STITCH_IOU = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_signal(memb: np.ndarray, fluo: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Measure a membrane-bright rim and dark fluo interior within one v7 cell."""
    yy, xx = np.where(mask)
    crop = np.s_[max(0, int(yy.min()) - 5):min(SHAPE[0], int(yy.max()) + 6),
                 max(0, int(xx.min()) - 5):min(SHAPE[1], int(xx.max()) + 6)]
    inside = mask[crop]
    core = binary_erosion(inside, iterations=4)
    rim = inside & ~core
    if core.sum() < 100 or rim.sum() < 30:
        return 0.0, 0
    membrane_ratio = float(np.median(memb[crop][rim]) / max(float(np.median(memb[crop][core])), 1.0))
    smooth_fluo = gaussian_filter(fluo[crop].astype(np.float32), 3)
    dark = core & (smooth_fluo < .85 * np.median(smooth_fluo[rim]))
    connected, count = label(dark)
    sizes = np.bincount(connected[dark], minlength=count + 1)
    largest_dark = int(sizes[1:].max()) if count else 0
    return membrane_ratio, largest_dark


def overlap_stats(v4: np.ndarray, v7: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if v4.shape != SHAPE or v7.shape != SHAPE:
        raise ValueError("Both predictions must be 1200x1200")
    if v4.max() >= SOURCE_OFFSET or v7.max() >= SOURCE_OFFSET:
        raise ValueError("Unexpected source label >=512")
    joint = np.bincount(
        v7.astype(np.int32).ravel() * SOURCE_OFFSET + v4.astype(np.int32).ravel(),
        minlength=SOURCE_OFFSET ** 2,
    ).reshape(SOURCE_OFFSET, SOURCE_OFFSET)
    return joint.sum(axis=1), joint[:, 1:].sum(axis=1), joint[:, 0]


def scan_candidates() -> tuple[list[dict], dict[int, set[int]]]:
    events: list[dict] = []
    event_by_track: dict[int, list[dict]] = defaultdict(list)
    previous_v7: np.ndarray | None = None
    previous_area: np.ndarray | None = None
    with tifffile.TiffFile(V4) as a, tifffile.TiffFile(V7) as b, \
            tifffile.TiffFile(CELL / "Memb.tif") as memb, \
            tifffile.TiffFile(CELL / "fluo.tif") as fluo:
        if len(a.pages) != N_PLANES or len(b.pages) != N_PLANES:
            raise ValueError("Unexpected prediction length")
        if not all(len(t.pages) == N_PLANES for t in (a, b, memb, fluo)):
            raise ValueError("Unaligned input stack")
        for z in range(N_PLANES):
            v4 = a.pages[z].asarray()
            v7 = b.pages[z].asarray()
            area, overlap, uncovered = overlap_stats(v4, v7)
            possible = [int(ident) for ident in np.flatnonzero(area >= MIN_AREA)
                        if ident and overlap[ident] / area[ident] <= MAX_V4_OVERLAP_FRACTION]
            if possible:
                membrane = memb.pages[z].asarray()
                fluorescence = fluo.pages[z].asarray()
                for ident in possible:
                    ratio, dark = candidate_signal(membrane, fluorescence, v7 == ident)
                    if previous_v7 is not None:
                        intersection = int(np.count_nonzero((previous_v7 == ident) & (v7 == ident)))
                        previous = int(previous_area[ident])
                        adjacent_iou = intersection / max(previous + int(area[ident]) - intersection, 1)
                    else:
                        adjacent_iou = 0.0
                    row = {"z": z, "v7_id": ident, "v7_area": int(area[ident]),
                           "v4_overlap_px": int(overlap[ident]),
                           "v4_overlap_fraction": float(overlap[ident] / area[ident]),
                           "uncovered_px": int(uncovered[ident]),
                           "memb_rim_to_core": ratio, "largest_dark_fluo_px": dark,
                           "previous_z_same_id_iou": adjacent_iou,
                           "decision": "review_pending"}
                    events.append(row)
                    event_by_track[ident].append(row)
            previous_v7, previous_area = v7, area
    accepted: dict[int, set[int]] = defaultdict(set)
    # A run is spatially continuous, not merely the same stitched v7 ID.
    for ident, rows in event_by_track.items():
        runs: list[list[dict]] = []
        current: list[dict] = []
        for row in rows:
            if current and (row["z"] != current[-1]["z"] + 1
                            or row["previous_z_same_id_iou"] < MIN_ADJACENT_IOU):
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        for run in runs:
            median_membrane = float(np.median([r["memb_rim_to_core"] for r in run]))
            dark_support = sum(r["largest_dark_fluo_px"] >= MIN_DARK_CORE_PX for r in run)
            if (len(run) >= MIN_CONSECUTIVE_PLANES
                    and median_membrane >= MIN_MEMBRANE_RATIO
                    and dark_support >= MIN_DARK_SUPPORT_PLANES):
                for row in run:
                    row["decision"] = "add"
                    accepted[row["z"]].add(ident)
            else:
                for row in run:
                    row["decision"] = "review_temporal_or_signal_gate"
    return events, accepted


def fused_plane(v4: np.ndarray, v7: np.ndarray, accepted_ids: set[int]) -> np.ndarray:
    """Retain every v4 pixel/label, placing only isolated novel v7 pixels."""
    result = v4.astype(np.uint16, copy=True)
    for ident in accepted_ids:
        source = (v7 == ident) & (v4 == 0)
        result[source] = SOURCE_OFFSET + ident
    if not np.array_equal(result[v4 > 0], v4[v4 > 0]):
        raise AssertionError("The v4 base was altered")
    return result


def stitch_one(prev_local: np.ndarray | None, prev_global: np.ndarray | None,
               current_local: np.ndarray, next_id: int) -> tuple[np.ndarray, int]:
    """One-to-one whole-stack IoU stitching; local source IDs only break ties."""
    ids = np.unique(current_local)
    ids = ids[ids > 0]
    mapping: dict[int, int] = {}
    if prev_local is not None and len(ids):
        previous_ids = np.unique(prev_local)
        previous_ids = previous_ids[previous_ids > 0]
        if len(previous_ids):
            packed = np.bincount(
                prev_local.astype(np.int32).ravel() * MAX_SOURCE + current_local.astype(np.int32).ravel(),
                minlength=MAX_SOURCE ** 2,
            ).reshape(MAX_SOURCE, MAX_SOURCE)
            intersection = packed[np.ix_(previous_ids, ids)]
            pa = packed.sum(axis=1)[previous_ids]
            ca = packed.sum(axis=0)[ids]
            iou = intersection / np.maximum(pa[:, None] + ca[None, :] - intersection, 1)
            # Prefer a proven same-source track at an exact IoU tie, while
            # retaining the same STITCH_IOU gate for every model and source.
            priority = iou + (previous_ids[:, None] == ids[None, :]) * 1e-6
            previous_idx, current_idx = linear_sum_assignment(-priority)
            for p, c in zip(previous_idx, current_idx):
                if iou[p, c] >= STITCH_IOU:
                    prior_local = int(previous_ids[p])
                    prior_global_ids = np.unique(prev_global[prev_local == prior_local])
                    if len(prior_global_ids) != 1 or prior_global_ids[0] == 0:
                        raise AssertionError("Inconsistent previous global ID")
                    mapping[int(ids[c])] = int(prior_global_ids[0])
    for ident in ids:
        if int(ident) not in mapping:
            next_id += 1
            mapping[int(ident)] = next_id
    if next_id > np.iinfo(np.uint16).max:
        raise RuntimeError("Unexpected >65535 three-dimensional tracks")
    lookup = np.zeros(MAX_SOURCE, dtype=np.uint16)
    for local, global_id in mapping.items():
        lookup[local] = global_id
    return lookup[current_local], next_id


def write_stitched(accepted: dict[int, set[int]], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    next_id = 0
    prev_local = prev_global = None
    added_slice_objects = 0
    added_pixels = 0
    tmp = out.with_suffix(".partial.tif")
    try:
        with tifffile.TiffFile(V4) as a, tifffile.TiffFile(V7) as b, \
                tifffile.TiffWriter(tmp, imagej=True) as writer:
            for z in range(N_PLANES):
                v4 = a.pages[z].asarray()
                v7 = b.pages[z].asarray()
                local = fused_plane(v4, v7, accepted.get(z, set()))
                global_plane, next_id = stitch_one(prev_local, prev_global, local, next_id)
                writer.write(global_plane, contiguous=True, photometric="minisblack")
                added_slice_objects += len(accepted.get(z, ()))
                added_pixels += int(np.count_nonzero(local >= SOURCE_OFFSET))
                prev_local, prev_global = local, global_plane
        # The working stack is contiguous for ImageJ/Fiji. Compress only after
        # global stitching so the permanent candidate does not use ~950 MB.
        stacked = tifffile.memmap(tmp)
        if stacked.shape != (N_PLANES, *SHAPE):
            raise RuntimeError(f"Unexpected stitched stack {stacked.shape}")
        tifffile.imwrite(out, stacked, imagej=True, photometric="minisblack",
                         compression="zlib", compressionargs={"level": 1},
                         metadata={"axes": "ZYX", "Info": "v4 base + isolated v7 additions; global IoU stitching"})
        del stacked
        tmp.unlink()
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"distinct_3d_ids": next_id, "added_2d_objects": added_slice_objects,
            "added_pixels": added_pixels, "output_sha256": sha256(out)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Run directory already exists: {OUT}")
    for source, expected in ((V4, EXPECTED_V4), (V7, EXPECTED_V7)):
        if sha256(source) != expected:
            raise RuntimeError(f"Prediction hash differs from frozen input: {source}")
    OUT.mkdir(parents=True)
    events, accepted = scan_candidates()
    result = write_stitched(accepted, OUT / "memb_instances_v4base_v7add_candidate.tif")
    config = {"status": "exploratory_candidate_not_production", "v4_sha256": EXPECTED_V4,
              "v7_sha256": EXPECTED_V7, "selection_uses_human_labels": False,
              "criteria": {"min_area": MIN_AREA, "max_v4_overlap_fraction": MAX_V4_OVERLAP_FRACTION,
                           "min_consecutive_planes": MIN_CONSECUTIVE_PLANES,
                           "min_adjacent_iou": MIN_ADJACENT_IOU,
                           "median_memb_rim_to_core": MIN_MEMBRANE_RATIO,
                           "dark_core_px": MIN_DARK_CORE_PX,
                           "dark_support_planes": MIN_DARK_SUPPORT_PLANES,
                           "global_stitch_iou": STITCH_IOU},
              "scan_events": len(events), "accepted_events": sum(len(s) for s in accepted.values()),
              **result}
    (OUT / "candidate_events.json").write_text(json.dumps(events, indent=2) + "\n")
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
