"""Batch-predict exploratory ER masks from paired ER/Fluo Z-stack TIFFs.

This freezes the user-reviewed ridge + density + nearby weak-ridge method.
The first ten-slice trial must reproduce its saved reference exactly. Artifact
seeds are field-specific and never inferred for an unreviewed cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, distance_transform_edt, label


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
from er_ridge_preview import segment_slice  # noqa: E402
from er_ridge_denoise_preview import denoise_stack  # noqa: E402
from er_ridge_relaxed_context_preview import _candidate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclude_temporally_confirmed_artifact(
    mask: np.ndarray, slice_audits: list[dict], padding: int = 5
) -> list[dict]:
    """Resolve an isolated tracker miss only from matched neighboring Z slices.

    The reviewed nuisance cluster occasionally splits into smaller connected
    groups in one plane, moving every group centroid away from the seed. We
    remove mask pixels only when *both* adjacent planes matched the same
    artifact and their bounding boxes overlap almost completely. This is
    intentionally unavailable to cells without a reviewed artifact seed.
    """
    repairs = []
    for z in range(1, len(slice_audits) - 1):
        current = slice_audits[z]["strong"]
        if current.get("artifact_status") != "unresolved_keep_candidates":
            continue
        previous = slice_audits[z - 1]["strong"]
        following = slice_audits[z + 1]["strong"]
        if (
            previous.get("artifact_status") != "matched"
            or following.get("artifact_status") != "matched"
        ):
            continue
        a = previous.get("artifact_group_bbox_yxyx")
        b = following.get("artifact_group_bbox_yxyx")
        if a is None or b is None:
            continue
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        box_iou = intersection / max(area_a + area_b - intersection, 1)
        if box_iou < 0.9:
            continue
        height, width = mask.shape[1:]
        y0 = max(0, min(a[0], b[0]) - padding)
        x0 = max(0, min(a[1], b[1]) - padding)
        y1 = min(height, max(a[2], b[2]) + padding)
        x1 = min(width, max(a[3], b[3]) + padding)
        removed = int(mask[z, y0:y1, x0:x1].sum())
        mask[z, y0:y1, x0:x1] = False
        slice_audits[z]["final_pixels"] = int(mask[z].sum())
        repair = {
            "source_z_zero_based": slice_audits[z]["source_z_zero_based"],
            "neighbor_bbox_iou": float(box_iou),
            "excluded_bbox_yxyx": [y0, x0, y1, x1],
            "removed_pixels": removed,
            "rule": "isolated unmatched plane between two near-identical matched artifact bounds",
        }
        slice_audits[z]["artifact_temporal_fallback"] = repair
        repairs.append(repair)
    return repairs


def pairs(input_root: Path) -> list[tuple[Path, Path, Path]]:
    er_paths = (
        [input_root / "ER.tif"]
        if (input_root / "ER.tif").is_file()
        else sorted(input_root.rglob("ER.tif"))
    )
    if not er_paths:
        raise FileNotFoundError(f"No ER.tif under {input_root}")
    result = []
    for er_path in er_paths:
        fluo_path = er_path.with_name("fluo.tif")
        if not fluo_path.is_file():
            raise FileNotFoundError(f"Missing paired fluo.tif: {er_path.parent}")
        result.append((er_path, fluo_path, er_path.parent.relative_to(input_root)))
    return result


def predict_stack(
    er_path: Path,
    fluo_path: Path,
    z_start: int,
    z_stop: int | None,
    artifact_seed_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, dict]:
    with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file:
        source_shape = er_file.series[0].shape
        if source_shape != fluo_file.series[0].shape or len(source_shape) != 3:
            raise ValueError(f"Misaligned ER/Fluo Z stacks: {er_path.parent}")
        depth, height, width = source_shape
        stop = depth if z_stop is None else z_stop
        if not (0 <= z_start < stop <= depth):
            raise ValueError(f"Invalid Z range {z_start}:{stop} for depth {depth}")
        if len(er_file.pages) != depth or len(fluo_file.pages) != depth:
            raise ValueError("Expected one TIFF page per Z slice")

        count = stop - z_start
        raw_strong = np.empty((count, height, width), dtype=bool)
        seed_by_z: list[tuple[float, float] | None] = []
        strong_audits = []
        tracked_seed = artifact_seed_xy
        for local_z, source_z in enumerate(range(z_start, stop)):
            raw_mask, audit = segment_slice(
                er_file.pages[source_z].asarray(),
                fluo_file.pages[source_z].asarray(),
                artifact_xy=tracked_seed,
                artifact_fail_closed=False,
            )
            raw_strong[local_z] = raw_mask > 0
            seed_by_z.append(tracked_seed)
            if tracked_seed is not None and audit["artifact_group_centroid_xy"] is not None:
                tracked_seed = tuple(audit["artifact_group_centroid_xy"])
            strong_audits.append({"source_z_zero_based": source_z, **audit})
            print(f"{er_path.parent.name} z{source_z + 1:03d}: strong", flush=True)

        strong, density_audits = denoise_stack(raw_strong.astype(np.uint8) * 255)
        del raw_strong
        strong = strong > 0
        weak_near = np.empty_like(strong)
        weak_audits = []
        for local_z, source_z in enumerate(range(z_start, stop)):
            if seed_by_z[local_z] is None:
                # Same weak-ridge rule, without any field-specific exclusion.
                weak, weak_audit = _candidate_no_artifact(
                    er_file.pages[source_z].asarray(), fluo_file.pages[source_z].asarray()
                )
            else:
                weak, weak_audit = _candidate(
                    er_file.pages[source_z].asarray(),
                    fluo_file.pages[source_z].asarray(),
                    seed_by_z[local_z],
                    artifact_fail_closed=False,
                )
            weak_near[local_z] = (
                weak & (distance_transform_edt(~strong[local_z]) <= 40) & ~strong[local_z]
            )
            weak_audits.append(weak_audit)
            print(f"{er_path.parent.name} z{source_z + 1:03d}: weak", flush=True)

        result = strong.copy()
        slice_audits = []
        near_structure = np.ones((5, 5), dtype=bool)
        for local_z, source_z in enumerate(range(z_start, stop)):
            adjacent = np.zeros((height, width), dtype=bool)
            for neighbor in (local_z - 1, local_z + 1):
                if 0 <= neighbor < count:
                    adjacent |= binary_dilation(weak_near[neighbor], structure=near_structure)
            components, n = label(weak_near[local_z])
            areas = np.bincount(components.ravel(), minlength=n + 1)
            overlap = np.bincount(components.ravel(), weights=adjacent.ravel(), minlength=n + 1)
            keep = (areas >= 25) | (overlap >= 2)
            keep[0] = False
            additions = keep[components]
            result[local_z] |= additions
            slice_audits.append({
                "source_z_zero_based": source_z,
                "strong": strong_audits[local_z],
                "density": density_audits[local_z],
                "weak": weak_audits[local_z],
                "added_pixels": int(additions.sum()),
                "final_pixels": int(result[local_z].sum()),
            })
        artifact_repairs = exclude_temporally_confirmed_artifact(result, slice_audits)
        manifest = {
            "method": (
                "frozen exploratory ER ridge + Fluo support + local density "
                "+ adjacent-Z coherence + 40px weak-ridge addition"
            ),
            "status": "candidate prediction, not ground-truth validated",
            "source_er": str(er_path),
            "source_fluo": str(fluo_path),
            "source_shape": list(source_shape),
            "output_shape": [count, height, width],
            "source_z_start_zero_based": z_start,
            "source_z_stop_exclusive": stop,
            "artifact_seed_initial_xy": artifact_seed_xy,
            "background": 0,
            "foreground": 255,
            "slices": slice_audits,
            "artifact_temporal_fallbacks": artifact_repairs,
        }
        return result.astype(np.uint8) * 255, manifest


def _candidate_no_artifact(er: np.ndarray, fluo: np.ndarray) -> tuple[np.ndarray, dict]:
    """Same weak-ridge operation for cells with no reviewed artifact seed."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import sato
    from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects

    er = er.astype(np.float32)
    fluo = fluo.astype(np.float32)
    background = gaussian_filter(er, 25)
    high_pass = gaussian_filter(er, 1) - background
    relative = np.clip((er - background) / np.maximum(background, 250), 0, 2)
    ridge = sato(relative, sigmas=(1, 2, 4), black_ridges=False)
    fluo_detail = gaussian_filter(fluo, 3) - gaussian_filter(fluo, 60)
    support = remove_small_objects(
        binary_dilation(binary_closing(fluo_detail > 30, disk(10)), disk(25)),
        min_size=5000,
    )
    weak = remove_small_objects((ridge > 0.025) & (high_pass > 20) & support, min_size=8)
    return weak, {"artifact_exclusion": "none; no reviewed seed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-seeds", type=Path, help="JSON mapping cell folder names to reviewed seed XY"
    )
    parser.add_argument("--z-start", type=int, default=0)
    parser.add_argument("--z-stop", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="Skip complete SHA-checked output cells"
    )
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    seeds = json.loads(args.artifact_seeds.read_text()) if args.artifact_seeds else {}
    tasks = pairs(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for er_path, fluo_path, relative in tasks:
        target = output_root / relative
        target.mkdir(parents=True, exist_ok=True)
        mask_path = target / "ER_mask.tif"
        manifest_path = target / "ER_mask_manifest.json"
        if args.resume and mask_path.is_file() and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            if existing.get("output_sha256") == sha256(mask_path):
                print(f"{relative}: verified existing output, skipped", flush=True)
                continue
        if mask_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Unverified existing output: {target}")
        seed_record = seeds.get(relative.as_posix(), seeds.get(er_path.parent.name))
        seed_xy = tuple(seed_record["seed_xy"]) if seed_record else None
        result, manifest = predict_stack(er_path, fluo_path, args.z_start, args.z_stop, seed_xy)
        partial = target / "ER_mask.partial.tif"
        tifffile.imwrite(partial, result, imagej=True, compression="zlib")
        partial.rename(mask_path)
        manifest["artifact_seed_note"] = seed_record.get("note") if seed_record else None
        manifest["output_sha256"] = sha256(mask_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{relative}: COMPLETE {result.shape}, {mask_path}", flush=True)


if __name__ == "__main__":
    main()
