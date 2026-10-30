"""Batch the reviewed ER coherent-fiber method over aligned ER/Fluo stacks.

Reads existing ridge base masks; writes only to a new output root. The final
ER_mask_line.tif is the user-selected topology-safe candidate. The historical
radius-4 mask is retained beside it for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from er_bright_line_pipeline import process_slice, settings_for_profile, sha256
from er_coherent_fiber import refine_coherent_fiber, ridge_network, settings_record
from er_ridge_batch_predict import pairs


def _base_path(base_root: Path, relative: Path, cell_name: str) -> Path:
    """Accept either a mirrored base tree or the existing flat per-cell tree."""
    mirrored = base_root / relative / "ER_mask.tif"
    flat = base_root / cell_name / "ER_mask.tif"
    if mirrored.is_file():
        return mirrored
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"No ER_mask.tif at {mirrored} or {flat}")


def _write_mask(path: Path, stack: np.ndarray) -> None:
    payload = stack[0] if len(stack) == 1 else stack
    temp = path.with_name(path.stem + ".partial.tif")
    tifffile.imwrite(temp, payload, imagej=True, compression="zlib",
                     metadata={"axes": "ZYX"} if payload.ndim == 3 else None)
    temp.replace(path)


def run_batch(
    input_root: Path,
    base_mask_root: Path,
    output_root: Path,
    *,
    profile_map: dict[str, str] | None = None,
    profile: str = "spheroid_default",
    z_start: int = 0,
    z_stop: int | None = None,
) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    base_mask_root = base_mask_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir() or not base_mask_root.is_dir():
        raise NotADirectoryError("Input root and base-mask root must exist")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    if z_start < 0 or (z_stop is not None and z_stop <= z_start):
        raise ValueError("Invalid Z range")
    tasks = pairs(input_root)
    names = [er.parent.name for er, _, _ in tasks]
    if len(names) != len(set(names)):
        raise ValueError("Cell names must be unique for base-mask and profile resolution")
    if profile_map is not None and set(profile_map) != set(names):
        raise ValueError(f"Profile map must name exactly: {sorted(names)}")
    settings_for_profile(profile)
    resolved = []
    # Validate the entire run before creating any output directory.
    for er_path, fluo_path, relative in tasks:
        base_path = _base_path(base_mask_root, relative, er_path.parent.name)
        with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file, \
                tifffile.TiffFile(base_path) as base_file:
            shape = er_file.series[0].shape
            if len(shape) != 3 or shape != fluo_file.series[0].shape or shape != base_file.series[0].shape:
                raise ValueError(f"Expected aligned ZYX stacks: {er_path.parent}")
            depth = shape[0]
            stop = depth if z_stop is None else z_stop
            if depth < 2 or not 0 <= z_start < stop <= depth:
                raise ValueError(f"Invalid Z range for {er_path.parent}: {z_start}:{stop}")
            if any(len(f.pages) != depth for f in (er_file, fluo_file, base_file)):
                raise ValueError(f"Expected one TIFF page per source Z: {er_path.parent}")
        cell_profile = profile_map[er_path.parent.name] if profile_map is not None else profile
        settings_for_profile(cell_profile)
        manifest_path = base_path.with_name("ER_mask_manifest.json")
        base_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if base_manifest is not None and base_manifest.get("source_shape") != list(shape):
            raise ValueError(f"Base manifest shape mismatch: {manifest_path}")
        bboxes = {}
        if base_manifest is not None:
            for row in base_manifest.get("slices", []):
                bboxes[row["source_z_zero_based"]] = row.get("strong", {}).get("artifact_group_bbox_yxyx")
        resolved.append((er_path, fluo_path, base_path, relative, shape, stop, cell_profile, bboxes))

    output_root.mkdir(parents=True)
    completed = []
    for er_path, fluo_path, base_path, relative, shape, stop, cell_profile, bboxes in resolved:
        target = output_root / relative
        if target != output_root:
            target.mkdir(parents=True, exist_ok=False)
        count = stop - z_start
        masks = {name: np.empty((count, *shape[1:]), dtype=np.uint8)
                 for name in ("ER_mask_bright", "ER_mask_line_legacy", "ER_mask_line")}
        audits = []
        with tifffile.TiffFile(er_path) as er_file, tifffile.TiffFile(fluo_path) as fluo_file, \
                tifffile.TiffFile(base_path) as base_file:
            ridge_cache: dict[int, np.ndarray] = {}

            def ridge_at(index: int) -> np.ndarray:
                if index not in ridge_cache:
                    ridge_cache[index] = ridge_network(er_file.pages[index].asarray())
                return ridge_cache[index]

            for offset, z in enumerate(range(z_start, stop)):
                er = er_file.pages[z].asarray()
                fluo = fluo_file.pages[z].asarray()
                base = base_file.pages[z].asarray()
                bright, old_line, old_audit = process_slice(
                    er, fluo, base, settings=settings_for_profile(cell_profile),
                    artifact_bbox_yxyx=bboxes.get(z),
                )
                adjacent = np.zeros(shape[1:], dtype=bool)
                neighbor_z = [index for index in (z - 1, z + 1) if 0 <= index < shape[0]]
                for index in neighbor_z:
                    adjacent |= ridge_at(index)
                candidate, new_audit = refine_coherent_fiber(ridge_at(z), old_line > 0, adjacent)
                for name, value in (("ER_mask_bright", bright),
                                    ("ER_mask_line_legacy", old_line),
                                    ("ER_mask_line", candidate)):
                    masks[name][offset] = value
                audits.append({"source_z_zero_based": z, "neighbor_z_zero_based": neighbor_z,
                               "legacy": old_audit, "coherent_fiber": new_audit})
                for old_index in tuple(ridge_cache):
                    if old_index < z:
                        del ridge_cache[old_index]
                print(f"{er_path.parent.name} z{z + 1}/{shape[0]} complete", flush=True)

        paths = {}
        for name, mask in masks.items():
            path = target / f"{name}.tif"
            _write_mask(path, mask)
            paths[name] = {"path": str(path), "sha256": sha256(path),
                           "shape": list(mask[0].shape if count == 1 else mask.shape)}
        manifest = {
            "status": "user-selected exploratory ER candidate; not ground-truth validated",
            "method": "topology-safe Z-supported hysteresis fibers over historical ER line",
            "source_er": str(er_path), "source_fluo": str(fluo_path),
            "source_er_sha256": sha256(er_path), "source_fluo_sha256": sha256(fluo_path),
            "source_shape": list(shape), "source_z_start_zero_based": z_start,
            "source_z_stop_exclusive": stop,
            "base_mask": str(base_path), "base_mask_sha256": sha256(base_path),
            "profile": cell_profile,
            "legacy_settings": vars(settings_for_profile(cell_profile)),
            "coherent_fiber_settings": settings_record(),
            "background": 0, "foreground": 255,
            "outputs": paths, "slice_audits": audits,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed.append(target / "ER_mask_line.tif")
    (output_root / "batch_manifest.json").write_text(json.dumps({
        "completed": [str(path) for path in completed],
        "cell_count": len(completed),
    }, indent=2) + "\n")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--base-mask-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--profile-map", type=Path)
    parser.add_argument("--profile", default="spheroid_default",
                        choices=("spheroid_default", "dim_organoid_candidate"))
    parser.add_argument("--z-start", type=int, default=0)
    parser.add_argument("--z-stop", type=int)
    args = parser.parse_args()
    profiles = json.loads(args.profile_map.read_text()) if args.profile_map else None
    run_batch(args.input_root, args.base_mask_root, args.output_root,
              profile_map=profiles, profile=args.profile,
              z_start=args.z_start, z_stop=args.z_stop)


if __name__ == "__main__":
    main()
