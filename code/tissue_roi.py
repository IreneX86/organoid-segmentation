"""Exploratory 2D tissue ROI from explicit, same-grid manual guide masks.

``org`` marks tissue and ``bk`` marks outside/background. They are separate
antialiased uint16 TIFFs, not model predictions. Pixels above the declared
16-bit midpoint are positive. This policy has NOT been validated as an
automatic tissue segmenter and must not be used to tune organelle models.

The conservative gate preserves org-positive and uncertain pixels. The class
map marks conflicting boundaries and other unclassified enclosed regions as
255 for review/training exclusion. Only definite outside and the putative
lumen are removed. The lumen topology must pass conservative checks.
No spatial registration, resampling, or automatic cross-channel pairing occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage as ndi


POLICY = "manual_org_bk_midpoint_enclosed_lumen_v1"
CLASS_VALUES = {0, 1, 2, 255}  # outside, tissue, putative lumen, uncertain
DEFAULT_THRESHOLD = 32767
GATE_POLICY = "remove_outside_and_lumen_preserve_uncertain_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tiff(path: Path, *, dtype: np.dtype) -> np.ndarray:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Expected a TIFF path: {path}")
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != 1 or len(tiff.series) != 1:
            raise ValueError(f"Expected one independent 2D TIFF plane: {path}")
        image = np.asarray(tiff.pages[0].asarray())
    if image.ndim != 2 or image.dtype != dtype:
        raise ValueError(f"Expected one 2D {dtype} TIFF at {path}, got {image.shape} {image.dtype}")
    return image


def _same_grid(*images: np.ndarray) -> tuple[int, int]:
    shapes = [image.shape for image in images]
    if len(set(shapes)) != 1:
        raise ValueError(f"Pixel-grid mismatch: {shapes}; no registration or resizing is allowed")
    return shapes[0]


def _binary_255(array: np.ndarray, *, description: str) -> np.ndarray:
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"{description} must contain only uint8 0/255, got {sorted(values)[:10]}")
    return array == 255


def _cell_ancestor(path: Path) -> Path | None:
    """Catch accidental pairing of same-number masks from different cells."""

    return next((part for part in path.resolve().parents if part.name.startswith("cell_")), None)


def _check_explicit_pair(fluo_path: Path, prediction_path: Path) -> str | None:
    fluo_cell = _cell_ancestor(fluo_path)
    prediction_cell = _cell_ancestor(prediction_path)
    if fluo_cell != prediction_cell:
        raise ValueError("fluo and prediction are not under the same cell directory")
    if prediction_path.name != fluo_path.name:
        fluo_match = re.fullmatch(r"fluo_(\d+)\.tif", fluo_path.name)
        mito_match = re.fullmatch(r"Mito_(\d+)\.tif", prediction_path.name)
        if fluo_match is None or mito_match is None or fluo_match[1] != mito_match[1]:
            raise ValueError("Cross-channel fluo/Mito slice numbers do not match")
    return fluo_cell.name if fluo_cell is not None else None


def derive_tissue_roi(
    fluo: np.ndarray,
    org: np.ndarray,
    bk: np.ndarray,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build tissue 0/255 and a four-class QC map from aligned manual guides.

    The threshold is a *declared coverage convention* for exploratory
    antialiased masks, never selected from organelle predictions or Dice.
    Any severe org/bk contradiction or missing enclosed lumen fails closed.
    """

    if not 0 <= threshold < 65535:
        raise ValueError("threshold must be in the uint16 range 0..65534")
    for name, array in (("fluo", fluo), ("org", org), ("bk", bk)):
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(f"{name} must be one 2D uint16 plane; got {array.shape} {array.dtype}")
    height, width = _same_grid(fluo, org, bk)
    total = height * width
    org_positive = org > threshold
    bk_positive = bk > threshold
    if not org_positive.any() or not bk_positive.any():
        raise ValueError("Both org and bk must have positive pixels; inspect the guide masks")

    overlap = org_positive & bk_positive
    overlap_count = int(overlap.sum())
    tissue_count = int(org_positive.sum())
    if overlap_count / tissue_count > 0.10 or overlap_count / total > 0.05:
        raise ValueError(
            f"Major org/bk conflict: overlap={overlap_count} pixels, "
            f"{overlap_count / tissue_count:.1%} of org; manual review required"
        )

    neither = ~(org_positive | bk_positive)
    labels, component_count = ndi.label(neither, structure=ndi.generate_binary_structure(2, 1))
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    sizes[0] = 0
    border_labels = np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    )))
    enclosed_sizes = sizes.copy()
    enclosed_sizes[border_labels] = 0
    lumen_label = int(np.argmax(enclosed_sizes))
    lumen_count = int(enclosed_sizes[lumen_label])
    enclosed_total = int(enclosed_sizes.sum())
    if lumen_count < max(9, int(np.ceil(total * 0.005))) or lumen_count < 0.5 * enclosed_total:
        raise ValueError("No dominant enclosed neither-marked lumen; inspect org/bk geometry")

    lumen = labels == lumen_label
    rim = ndi.binary_dilation(lumen, structure=ndi.generate_binary_structure(2, 1)) & ~lumen
    org_rim_fraction = float(np.count_nonzero(rim & org_positive) / np.count_nonzero(rim))
    if org_rim_fraction < 0.8:
        raise ValueError(
            f"Enclosed region is not surrounded by tissue: org rim={org_rim_fraction:.1%}"
        )

    # Border-connected neither is outside. Other enclosed neither regions
    # remain uncertain: assigning them to tissue could create false positives.
    border_neither = neither & np.isin(labels, border_labels)
    residual_neither = neither & ~border_neither & ~lumen
    residual_count = int(residual_neither.sum())
    if residual_count / total > 0.10:
        raise ValueError("Too much enclosed unclassified area; manual guide masks need review")

    class_map = np.full((height, width), 255, dtype=np.uint8)
    class_map[bk_positive & ~org_positive] = 0
    class_map[border_neither] = 0
    class_map[org_positive & ~bk_positive] = 1
    class_map[lumen] = 2
    tissue = (org_positive.astype(np.uint8) * 255)
    assert set(np.unique(class_map).tolist()).issubset(CLASS_VALUES)

    qc: dict[str, Any] = {
        "height": height,
        "width": width,
        "threshold": threshold,
        "org_positive_pixels": tissue_count,
        "bk_positive_pixels": int(bk_positive.sum()),
        "org_bk_overlap_pixels": overlap_count,
        "org_bk_overlap_fraction_of_org": overlap_count / tissue_count,
        "org_bk_overlap_fraction_of_image": overlap_count / total,
        "lumen_pixels": lumen_count,
        "lumen_fraction_of_image": lumen_count / total,
        "enclosed_neither_components": int(np.count_nonzero(enclosed_sizes)),
        "lumen_fraction_of_enclosed_neither": lumen_count / enclosed_total,
        "lumen_org_rim_fraction": org_rim_fraction,
        "border_connected_neither_pixels": int(border_neither.sum()),
        "other_enclosed_neither_pixels": residual_count,
        "uncertain_pixels": int(np.count_nonzero(class_map == 255)),
        "class_counts": {str(value): int(np.count_nonzero(class_map == value))
                         for value in (0, 1, 2, 255)},
    }
    return tissue, class_map, qc


def _check_output_paths(inputs: list[Path], outputs: list[Path], suffixes: list[str]) -> None:
    if len(outputs) != len(set(path.resolve() for path in outputs)):
        raise ValueError("Output paths must be distinct")
    resolved_inputs = {path.resolve() for path in inputs}
    for path, suffix in zip(outputs, suffixes):
        if path.suffix.lower() != suffix:
            raise ValueError(f"Output must end in {suffix}: {path}")
        if path.resolve() in resolved_inputs:
            raise ValueError(f"Refusing to write over an input: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _publish(outputs: list[tuple[Path, np.ndarray | dict[str, Any]]]) -> None:
    """Stage complete files, then hard-link only if destinations are absent."""

    temporary: list[Path] = []
    installed: list[Path] = []
    try:
        for destination, data in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{destination.stem}.",
                                        suffix=destination.suffix, dir=destination.parent)
            os.close(fd)
            tmp = Path(name)
            temporary.append(tmp)
            if isinstance(data, np.ndarray):
                tifffile.imwrite(tmp, data, photometric="minisblack", metadata=None)
            else:
                tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for (destination, _), tmp in zip(outputs, temporary):
            os.link(tmp, destination)  # O_EXCL semantics: a concurrent file is never replaced.
            installed.append(destination)
    except BaseException:
        for destination in installed:
            destination.unlink()
        raise
    finally:
        for tmp in temporary:
            tmp.unlink(missing_ok=True)


def prepare(
    fluo_path: Path,
    org_path: Path,
    bk_path: Path,
    tissue_out: Path,
    class_out: Path,
    manifest_out: Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Write new ROI TIFFs and a traceable manifest; never change sources."""

    _check_output_paths([fluo_path, org_path, bk_path],
                        [tissue_out, class_out, manifest_out], [".tif", ".tif", ".json"])
    fluo = _read_tiff(fluo_path, dtype=np.dtype("uint16"))
    org = _read_tiff(org_path, dtype=np.dtype("uint16"))
    bk = _read_tiff(bk_path, dtype=np.dtype("uint16"))
    tissue, classes, qc = derive_tissue_roi(fluo, org, bk, threshold=threshold)
    source_sha = {name: _sha256(path) for name, path in (
        ("fluo", fluo_path), ("org", org_path), ("bk", bk_path)
    )}
    # File-level checksums are computed after deterministic, staged TIFF writes.
    # The manifest records array checksums independent of TIFF metadata.
    array_sha = {
        "tissue_array_sha256": hashlib.sha256(tissue.tobytes()).hexdigest(),
        "class_map_array_sha256": hashlib.sha256(classes.tobytes()).hexdigest(),
    }
    manifest: dict[str, Any] = {
        "policy": POLICY,
        "status": "exploratory_manual_guides_not_validated_tissue_model",
        "source_names": {"fluo": fluo_path.name, "org": org_path.name, "bk": bk_path.name},
        "source_sha256": source_sha,
        "output_names": {"tissue": tissue_out.name, "class_map": class_out.name},
        "gate_policy": GATE_POLICY,
        "class_definitions": {
            "0": "outside/background", "1": "uncontested tissue",
            "2": "largest enclosed neither-marked putative lumen",
            "255": "uncertain overlap or other enclosed neither-marked regions",
        },
        "output_array_sha256": array_sha,
        "qc": qc,
    }
    _publish([(tissue_out, tissue), (class_out, classes), (manifest_out, manifest)])
    return manifest


def apply_gate(
    fluo_path: Path,
    prediction_path: Path,
    tissue_path: Path,
    class_path: Path,
    prepare_manifest_path: Path,
    output_path: Path,
    manifest_out: Path,
    *,
    cross_channel_alignment_confirmed: bool = False,
) -> dict[str, Any]:
    """Suppress positives in definite outside/lumen, retaining uncertain pixels."""

    _check_output_paths([fluo_path, prediction_path, tissue_path, class_path,
                         prepare_manifest_path], [output_path, manifest_out], [".tif", ".json"])
    prepared = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    if prepared.get("policy") != POLICY:
        raise ValueError("Preparation manifest policy mismatch")
    if prepared.get("gate_policy") != GATE_POLICY:
        raise ValueError("Preparation manifest gate policy mismatch")
    cell_id = _check_explicit_pair(fluo_path, prediction_path)
    if (
        prediction_path.name != prepared["source_names"]["fluo"]
        and not cross_channel_alignment_confirmed
    ):
        raise ValueError(
            "Different prediction/fluo filenames require explicit "
            "--cross-channel-alignment-confirmed; no registration is performed"
        )
    if output_path.name != prediction_path.name:
        raise ValueError("Output TIFF must keep the exact prediction filename")
    if (
        fluo_path.name != prepared["source_names"]["fluo"]
        or _sha256(fluo_path) != prepared["source_sha256"]["fluo"]
    ):
        raise ValueError("fluo TIFF does not match the prepared source name/SHA256")
    if (
        tissue_path.name != prepared["output_names"]["tissue"]
        or class_path.name != prepared["output_names"]["class_map"]
    ):
        raise ValueError("Tissue and class-map filenames do not match preparation manifest")
    fluo = _read_tiff(fluo_path, dtype=np.dtype("uint16"))
    prediction = _read_tiff(prediction_path, dtype=np.dtype("uint8"))
    tissue = _read_tiff(tissue_path, dtype=np.dtype("uint8"))
    classes = _read_tiff(class_path, dtype=np.dtype("uint8"))
    _same_grid(fluo, prediction, tissue, classes)
    predicted = _binary_255(prediction, description="Prediction")
    in_tissue = _binary_255(tissue, description="Tissue ROI")
    if not set(np.unique(classes).tolist()).issubset(CLASS_VALUES):
        raise ValueError("Class map must contain only 0/1/2/255")
    for name, array in (("tissue_array_sha256", tissue), ("class_map_array_sha256", classes)):
        if hashlib.sha256(array.tobytes()).hexdigest() != prepared["output_array_sha256"][name]:
            raise ValueError(f"Prepared {name} checksum mismatch")
    if not np.array_equal(in_tissue, (classes == 1) | ((classes == 255) & in_tissue)):
        raise ValueError("Tissue ROI and class map disagree")
    # Uncertain pixels can be annotation gaps near real tissue. Preserve them
    # for human review so an exploratory ROI does not silently create FN.
    allowed = in_tissue | (classes == 255)
    gated = (predicted & allowed).astype(np.uint8) * 255
    if np.any((gated > 0) & ~predicted):
        raise AssertionError("Gate unexpectedly created organelle positives")
    manifest = {
        "policy": POLICY,
        "status": "exploratory_gated_prediction_not_validated_accuracy",
        "gate_policy": GATE_POLICY,
        "cell_id": cell_id,
        "cross_channel_alignment_confirmed_by_analyst": bool(cross_channel_alignment_confirmed),
        "spatial_registration_performed": False,
        "source_names": {"fluo": fluo_path.name, "prediction": prediction_path.name,
                         "tissue": tissue_path.name, "class_map": class_path.name,
                         "prepare_manifest": prepare_manifest_path.name},
        "source_sha256": {"fluo": _sha256(fluo_path), "prediction": _sha256(prediction_path),
                           "tissue": _sha256(tissue_path), "class_map": _sha256(class_path),
                           "prepare_manifest": _sha256(prepare_manifest_path)},
        "output_name": output_path.name,
        "output_array_sha256": hashlib.sha256(gated.tobytes()).hexdigest(),
        "shape": list(gated.shape),
        "input_prediction_positive_pixels": int(predicted.sum()),
        "retained_positive_pixels": int(np.count_nonzero(gated)),
        "removed_prediction_pixels": int(np.count_nonzero(predicted & ~allowed)),
        "removed_inside_putative_lumen_pixels": int(np.count_nonzero(predicted & (classes == 2))),
        "removed_in_definite_outside_pixels": int(np.count_nonzero(predicted & (classes == 0))),
        "retained_uncertain_pixels": int(np.count_nonzero(predicted & (classes == 255))),
    }
    _publish([(output_path, gated), (manifest_out, manifest)])
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Build an exploratory manual tissue ROI")
    for flag in ("fluo", "org", "bk", "tissue-out", "class-out", "manifest-out"):
        prepare_parser.add_argument(f"--{flag}", required=True, type=Path)
    prepare_parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    apply_parser = commands.add_parser("apply", help="Gate one same-grid 0/255 prediction")
    apply_flags = (
        "fluo", "prediction", "tissue", "class-map", "prepare-manifest", "out", "manifest-out"
    )
    for flag in apply_flags:
        apply_parser.add_argument(f"--{flag}", required=True, type=Path)
    apply_parser.add_argument(
        "--cross-channel-alignment-confirmed",
        action="store_true",
        help="Required for differently named prediction channel; no registration occurs",
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.fluo, args.org, args.bk, args.tissue_out,
                         args.class_out, args.manifest_out, threshold=args.threshold)
    else:
        result = apply_gate(args.fluo, args.prediction, args.tissue,
                            args.class_map, args.prepare_manifest, args.out,
                            args.manifest_out,
                            cross_channel_alignment_confirmed=(
                                args.cross_channel_alignment_confirmed
                            ))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
