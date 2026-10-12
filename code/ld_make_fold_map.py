"""Freeze LD image-level CV folds with conservative input-only suspect groups.

The four suspect groups were identified before LD model training from raw
image similarity. They are not claimed to recover organoid or Z identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from tiff_io import list_tiff_files, read_tiff_2d


SUSPECT_GROUPS = (
    ("ld_001.tif", "ld_003.tif"),
    ("ld_006.tif", "ld_007.tif", "ld_034.tif", "ld_035.tif"),
    ("ld_030.tif", "ld_032.tif"),
    ("ld_033.tif", "ld_036.tif"),
)
RAW_CORRELATION_CUTOFF = 0.92
HIGH_PASS_CORRELATION_CUTOFF = 0.08


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standardized_vector(image: np.ndarray) -> np.ndarray:
    flat = image.reshape(-1).astype(np.float32, copy=False)
    centered = flat - flat.mean()
    return centered / (np.linalg.norm(centered) + 1e-9)


def similarity_edges(input_files: list[Path]) -> list[dict[str, object]]:
    """Find unusually similar raw views without reading any LD annotation.

    This reproduces the pre-training audit: full-image percentile scaling,
    exact 4×4 block mean to 300×300, and unshifted global/high-pass Pearson
    correlations. Low-frequency organoid shape alone is not treated as a
    specimen identity; the graph only groups conservative suspect views.
    """

    raw_vectors = []
    high_vectors = []
    for path in input_files:
        image = read_tiff_2d(path).astype(np.float32)
        if image.shape != (1200, 1200):
            raise ValueError(f"Expected 1200×1200 LD input: {path}")
        low, high = np.percentile(image, [1.0, 99.8])
        scaled = np.clip((image - low) / (high - low + 1e-6), 0.0, 1.0)
        small = scaled.reshape(300, 4, 300, 4).mean(axis=(1, 3)).astype(np.float32)
        high_pass = small - gaussian_filter(small, sigma=5)
        raw_vectors.append(_standardized_vector(small))
        high_vectors.append(_standardized_vector(high_pass))

    raw_correlation = np.stack(raw_vectors) @ np.stack(raw_vectors).T
    high_correlation = np.stack(high_vectors) @ np.stack(high_vectors).T
    edges = []
    for left in range(len(input_files)):
        for right in range(left + 1, len(input_files)):
            raw = float(raw_correlation[left, right])
            high = float(high_correlation[left, right])
            if raw >= RAW_CORRELATION_CUTOFF or high >= HIGH_PASS_CORRELATION_CUTOFF:
                edges.append({
                    "left": input_files[left].name,
                    "right": input_files[right].name,
                    "raw_correlation": raw,
                    "high_pass_correlation": high,
                })
    return edges


def _groups_from_edges(
    names: list[str], edges: list[dict[str, object]]
) -> set[frozenset[str]]:
    neighbors = {name: {name} for name in names}
    for edge in edges:
        left, right = str(edge["left"]), str(edge["right"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    groups = set()
    visited = set()
    for name in names:
        if name in visited:
            continue
        pending = [name]
        group = set()
        while pending:
            current = pending.pop()
            if current in group:
                continue
            group.add(current)
            pending.extend(neighbors[current] - group)
        visited.update(group)
        if len(group) > 1:
            groups.add(frozenset(group))
    return groups


def make_fold_map(input_dir: Path, seed: int = 42, num_folds: int = 5) -> dict[str, object]:
    """Group suspect views, then balance fold *image counts* without labels.

    Labels are deliberately not read here. Once written, the map is reused
    unchanged for every LD candidate, so outer-fold assignment cannot drift.
    """

    if num_folds != 5:
        raise ValueError("The predeclared LD study uses exactly five folds")
    input_files = list_tiff_files(input_dir.resolve())
    names = [path.name for path in input_files]
    if len(names) != 36 or names != [f"ld_{index:03d}.tif" for index in range(1, 37)]:
        raise ValueError("The locked LD fold map requires ld_001.tif–ld_036.tif")
    grouped = {name for group in SUSPECT_GROUPS for name in group}
    if len(grouped) != 10 or not grouped.issubset(set(names)):
        raise ValueError("Invalid predeclared suspect groups")
    edges = similarity_edges(input_files)
    observed_groups = _groups_from_edges(names, edges)
    expected_groups = {frozenset(group) for group in SUSPECT_GROUPS}
    if observed_groups != expected_groups:
        raise ValueError("LD input-only similarity groups differ from locked audit")

    rng = np.random.default_rng(seed)
    # Place larger groups first so a four-image suspect group stays intact
    # without unbalancing the final seven- or eight-image outer folds.
    group_order = sorted(
        rng.permutation(len(SUSPECT_GROUPS)),
        key=lambda index: -len(SUSPECT_GROUPS[index]),
    )
    suspect_groups = [SUSPECT_GROUPS[index] for index in group_order]
    singles = [(names[index],) for index in rng.permutation(len(names)) if names[index] not in grouped]
    groups = suspect_groups + singles
    loads = [0] * num_folds
    fold_ids: dict[str, int] = {}
    for group in groups:
        minimum = min(loads)
        candidates = [index for index, load in enumerate(loads) if load == minimum]
        fold = int(rng.choice(candidates))
        for name in group:
            fold_ids[name] = fold
        loads[fold] += len(group)

    if sorted(loads) != [7, 7, 7, 7, 8]:
        raise RuntimeError(f"Unexpected LD fold sizes: {loads}")
    return {
        "target": "ld",
        "num_folds": num_folds,
        "seed": seed,
        "fold_ids": dict(sorted(fold_ids.items())),
        "fold_sizes": loads,
        "suspect_groups": [list(group) for group in SUSPECT_GROUPS],
        "grouping_basis": "pre-training raw-input similarity; no organoid/Z identity inferred",
        "similarity_method": {
            "normalization_percentiles": [1.0, 99.8],
            "thumbnail": "300x300 exact 4x4 block mean",
            "high_pass_gaussian_sigma": 5,
            "correlation": "mean-centered unit-L2 vector dot product, no shift",
            "raw_cutoff": RAW_CORRELATION_CUTOFF,
            "high_pass_cutoff": HIGH_PASS_CORRELATION_CUTOFF,
        },
        "similarity_edges": edges,
        "input_sha256": {path.name: sha256_file(path) for path in input_files},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen fold map: {args.output}")
    result = make_fold_map(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Locked LD folds: {result['fold_sizes']}")


if __name__ == "__main__":
    main()
