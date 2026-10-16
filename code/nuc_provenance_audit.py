"""Read-only Nuc provenance and split-dependence audit.

This script does not infer biological identity or Z order from visual similarity.
The similarity graph uses inputs only; a separate mask-IoU summary describes
label continuity. Outputs contain only statistics and a small vector chart.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = tifffile.imread(path)
    if image.shape != (1200, 1200) or image.ndim != 2:
        raise ValueError(f"Unexpected input shape: {path}: {image.shape}")
    # Exact 12x12 area mean; contrast-clipping is computed on the thumbnail.
    small = image.astype(np.float32).reshape(100, 12, 100, 12).mean((1, 3))
    lo, hi = np.percentile(small, (1, 99.5))
    small = np.clip((small - lo) / max(hi - lo, 1.0), 0, 1)
    broad = np.repeat(np.repeat(small.reshape(10, 10, 10, 10).mean((1, 3)), 10, 0), 10, 1)
    return _unit(small), _unit(small - broad)


def _unit(image: np.ndarray) -> np.ndarray:
    v = image.ravel().astype(np.float32)
    v -= v.mean()
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _groups(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    parent = list(range(similarity.shape[0]))

    def root(i: int) -> int:
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(similarity >= threshold, 1))):
        a, b = root(int(i)), root(int(j))
        parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(similarity.shape[0]):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _stat(values: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_matches(queries: list[Path], references: list[Path]) -> list[tuple[str, str]]:
    # Hash *all* files rather than limiting candidates by a visual-similarity
    # threshold, so this is a complete byte-level cross-reference.
    by_hash: dict[str, list[str]] = {}
    for path in queries:
        by_hash.setdefault(_sha256(path), []).append(path.name)
    matches: list[tuple[str, str]] = []
    for path in references:
        for name in by_hash.get(_sha256(path), []):
            matches.append((name, path.name))
    return matches


def _metadata(paths: list[Path]) -> dict:
    pages: Counter[int] = Counter()
    descriptions: Counter[str] = Counter()
    software: Counter[str] = Counter()
    tags: set[str] = set()
    for path in paths:
        with tifffile.TiffFile(path) as tif:
            pages[len(tif.pages)] += 1
            page = tif.pages[0]
            descriptions[page.description or ""] += 1
            software[str(page.tags["Software"].value) if "Software" in page.tags else ""] += 1
            tags.update(tag.name for tag in page.tags.values())
    return {
        "page_counts": dict(pages),
        "image_descriptions": dict(descriptions),
        "software": dict(software),
        "tag_names": sorted(tags),
    }


def _svg(result: dict, out: Path) -> None:
    # No source pixels are embedded: the SVG contains only aggregate correlations.
    gaps = result["index_gap_correlation"]
    rows = []
    for feature in ("global", "local_detail"):
        points = [(int(k), vals[feature]["median"]) for k, vals in gaps.items()]
        pts = " ".join(f"{80+min(x, 80)*6:.1f},{360-y*260:.1f}" for x, y in points)
        color = "#1f77b4" if feature == "global" else "#e07032"
        rows.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}"/>')
        rows.extend(f'<circle cx="{80+min(x, 80)*6:.1f}" cy="{360-y*260:.1f}" r="4" fill="{color}"/>' for x, y in points)
    thresholds = result["similarity_graph"]
    bars = []
    for k, t in enumerate(("0.99", "0.995", "0.999")):
        n = thresholds[t]["largest_global"]
        bars.append(f'<rect x="{660+k*130}" y="{360-n*1.4:.1f}" width="70" height="{n*1.4:.1f}" fill="#5a84a8"/>')
        bars.append(f'<text x="{695+k*130}" y="{350-n*1.4:.1f}" text-anchor="middle">{n}</text>')
        bars.append(f'<text x="{695+k*130}" y="390" text-anchor="middle">{t}</text>')
    val = result["validation_nearest_training"]
    graph = result["similarity_graph"]["0.99"]
    source = result["source_crossmatch"]
    ld = result["ld_reference_crossmatch"]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="520" viewBox="0 0 1100 520">',
        '<rect width="1100" height="520" fill="white"/>',
        '<style>text{font:16px Arial,sans-serif;fill:#18212b}.title{font:bold 22px Arial,sans-serif}.note{font:13px Arial,sans-serif;fill:#4b5563}</style>',
        '<text x="40" y="35" class="title">Nuc input-only similarity audit</text>',
        '<text x="40" y="62" class="note">Similarity is not verified Z order or organoid identity; no labels or raw pixels used in the graph.</text>',
        '<text x="40" y="81" class="note">100×100 area mean; p1–99.5 clip; local detail removes broad 10×10 tile means.</text>',
        '<text x="40" y="105">Correlation by filename index gap (median)</text>',
        '<line x1="80" y1="360" x2="570" y2="360" stroke="#777"/><line x1="80" y1="96" x2="80" y2="360" stroke="#777"/>',
        '<text x="80" y="390">0</text><text x="320" y="390">40</text><text x="550" y="390">80</text>',
        '<text x="52" y="365">0</text><text x="44" y="300">.25</text><text x="44" y="235">.50</text><text x="44" y="170">.75</text><text x="44" y="105">1.0</text>',
        '<text x="280" y="411">Filename index gap |Δindex| (files)</text>',
        '<text x="20" y="250" transform="rotate(-90 20 250)">Pearson r</text>',
        *rows,
        '<line x1="120" y1="435" x2="150" y2="435" stroke="#1f77b4" stroke-width="3"/><text x="160" y="440">Global contrast</text>',
        '<line x1="320" y1="435" x2="350" y2="435" stroke="#e07032" stroke-width="3"/><text x="360" y="440">Local detail</text>',
        '<text x="620" y="105">Largest connected input group</text>',
        '<text x="620" y="128" class="note">Y: images (count); X: correlation threshold r</text>',
        '<line x1="640" y1="360" x2="1080" y2="360" stroke="#777"/>',
        *bars,
        '<text x="755" y="411">Correlation threshold r</text>',
        f'<text x="630" y="430" class="note">At r ≥ 0.99: {graph["global_pairs"]} edges; {graph["largest_global"]}/167 in largest group</text>',
        f'<text x="630" y="453" class="note">Validation nearest train r ≥ 0.99: {val["global_at_least_0.99"]}/34</text>',
        f'<text x="630" y="476" class="note">Crossmatch to {source["fluo_files"]} fluo TIFFs: exact SHA matches {source["exact_sha_matches"]}</text>',
        f'<text x="630" y="499" class="note">LD_input exact SHA anchors: {ld["exact_sha_matches"]}; no acquisition metadata</text>',
        '</svg>',
    ]
    out.write_text("\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fluo-root", type=Path, required=True)
    parser.add_argument("--ld-input-dir", type=Path, required=True)
    parser.add_argument("--mito-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in args.input_dir.glob("nuc_*.tif"))
    if len(names) != 167:
        raise ValueError(f"Expected 167 input TIFFs, got {len(names)}")
    split = json.loads(args.split_manifest.read_text())
    train = np.array([names.index(n) for n in split["train"]])
    val = np.array([names.index(n) for n in split["validation"]])
    whole, detail = zip(*(features(args.input_dir / n) for n in names))
    whole = np.stack(whole)
    detail = np.stack(detail)
    sim = np.clip(whole @ whole.T, -1, 1)
    local = np.clip(detail @ detail.T, -1, 1)
    np.fill_diagonal(sim, -1)
    np.fill_diagonal(local, -1)
    result: dict = {
        "method": "1200x1200 input TIFF -> exact 12x12 area mean to 100x100 -> thumbnail p1-p99.5 clipping -> centered unit vectors; local detail removes independent 10x10 tile means; Pearson dot products",
        "caveat": "Image similarity does not establish organoid identity, acquisition sequence, or Z interval. Filename gap is descriptive only.",
        "input_files": len(names), "split": {"train": len(train), "validation": len(val)},
        "nuc_input_tiff_metadata": _metadata([args.input_dir / n for n in names]),
    }
    result["index_gap_correlation"] = {}
    indices = np.arange(len(names))
    for gap in (1, 2, 3, 5, 10, 20, 40, 80):
        result["index_gap_correlation"][str(gap)] = {
            "global": _stat(sim[indices[:-gap], indices[gap:]]),
            "local_detail": _stat(local[indices[:-gap], indices[gap:]]),
        }
    masks = [tifffile.imread(args.mask_dir / name) > 0 for name in names]
    result["mask_index_gap_iou"] = {
        "method": "Binarize existing GT masks at >0; for each filename-index gap, calculate intersection/union only if both masks are nonempty; report aggregate IoU. This describes label continuity but does not establish Z order or acquisition provenance.",
        "gaps": {},
    }
    for gap in (1, 2, 3, 5, 10, 20, 40):
        values = []
        for i in range(len(masks) - gap):
            a, b = masks[i], masks[i + gap]
            if a.any() and b.any():
                values.append(np.logical_and(a, b).sum() / np.logical_or(a, b).sum())
        result["mask_index_gap_iou"]["gaps"][str(gap)] = {"pairs": len(values), **_stat(np.asarray(values))}
    for label, s in (("global", sim), ("local_detail", local)):
        nearest = np.argmax(s, axis=1)
        same_next = sum(abs(i - int(j)) == 1 for i, j in enumerate(nearest))
        result[f"nearest_neighbor_{label}"] = {
            "adjacent_index_count": same_next,
            "index_distance": _stat(np.abs(indices - nearest)),
            "similarity": _stat(s[indices, nearest]),
        }
    result["similarity_graph"] = {}
    for threshold in (0.99, 0.995, 0.999):
        groups = _groups(sim, threshold)
        local_groups = _groups(local, threshold)
        result["similarity_graph"][str(threshold)] = {
            "global_pairs": int(np.count_nonzero(np.triu(sim >= threshold, 1))),
            "largest_global": len(groups[0]),
            "global_groups": len(groups),
            "global_group_sizes_top10": [len(g) for g in groups[:10]],
            "local_pairs": int(np.count_nonzero(np.triu(local >= threshold, 1))),
            "largest_local": len(local_groups[0]),
            "local_groups": len(local_groups),
        }
    near_global = sim[np.ix_(val, train)].max(axis=1)
    near_local = local[np.ix_(val, train)].max(axis=1)
    result["validation_nearest_training"] = {
        "global": _stat(near_global), "local_detail": _stat(near_local),
        "global_at_least_0.99": int((near_global >= 0.99).sum()),
        "global_at_least_0.995": int((near_global >= 0.995).sum()),
        "local_at_least_0.99": int((near_local >= 0.99).sum()),
        "global_nearest_index_gap_median": float(np.median(np.abs(val - train[np.argmax(sim[np.ix_(val, train)], axis=1)]))),
    }
    fluo_files = sorted(args.fluo_root.glob("*/fluo/*.tif"))
    source_whole, source_local = zip(*(features(p) for p in fluo_files))
    source_sim = np.clip(whole @ np.stack(source_whole).T, -1, 1)
    source_local_sim = np.clip(detail @ np.stack(source_local).T, -1, 1)
    best = np.argmax(source_sim, axis=1)
    source_best = source_sim[np.arange(len(names)), best]
    source_local_best = source_local_sim.max(axis=1)
    input_files = [args.input_dir / n for n in names]
    # Source filenames repeat across cell directories, so preserve relative paths.
    input_hashes = {_sha256(p): p.name for p in input_files}
    exact = []
    for p in fluo_files:
        match = input_hashes.get(_sha256(p))
        if match is not None:
            exact.append((match, p.relative_to(args.fluo_root).as_posix()))
    result["source_crossmatch"] = {
        "fluo_files": len(fluo_files),
        "best_global": _stat(source_best),
        "best_local_detail": _stat(source_local_best),
        "at_least_0.99": int((source_best >= 0.99).sum()),
        "at_least_0.995": int((source_best >= 0.995).sum()),
        "at_least_0.999": int((source_best >= 0.999).sum()),
        "exact_sha_matches": len(exact),
        "exact_match_pairs": exact,
    }
    ld_files = sorted(args.ld_input_dir.glob("*.tif"))
    result["ld_input_tiff_metadata"] = _metadata(ld_files)
    ld_features = np.stack([features(p)[0] for p in ld_files])
    ld_sim = np.clip(whole @ ld_features.T, -1, 1)
    ld_exact = _exact_matches(input_files, ld_files)
    result["ld_reference_crossmatch"] = {
        "ld_input_files": len(ld_files),
        "best_global": _stat(ld_sim.max(axis=1)),
        "at_least_0.99": int((ld_sim.max(axis=1) >= 0.99).sum()),
        "exact_sha_matches": len(ld_exact),
        "exact_match_pairs": ld_exact,
    }
    mito_files = sorted(args.mito_input_dir.glob("*.tif"))
    result["mito_input_tiff_metadata"] = _metadata(mito_files)
    mito_features = np.stack([features(p)[0] for p in mito_files])
    mito_sim = np.clip(whole @ mito_features.T, -1, 1)
    mito_exact = _exact_matches(input_files, mito_files)
    result["mito_reference_crossmatch"] = {
        "mito_input_files": len(mito_files),
        "best_global": _stat(mito_sim.max(axis=1)),
        "at_least_0.99": int((mito_sim.max(axis=1) >= 0.99).sum()),
        "exact_sha_matches": len(mito_exact),
        "exact_match_pairs": mito_exact,
    }
    (args.output_dir / "nuc_provenance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "nuc_validation_nearest_train.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("validation", "nearest_training", "global_correlation", "local_detail_correlation", "filename_index_gap"))
        for i in val:
            j = train[np.argmax(sim[i, train])]
            writer.writerow((names[i], names[j], f"{sim[i, j]:.6f}", f"{local[i, j]:.6f}", abs(int(i - j))))
    _svg(result, args.output_dir / "nuc_provenance_summary.svg")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
