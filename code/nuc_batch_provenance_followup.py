"""Read-only audit of Nuc export batches, sequence interruptions, and source matches.

The output contains only filenames and aggregate measurements. Correlation,
filesystem timestamps, and cross-pool matches are provenance *clues*, never a
substitute for microscope acquisition identifiers or verified Z indices.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from nuc_provenance_audit import _groups, features


def _unit(image: np.ndarray) -> np.ndarray:
    vector = image.ravel().astype(np.float32).copy()
    vector -= vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-10)


def _thumbnail(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.shape != (1200, 1200):
        raise ValueError(f"Expected one 1200x1200 TIFF: {path}")
    small = image.astype(np.float32).reshape(60, 20, 60, 20).mean((1, 3))
    low, high = np.percentile(small, (1, 99.5))
    return np.clip((small - low) / max(float(high - low), 1.0), 0, 1)


def _d4(image: np.ndarray, orientation: int) -> np.ndarray:
    # Four right-angle rotations plus their horizontal reflections.
    return np.rot90(image[:, ::-1] if orientation >= 4 else image, orientation % 4)


def _shifted_correlation(
    query: np.ndarray, reference: np.ndarray, max_shift: int = 6
) -> float:
    """Explore small shifts for a coarse candidate, not prove correspondence."""
    best = -1.0
    side = query.shape[0]
    for dy in range(-max_shift, max_shift + 1):
        qy = slice(max(dy, 0), min(side + dy, side))
        ry = slice(max(-dy, 0), min(side - dy, side))
        for dx in range(-max_shift, max_shift + 1):
            qx = slice(max(dx, 0), min(side + dx, side))
            rx = slice(max(-dx, 0), min(side - dx, side))
            best = max(best, float(_unit(query[qy, qx]) @ _unit(reference[ry, rx])))
    return best


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _svg(adjacent: np.ndarray, cohorts: dict[str, list[str]], output: Path) -> None:
    """Draw only aggregate input correlations and export-month markers."""
    x = lambda i: 80 + (i - 1) * 1010 / (len(adjacent) - 1)
    y = lambda corr: 420 - corr * 290
    points = " ".join(
        f"{x(i + 1):.1f},{y(float(corr)):.1f}"
        for i, corr in enumerate(adjacent)
    )
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="590" viewBox="0 0 1200 590">',
        '<rect width="1200" height="590" fill="white"/>',
        '<style>text{font:15px Arial,sans-serif;fill:#18212b}.title{font:bold 24px Arial,sans-serif}.note{font:13px Arial,sans-serif;fill:#475569}</style>',
        '<text x="45" y="38" class="title">Nuc filenames mix export cohorts and interrupt a smooth image sequence</text>',
        '<text x="45" y="65" class="note">Input-only adjacent-image Pearson r; filesystem dates are export clues, not biological Z coordinates.</text>',
        '<line x1="80" y1="420" x2="1090" y2="420" stroke="#444"/>',
        '<line x1="80" y1="130" x2="80" y2="420" stroke="#444"/>',
    ]
    for value in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(value)
        svg.append(
            f'<line x1="80" y1="{yy:.1f}" x2="1090" y2="{yy:.1f}" stroke="#e8edf2"/>'
            f'<text x="39" y="{yy + 5:.1f}">{value:.2f}</text>'
        )
    for index in (1, 20, 40, 60, 80, 100, 120, 140, 160):
        xx = x(index)
        svg.append(
            f'<line x1="{xx:.1f}" y1="420" x2="{xx:.1f}" y2="426" stroke="#444"/>'
            f'<text x="{xx:.1f}" y="445" text-anchor="middle">{index}</text>'
        )
    for cohort, color in (("2025-10", "#d75a5a"), ("2026-05", "#c99321")):
        for filename in cohorts.get(cohort, []):
            index = int(Path(filename).stem.split("_")[1])
            xx = x(index)
            svg.append(
                f'<line x1="{xx:.1f}" y1="125" x2="{xx:.1f}" y2="420" '
                f'stroke="{color}" stroke-width="1" opacity=".42"/>'
            )
    svg.extend(
        [
            f'<polyline fill="none" stroke="#245a81" stroke-width="2.4" points="{points}"/>',
            '<text x="380" y="474">Filename index i for adjacent pair (nuc_i, nuc_{i+1})</text>',
            '<text x="25" y="300" transform="rotate(-90 25 300)">Pearson r</text>',
            '<line x1="55" y1="507" x2="85" y2="507" stroke="#d75a5a" stroke-width="3"/>',
            '<text x="93" y="512">Oct 2025 cohort (10)</text>',
            '<line x1="315" y1="507" x2="345" y2="507" stroke="#c99321" stroke-width="3"/>',
            '<text x="353" y="512">May 2026 additions (9)</text>',
            '<line x1="580" y1="507" x2="610" y2="507" stroke="#245a81" stroke-width="3"/>',
            '<text x="618" y="512">Adjacent image correlation</text>',
            '<text x="55" y="544" class="note">All 148 Nov 2025 files exactly form the largest r ≥ 0.99 connected component; 19 other-cohort files lie outside it.</text>',
            '<text x="55" y="565" class="note">At nuc_010, r(009,011)=0.99913 while r(009,010)=0.39616; numeric adjacency is unsafe for 2.5D training.</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(svg) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuc-input-dir", required=True, type=Path)
    parser.add_argument("--ld-input-dir", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    nuc_paths = sorted(args.nuc_input_dir.glob("nuc_*.tif"))
    ld_paths = sorted(args.ld_input_dir.glob("*.tif"))
    target_paths = sorted(args.target_root.glob("*/fluo/*.tif"))
    if len(nuc_paths) != 167 or len(ld_paths) != 36 or not target_paths:
        raise ValueError("Expected 167 Nuc, 36 LD, and at least one target TIFF")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nuc_vectors = np.stack([features(path)[0] for path in nuc_paths])
    sim = nuc_vectors @ nuc_vectors.T
    adjacent = np.diag(sim, 1)
    grouped_sim = sim.copy()
    np.fill_diagonal(grouped_sim, -1)
    largest = {nuc_paths[i].name for i in _groups(grouped_sim, 0.99)[0]}
    cohorts: dict[str, list[str]] = {}
    for path in nuc_paths:
        # TIFF has no acquisition DateTime tag. mtime is only an export clue;
        # a copy operation can also change it on another machine.
        month = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m")
        cohorts.setdefault(month, []).append(path.name)

    def triplet(index: int) -> dict:
        i = index - 1
        return {
            "image": nuc_paths[i].name,
            "prev_current_r": float(sim[i - 1, i]) if i else None,
            "current_next_r": float(sim[i, i + 1]) if i + 1 < len(nuc_paths) else None,
            "prev_next_r": float(sim[i - 1, i + 1]) if 0 < i < len(nuc_paths) - 1 else None,
        }

    input_hashes = {_sha256(path): path.name for path in nuc_paths}
    anchors = [
        [input_hashes[digest], path.name]
        for path in ld_paths
        if (digest := _sha256(path)) in input_hashes
    ]
    nuc_small = np.stack([_thumbnail(path) for path in nuc_paths])
    target_small = np.stack([_thumbnail(path) for path in target_paths])
    nuc_small_vec = np.stack([_unit(image) for image in nuc_small])
    best_score = np.full(len(nuc_paths), -1.0, np.float32)
    best_index = np.full(len(nuc_paths), -1, np.int32)
    best_orientation = np.full(len(nuc_paths), -1, np.int32)
    for orientation in range(8):
        ref = np.stack([_unit(_d4(image, orientation)) for image in target_small])
        scores = nuc_small_vec @ ref.T
        index = scores.argmax(axis=1)
        score = scores[np.arange(len(nuc_paths)), index]
        improved = score > best_score
        best_score[improved] = score[improved]
        best_index[improved] = index[improved]
        best_orientation[improved] = orientation
    best_shifted_top20 = max(
        _shifted_correlation(
            nuc_small[i],
            _d4(target_small[best_index[i]], int(best_orientation[i])),
        )
        for i in np.argsort(best_score)[-20:]
    )
    date_tag_count = 0
    for path in nuc_paths:
        with tifffile.TiffFile(path) as image:
            date_tag_count += int("DateTime" in image.pages[0].tags)
    summary = {
        "method": "Input-only 100x100 area-mean p1-p99.5 Pearson r for sequence; 60x60 area-mean p1-p99.5 D4 orientation and up-to-6-pixel translation for target candidate search.",
        "warning": "Neither export batches, image similarity, exact cross-pool anchors, nor numeric filenames establish organoid identity, Z index, Z interval, or acquisition order.",
        "nuc_count": len(nuc_paths),
        "target_count": len(target_paths),
        "cohort_counts": {month: len(paths) for month, paths in cohorts.items()},
        "non_nov_files": sorted(set(path.name for path in nuc_paths) - set(cohorts.get("2025-11", []))),
        "r_ge_099_largest_component": len(largest),
        "largest_component_exactly_equals_nov_2025_cohort": largest == set(cohorts.get("2025-11", [])),
        "internal_sequence_interruptions": [triplet(index) for index in (10, 17, 25, 38, 66, 74, 101, 128)],
        "tail_examples": [triplet(index) for index in (158, 159, 160, 161, 165, 166)],
        "nuc_input_tiffs_with_datetime_tag": date_tag_count,
        "nuc_ld_exact_sha_anchors": anchors,
        "target_d4_max_r": float(best_score.max()),
        "target_d4_nuc_count_r_ge_099": int((best_score >= 0.99).sum()),
        "target_best_shifted_r_among_top20_d4_candidates": best_shifted_top20,
    }
    (args.output_dir / "nuc_source_batch_followup_20261016.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _svg(adjacent, cohorts, args.output_dir / "nuc_source_batch_followup_20261016.svg")
    print(json.dumps({key: summary[key] for key in ("cohort_counts", "r_ge_099_largest_component", "largest_component_exactly_equals_nov_2025_cohort", "nuc_ld_exact_sha_anchors", "target_d4_max_r")}, indent=2))


if __name__ == "__main__":
    main()
