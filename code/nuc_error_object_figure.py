"""Render an aggregate, image-free Nuc error audit figure from saved metrics.

The input JSON contains only group counts and contrast quantiles. It cannot
reconstruct any biological image or mask. This chart is descriptive: the
internal validation set contains many near-duplicates of training images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def render(summary: dict, destination: Path) -> None:
    counts = [
        ("Missed (<10%)", summary["missed_lt_10pct"], "#b94b4e"),
        ("Partial (10–80%)", summary["partial_10_to_80pct"], "#d8973a"),
        ("Well-covered (≥80%)", summary["well_covered_ge_80pct"], "#2b8293"),
    ]
    total = summary["gt_components_ge_100"]
    distributions = [
        ("Missed GT", summary["missed_core_minus_ring"], "#b94b4e"),
        ("Well-covered GT", summary["well_core_minus_ring"], "#2b8293"),
        ("Isolated FP", summary["isolated_fp_core_minus_ring"], "#775aaa"),
    ]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="580" viewBox="0 0 1040 580">',
        '<rect width="1040" height="580" fill="#f7fafb"/>',
        '<text x="48" y="56" font-family="Arial, sans-serif" font-size="27" font-weight="700" fill="#1d3540">Nuc: archived validation object audit</text>',
        '<text x="48" y="83" font-family="Arial, sans-serif" font-size="14" fill="#516770">Frozen baseline · 34 validation images · threshold 0.45 · GT components ≥100 px</text>',
        '<rect x="42" y="108" width="950" height="382" rx="14" fill="white" stroke="#dde8eb"/>',
        '<text x="66" y="150" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#1d3540">GT object coverage</text>',
        '<text x="545" y="150" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#1d3540">Core − bright-ring contrast</text>',
        '<text x="66" y="177" font-family="Arial, sans-serif" font-size="13" fill="#516770">Fraction of labelled object pixels predicted positive</text>',
        '<text x="545" y="177" font-family="Arial, sans-serif" font-size="13" fill="#516770">P10—median—P90, more negative = darker core</text>',
    ]
    # Stacked proportions keep the small but meaningful missed class visible.
    x, width, y, height = 66.0, 430.0, 205, 38
    for _, count, color in counts:
        segment = width * count / total
        elements.append(f'<rect x="{x:.2f}" y="{y}" width="{segment:.2f}" height="{height}" fill="{color}"/>')
        x += segment
    for index, (label, count, color) in enumerate(counts):
        yy = 283 + 63 * index
        elements.append(f'<circle cx="77" cy="{yy-6}" r="8" fill="{color}"/>')
        elements.append(f'<text x="97" y="{yy}" font-family="Arial, sans-serif" font-size="16" fill="#223a44">{escape(label)}</text>')
        elements.append(f'<text x="446" y="{yy}" text-anchor="end" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#223a44">{count} ({100*count/total:.1f}%)</text>')

    # Fixed x-domain, not tuned after seeing the numbers.
    x_min, x_max = -0.20, 0.02
    start, end = 555.0, 970.0
    def xpos(value):
        return start + (float(value)-x_min)/(x_max-x_min)*(end-start)
    for tick in [-.20,-.15,-.10,-.05,0.0]:
        xx = xpos(tick)
        elements.append(f'<line x1="{xx:.1f}" y1="205" x2="{xx:.1f}" y2="415" stroke="#e2eaed"/>')
        elements.append(f'<text x="{xx:.1f}" y="444" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#516770">{tick:.2f}</text>')
    elements.append('<text x="755" y="469" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#516770">Normalized intensity difference (unitless)</text>')
    for index,(label,dist,color) in enumerate(distributions):
        yy=254+64*index
        left,mid,right=(xpos(dist[key]) for key in ("p10","median","p90"))
        elements.append(f'<text x="555" y="{yy-21}" font-family="Arial, sans-serif" font-size="14" fill="#223a44">{escape(label)} (n={dist["n"]})</text>')
        elements.append(f'<line x1="{left:.1f}" y1="{yy}" x2="{right:.1f}" y2="{yy}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        for pos in (left,right):
            elements.append(f'<line x1="{pos:.1f}" y1="{yy-8}" x2="{pos:.1f}" y2="{yy+8}" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<circle cx="{mid:.1f}" cy="{yy}" r="8" fill="{color}" stroke="white" stroke-width="2"/>')
        elements.append(f'<text x="975" y="{yy+5}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#223a44">{dist["median"]:.3f}</text>')
    elements.extend([
        '<text x="48" y="518" font-family="Arial, sans-serif" font-size="14" fill="#4c5f69">Object-level coverage is descriptive only; it is not independent generalization performance.</text>',
        '<text x="48" y="543" font-family="Arial, sans-serif" font-size="14" fill="#8b4454">Many validation images are near-duplicates of training inputs. FP means predicted component with no GT overlap.</text>',
        '</svg>',
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.summary.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
