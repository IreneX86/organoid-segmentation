"""Render a reporting-only LD five-fold baseline report and static SVG figures.

This script refuses incomplete or altered folds by running ``aggregate_cv``
again, then checks the saved CV CSV/JSON against that result. It never trains,
changes a prediction threshold, or selects a candidate from outer-test scores.
The generated figures contain aggregate metrics only, not biological pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ld_cv_report import aggregate_cv


FOLD_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
LOW_FOREGROUND_FRACTION = 0.001  # Fixed descriptive stratum, not a tuning gate.
PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT = 95, 90, 670, 350


def _text(x: float, y: float, value: object, *, size: int = 14,
          anchor: str = "start", fill: str = "#20252B", weight: str = "normal",
          rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"'
            f' font-family="Arial,Helvetica,sans-serif" font-size="{size}"'
            f' font-weight="{weight}" fill="{fill}"{transform}>'
            f'{escape(str(value))}</text>')


def _line(x1: float, y1: float, x2: float, y2: float, *,
          color: str = "#C8CDD1", width: float = 1, dash: str = "") -> str:
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}"'
            f' y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{pattern}/>' )


def _svg(title: str, description: str, elements: list[str], *,
         width: int = 940, height: int = 555) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" role="img"'
            f' aria-labelledby="title description" viewBox="0 0 {width} {height}">\n'
            f'<title id="title">{escape(title)}</title>\n'
            f'<desc id="description">{escape(description)}</desc>\n'
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>\n'
            + "\n".join(elements) + "\n</svg>\n")


def _figure_frame(title: str, subtitle: str, y_label: str,
                  y_ticks: list[tuple[float, str]]) -> list[str]:
    elements = [_text(45, 38, title, size=22, weight="bold"),
                _text(45, 62, subtitle, size=12, fill="#5D666E")]
    for y, label in y_ticks:
        elements.append(_line(PLOT_LEFT, y, PLOT_LEFT + PLOT_WIDTH, y,
                              color="#E7E9EB"))
        elements.append(_text(PLOT_LEFT - 11, y + 5, label, anchor="end", size=12))
    elements.extend((_line(PLOT_LEFT, PLOT_TOP, PLOT_LEFT,
                           PLOT_TOP + PLOT_HEIGHT, color="#48515A", width=1.5),
                     _line(PLOT_LEFT, PLOT_TOP + PLOT_HEIGHT,
                           PLOT_LEFT + PLOT_WIDTH, PLOT_TOP + PLOT_HEIGHT,
                           color="#48515A", width=1.5),
                     _text(24, PLOT_TOP + PLOT_HEIGHT / 2, y_label,
                           size=14, anchor="middle", rotate=-90)))
    return elements


def _fold_legend(elements: list[str]) -> None:
    elements.append(_text(795, 113, "Outer fold", size=13, weight="bold"))
    for fold, color in enumerate(FOLD_COLORS):
        y = 135 + fold * 25
        elements.append(f'<circle cx="804" cy="{y}" r="6" fill="{color}"/>')
        elements.append(_text(821, y + 4, f"Fold {fold}", size=12))


def _scatter_svg(rows: list[dict[str, object]]) -> str:
    """Use a logarithmic x-axis because LD masks span very sparse ratios."""
    bottom = PLOT_TOP + PLOT_HEIGHT
    elements = _figure_frame(
        "LD OOF Dice versus labelled foreground",
        "36 held-out images · per-image Dice · selected thresholds from inner validation",
        "Dice score", [(PLOT_TOP + PLOT_HEIGHT * (1 - value), f"{value:.1f}")
                       for value in np.linspace(0, 1, 6)])
    x_min, x_max = math.log10(0.005), math.log10(1.0)  # foreground percentage

    def x_position(fraction: float) -> float:
        percent = max(fraction * 100, 0.005)
        return PLOT_LEFT + (math.log10(percent) - x_min) / (x_max - x_min) * PLOT_WIDTH

    for tick in (0.01, 0.03, 0.1, 0.3, 1.0):
        x = x_position(tick / 100)
        elements.append(_line(x, bottom, x, bottom + 6, color="#48515A"))
        elements.append(_text(x, bottom + 24, f"{tick:g}%", anchor="middle", size=12))
    boundary_x = x_position(LOW_FOREGROUND_FRACTION)
    elements.append(_line(boundary_x, PLOT_TOP, boundary_x, bottom,
                          color="#6F777E", dash="5 5"))
    elements.append(_text(boundary_x + 5, PLOT_TOP + 16,
                          "low foreground <=0.10%", size=11, fill="#5D666E"))
    for row in rows:
        fold = int(row["fold"])
        x = x_position(float(row["foreground_fraction_true"]))
        y = PLOT_TOP + (1 - float(row["dice"])) * PLOT_HEIGHT
        label = (f'{row["filename"]}: fold {fold}, foreground '
                 f'{100 * float(row["foreground_fraction_true"]):.4f}%, '
                 f'Dice {float(row["dice"]):.4f}')
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8"'
                        f' fill="{FOLD_COLORS[fold]}" fill-opacity="0.78"'
                        f' stroke="#FFFFFF" stroke-width="1"><title>'
                        f'{escape(label)}</title></circle>')
    elements.append(_text(PLOT_LEFT + PLOT_WIDTH / 2, 505,
                          "Labelled foreground area (% of image; log scale)",
                          anchor="middle", size=14))
    _fold_legend(elements)
    elements.append(_text(45, 537,
                          "Source: locked five-fold LD outer-test reports; one point per TIFF.",
                          size=11, fill="#5D666E"))
    return _svg("LD OOF Dice versus labelled foreground",
                "Logarithmic foreground percentage by full-image Dice; colors mark outer folds.",
                elements)


def _distribution_svg(rows: list[dict[str, object]]) -> str:
    elements = _figure_frame(
        "LD Dice distribution by outer fold",
        "Box: 25th–75th percentile; line: median; whiskers: observed min–max",
        "Dice score", [(PLOT_TOP + PLOT_HEIGHT * (1 - value), f"{value:.1f}")
                       for value in np.linspace(0, 1, 6)])

    def y_position(value: float) -> float:
        return PLOT_TOP + (1 - value) * PLOT_HEIGHT

    for fold in range(5):
        members = [row for row in rows if int(row["fold"]) == fold]
        scores = np.asarray([float(row["dice"]) for row in members])
        minimum, q1, median, q3, maximum = np.percentile(scores, [0, 25, 50, 75, 100])
        x = PLOT_LEFT + PLOT_WIDTH * (fold + 0.5) / 5
        y_q3, y_q1 = y_position(q3), y_position(q1)
        elements.extend((_line(x, y_position(minimum), x, y_position(maximum),
                               color="#4E5961", width=1.5),
                         _line(x - 17, y_position(minimum), x + 17, y_position(minimum),
                               color="#4E5961", width=1.5),
                         _line(x - 17, y_position(maximum), x + 17, y_position(maximum),
                               color="#4E5961", width=1.5),
                         f'<rect x="{x - 29:.1f}" y="{y_q3:.1f}" width="58"'
                         f' height="{max(y_q1 - y_q3, 1):.1f}" fill="{FOLD_COLORS[fold]}"'
                         ' fill-opacity="0.22" stroke="#48515A" stroke-width="1.3"/>',
                         _line(x - 29, y_position(median), x + 29, y_position(median),
                               color="#1B252C", width=2)))
        for row in members:
            digest = hashlib.sha256(str(row["filename"]).encode()).digest()
            jitter = (int.from_bytes(digest[:2], "big") / 65535 - 0.5) * 36
            score = float(row["dice"])
            elements.append(f'<circle cx="{x + jitter:.1f}"'
                            f' cy="{y_position(score):.1f}" r="4.7"'
                            f' fill="{FOLD_COLORS[fold]}" stroke="#FFFFFF"'
                            f' stroke-width="0.8"><title>'
                            f'{escape(str(row["filename"]))}: {score:.4f}</title></circle>')
        elements.append(_text(x, 465, f"Fold {fold}", anchor="middle", size=13))
        elements.append(_text(x, 482, f"n={len(members)}", anchor="middle",
                              size=11, fill="#5D666E"))
    elements.append(_text(PLOT_LEFT + PLOT_WIDTH / 2, 515,
                          "Frozen grouped outer-test fold", anchor="middle", size=14))
    elements.append(_text(45, 537,
                          "Source: locked LD outer-test reports; dots are individual images.",
                          size=11, fill="#5D666E"))
    return _svg("LD Dice distribution by outer fold",
                "Five box plots and individual image Dice scores from fixed grouped folds.",
                elements)


def _threshold_svg(rows: list[dict[str, object]]) -> str:
    """Show a paired comparison, never treat its OOF delta as a tuning gate."""
    changes = [float(row["dice"]) - float(row["fixed_050_dice"]) for row in rows]
    extent = max(0.05, math.ceil(max(map(abs, changes)) * 20) / 20)
    bottom = PLOT_TOP + PLOT_HEIGHT

    def y_position(delta: float) -> float:
        return PLOT_TOP + (extent - delta) / (2 * extent) * PLOT_HEIGHT

    elements = _figure_frame(
        "Paired Dice difference: selected threshold minus 0.50",
        "Same checkpoint and image; selected threshold was chosen on inner validation",
        "Difference in Dice score",
        [(y_position(float(delta)), f"{delta:+.2f}")
         for delta in np.linspace(-extent, extent, 5)])
    elements.append(_line(PLOT_LEFT, y_position(0), PLOT_LEFT + PLOT_WIDTH,
                          y_position(0), color="#33424B", width=1.7))
    for index, row in enumerate(rows):
        x = PLOT_LEFT + PLOT_WIDTH * (index + 0.5) / len(rows)
        delta = changes[index]
        y = y_position(delta)
        fold = int(row["fold"])
        elements.append(_line(x, y_position(0), x, y,
                              color=FOLD_COLORS[fold], width=1.5))
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4"'
                        f' fill="{FOLD_COLORS[fold]}"><title>'
                        f'{escape(str(row["filename"]))}: {delta:+.4f} Dice'
                        '</title></circle>')
        if (index + 1) % 5 == 0 or index == 0 or index == len(rows) - 1:
            elements.append(_text(x, bottom + 22, str(index + 1),
                                  anchor="middle", size=11))
    elements.append(_text(PLOT_LEFT + PLOT_WIDTH / 2, 500,
                          "Image index in filename order", anchor="middle", size=14))
    _fold_legend(elements)
    elements.append(_text(45, 537,
                          "Source: same locked checkpoints; OOF differences are descriptive only.",
                          size=11, fill="#5D666E"))
    return _svg("Paired Dice difference: selected threshold minus 0.50",
                "Differences across 36 held-out images; threshold 0.50 compared with an inner-validation-selected threshold.",
                elements)


def _verified_inputs(run_root: Path, cv_report_dir: Path) -> tuple[list[dict[str, object]], dict]:
    """Require the archived CV CSV/JSON to match recomputation from locked folds."""
    rows, report = aggregate_cv(run_root)
    if len(rows) != 36 or report.get("image_count") != 36:
        raise ValueError("A baseline report requires all 36 unique LD OOF images")
    with (cv_report_dir / "oof_summary.json").open(encoding="utf-8") as handle:
        published = json.load(handle)
    if published != report:
        raise ValueError("Saved CV summary differs from the verified five-fold aggregation")
    with (cv_report_dir / "oof_per_image.csv").open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    if len(saved_rows) != len(rows):
        raise ValueError("Saved CV per-image CSV is incomplete")
    by_name = {str(row["filename"]): row for row in rows}
    if len({row.get("filename") for row in saved_rows}) != 36 or {
        row.get("filename") for row in saved_rows
    } != set(by_name):
        raise ValueError("Saved CV CSV names differ from verified outer-test images")
    for saved in saved_rows:
        expected = by_name[saved["filename"]]
        if set(saved) != set(expected) or int(saved["fold"]) != expected["fold"]:
            raise ValueError(f"Saved CV CSV differs for {saved['filename']}")
        for field in expected:
            if field not in ("filename", "fold") and not math.isclose(
                float(saved[field]), float(expected[field]), rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError(f"Saved CV {field} changed for {saved['filename']}")
    return rows, report


def _metrics_table(report: dict, key: str) -> str:
    lines = ["| Metric | Mean ± sample SD | Median | Minimum | Maximum |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for metric in ("dice", "iou", "precision", "recall"):
        stats = report[key][metric]
        lines.append(f"| {metric.title()} | {stats['mean']:.4f} ± {stats['std']:.4f} | "
                     f"{stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
    return "\n".join(lines)


def _markdown(rows: list[dict[str, object]], report: dict) -> str:
    settings = report["settings"]
    low = [row for row in rows
           if float(row["foreground_fraction_true"]) <= LOW_FOREGROUND_FRACTION]
    high = [row for row in rows
            if float(row["foreground_fraction_true"]) > LOW_FOREGROUND_FRACTION]
    low_mean = float(np.mean([float(row["dice"]) for row in low])) if low else None
    high_mean = float(np.mean([float(row["dice"]) for row in high])) if high else None
    shifts = np.asarray([float(row["dice"]) - float(row["fixed_050_dice"])
                         for row in rows])
    worst = sorted(rows, key=lambda row: float(row["dice"]))[:5]
    code_hashes = settings["code_sha256"]
    code_digest = hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()
    folds = ["| Fold | Held-out images | Validation-selected threshold | Mean Dice | "
             "Median Dice | Min Dice | Mean IoU | Mean precision | Mean recall |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["folds"]:
        folds.append(
            f"| {item['fold']} | {item['n']} | {item['threshold']:.2f} | "
            f"{item['dice']['mean']:.4f} | {item['dice']['median']:.4f} | "
            f"{item['dice']['min']:.4f} | {item['iou']['mean']:.4f} | "
            f"{item['precision']['mean']:.4f} | {item['recall']['mean']:.4f} |"
        )
    low_text = f"{low_mean:.4f}" if low_mean is not None else "not estimable"
    high_text = f"{high_mean:.4f}" if high_mean is not None else "not estimable"
    worst_text = ", ".join(
        f"`{row['filename']}` ({float(row['dice']):.4f})" for row in worst
    )
    return f"""# LD 2D U-Net five-fold baseline

Generated {date.today().isoformat()} from the completed, locked five-fold LD
evaluation. This document is **reporting only**: no candidate model, checkpoint,
mask-conversion rule, or prediction threshold was selected using the scores below.

## Data and locked protocol

All 36 same-name LD image and manual-mask TIFF pairs are labelled development
data, not independent external test cases. Source masks are preserved. The
[pretraining label rule](../../experiments/ld_antialias_label_protocol.md)
converts Photoshop-antialiased masks at 50% of each file's observed maximum
to a separate `uint8` 0/1 mask. Relative to a nonzero rule, the midpoint
excludes 5,125 faint pixels, mostly along boundaries; eight faint connected
regions containing 12 pixels disappear entirely and remain a biological
annotation-review uncertainty. It was fixed before training and never changed
in response to model performance.

The [frozen five-fold protocol](../../experiments/ld_five_fold_baseline_protocol.md)
groups input-only suspected near-duplicates before splitting. Fold sizes are
7, 7, 8, 7, 7. For each outer fold, the next fold is inner validation and the
remaining three folds are training. The LD model starts from random weights:
2D U-Net with four pooling stages, base width 32, 512×512 crops in original
coordinates, BCE + soft-Dice, 0.6 positive-centered crop probability, batch
size 4, Adam learning rate 1e-4, 100 steps/epoch and at most 100 epochs.
Full-resolution 512-pixel overlapping-window validation chooses the checkpoint
at fixed 0.50. Only inner validation selects its deployment threshold. Each
outer-test label is opened after the fold's selection lock is written.

The five-fold reporter rechecked 36 distinct held-out filenames, unchanged
splits and scientific settings, source-code snapshots, checkpoint and threshold
hashes, and saved validation artifacts before producing these figures. Its
sample standard deviation uses `ddof=1` across images. Derived-label manifest
SHA256: `{settings['label_manifest_sha256']}`. Fold-map SHA256:
`{settings['fold_map_sha256']}`. Canonical frozen-source hash-list SHA256:
`{code_digest}`. Small verified metric files are included beside this report;
TIFFs, predictions, and model weights remain outside Git.

## Out-of-fold results

Each image contributes exactly one outer held-out score. The primary reported
policy uses the threshold selected by that fold's inner validation.

{_metrics_table(report, 'oof')}

The same checkpoints evaluated at fixed threshold 0.50 provide an offline
policy comparison; this comparison **must not** be used to pick whichever
policy looks best on outer-test data.

{_metrics_table(report, 'fixed_050_oof')}

The mean paired Dice difference (validation-selected minus fixed 0.50) is
{float(shifts.mean()):+.4f}; {int((shifts > 0).sum())} images improve,
{int((shifts < 0).sum())} worsen, and {int((shifts == 0).sum())} tie. These are
descriptive outer-test observations, not a selection rule.

### Fold-level results

{chr(10).join(folds)}

![OOF Dice against foreground ratio](dice_vs_foreground.svg)

![Per-fold OOF Dice distribution](fold_dice_distribution.svg)

![Paired threshold-policy Dice differences](threshold_policy_difference.svg)

### Sparse-label and worst-case audit

Using a fixed descriptive cutoff of at most 0.10% labelled foreground,
the low-foreground subset has {len(low)} images and mean Dice {low_text}; the
remaining {len(high)} images have mean Dice {high_text}. These summaries are
descriptive and should not define a new model-selection gate after viewing
outer-test results. The five lowest Dice images are {worst_text}. Inspecting
them may reveal hypotheses, but targeted tuning to these same held-out images
would invalidate an unbiased comparison on this 36-image pool.

Pixel-overlap metrics do not by themselves establish droplet-level recall,
boundary accuracy, or annotation consistency. Eight-connected regions no
larger than 25 pixels may be analysed as **semantic component proxies**, never
as verified individual droplets. False-positive distance, false-negative
depth, component-scale detection and representative input/GT/probability
overlays require the separate validation-only diagnostic; no such mechanism
is inferred from these aggregate scores.

## Interpretation and limitations

This result estimates image-level performance within 36 heterogeneous,
manually labelled development images. Similar-input groups were kept intact,
but specimen identity and Z order cannot be recovered, so additional related
views may cross folds. No independently labelled external LD images are
available; prospective generalization is unknown. The antialias midpoint is
an operational annotation convention, not proof that gray edge pixels are
biologically negative. Precision/recall changes must be interpreted alongside
this label uncertainty.

The full OOF and all-fold validation aggregates must not be used to choose
architectures, sampling, loss, threshold or postprocessing and then claim an
untouched comparison on these same outer folds. Because rotated validation
places each image in one fold's validation set and another fold's outer test,
even an all-fold validation error aggregate can indirectly expose every
image. A later single-factor candidate needs a predeclared hypothesis and
fold-local training/validation-only selection before outer evaluation; any
selection using repeated OOF comparisons is exploratory and needs independent
confirmation.

## Reproducibility

- Baseline protocol frozen in private Git commit `6e3727b` before LD GPU work.
- `oof_per_image.csv` and `oof_summary.json` beside this report are exact
  copies of the independently verified five-fold CV outputs.
- Figures are static, self-labelled SVGs generated from those OOF metrics;
  they contain no image pixels or model predictions.
- The GPU run, full checkpoints, TIFF probabilities/masks and logs are kept
  in the local ignored result archive. RunPod costs and cleanup are recorded
  separately from model metrics and cannot be inferred from this report.
"""


def create_report(run_root: Path, cv_report_dir: Path, output_dir: Path) -> Path:
    """Write a new report atomically; refuse to overwrite any previous study."""
    run_root, cv_report_dir, output_dir = (path.resolve() for path in
                                           (run_root, cv_report_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Report destination already exists: {output_dir}")
    rows, report = _verified_inputs(run_root, cv_report_dir)
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"Report parent directory does not exist: {output_dir.parent}")
    scratch = Path(tempfile.mkdtemp(prefix=".ld_report_in_progress_", dir=output_dir.parent))
    try:
        (scratch / "README.md").write_text(_markdown(rows, report), encoding="utf-8")
        (scratch / "dice_vs_foreground.svg").write_text(_scatter_svg(rows), encoding="utf-8")
        (scratch / "fold_dice_distribution.svg").write_text(
            _distribution_svg(rows), encoding="utf-8")
        (scratch / "threshold_policy_difference.svg").write_text(
            _threshold_svg(rows), encoding="utf-8")
        shutil.copyfile(cv_report_dir / "oof_per_image.csv", scratch / "oof_per_image.csv")
        shutil.copyfile(cv_report_dir / "oof_summary.json", scratch / "oof_summary.json")
        os.replace(scratch, output_dir)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return output_dir / "README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Extracted five-fold run root, containing ld/fold_0..4")
    parser.add_argument("--cv-report-dir", type=Path, required=True,
                        help="Existing output of code/ld_cv_report.py")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh reports/ subdirectory; it must not exist")
    args = parser.parse_args()
    print(create_report(args.run_root, args.cv_report_dir, args.output_dir))


if __name__ == "__main__":
    main()
