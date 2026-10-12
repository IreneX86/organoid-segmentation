"""Render descriptive LD paired-OOF SVGs after the method policy is locked.

Inputs must be previously verified five-fold CSVs. This module checks their
pairing again, but it does not choose a method, tune a threshold, load TIFFs,
or treat outer-test scores as a model-selection gate. The 36 images appeared in
rotated inner validation, so the charts are exploratory rather than an
independent confirmation of the global production decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from ld_make_report import FOLD_COLORS, _line, _svg, _text


METRICS = ("dice", "iou", "precision", "recall")
LEFT, TOP, WIDTH, HEIGHT = 102, 92, 650, 340
BOTTOM = TOP + HEIGHT


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36 or any(not row.get("filename") for row in rows):
        raise ValueError(f"Expected 36 named OOF image rows: {path}")
    names = [row["filename"] for row in rows]
    if len(set(names)) != 36:
        raise ValueError(f"Duplicated OOF image filename: {path}")
    return rows


def _number(row: dict[str, str], field: str, *, minimum: float,
            maximum: float) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Missing/invalid {field}: {row.get('filename')}") from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Out-of-range {field}: {row.get('filename')}")
    return value


def _close(actual: float, expected: float, field: str, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Paired {field} differs from baseline/selected: {name}")


def _validate_inputs(
    paired_csv: Path, baseline_csv: Path, method_policy: Path,
    audit_csv: Path | None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]] | None]:
    with method_policy.open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if (policy.get("policy_version") != "ld_global_validation_method_v1"
            or policy.get("target") != "ld"
            or policy.get("selection_basis") != "predeclared_five_fold_inner_validation_only"
            or policy.get("outer_test_used_for_selection") is not False
            or policy.get("selected_method") not in {"baseline", "candidate"}
            or policy.get("image_count") != 36):
        raise ValueError("A valid, validation-only LD method policy is required")
    gate_records = policy.get("fold_gate_records")
    if (not isinstance(gate_records, list) or len(gate_records) != 5
            or {record.get("fold") for record in gate_records
                if isinstance(record, dict)} != set(range(5))):
        raise ValueError("Method policy lacks five locked fold gates")
    gate_by_fold = {int(record["fold"]): record.get("gate_selected")
                    for record in gate_records}
    if set(gate_by_fold.values()) - {"baseline", "candidate"}:
        raise ValueError("Method policy has an invalid locked fold gate")
    baseline_rows = _read_csv(baseline_csv)
    paired_rows = _read_csv(paired_csv)
    baseline_by_name = {row["filename"]: row for row in baseline_rows}
    if set(baseline_by_name) != {row["filename"] for row in paired_rows}:
        raise ValueError("Paired and baseline OOF image sets differ")
    normalised: list[dict[str, float | int | str]] = []
    for raw in paired_rows:
        name = raw["filename"]
        baseline = baseline_by_name[name]
        try:
            fold = int(raw["fold"])
            baseline_fold = int(baseline["fold"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid outer fold: {name}") from error
        if fold not in range(5) or fold != baseline_fold:
            raise ValueError(f"Paired image has wrong outer fold: {name}")
        selected = raw.get("selected")
        if selected not in {"candidate", "baseline"}:
            raise ValueError(f"Invalid locked fold choice: {name}")
        if selected != gate_by_fold[fold]:
            raise ValueError(f"Paired image differs from locked fold gate: {name}")
        foreground = _number(raw, "foreground_fraction_true", minimum=0, maximum=1)
        _close(foreground, _number(baseline, "foreground_fraction_true",
                                   minimum=0, maximum=1), "foreground", name)
        row: dict[str, float | int | str] = {
            "filename": name, "fold": fold, "selected": selected,
            "foreground_fraction_true": foreground,
        }
        for metric in METRICS:
            base = _number(raw, f"baseline_{metric}", minimum=0, maximum=1)
            choice = _number(raw, f"selected_{metric}", minimum=0, maximum=1)
            delta = _number(raw, f"paired_{metric}_delta", minimum=-1, maximum=1)
            _close(base, _number(baseline, metric, minimum=0, maximum=1),
                   f"baseline_{metric}", name)
            _close(delta, choice - base, f"{metric} delta", name)
            if selected == "baseline":
                _close(choice, base, f"baseline-selected {metric}", name)
            row[f"baseline_{metric}"] = base
            row[f"selected_{metric}"] = choice
            row[f"paired_{metric}_delta"] = delta
        normalised.append(row)
    if {int(row["fold"]) for row in normalised} != set(range(5)):
        raise ValueError("Paired OOF CSV lacks one or more outer folds")
    normalised.sort(key=lambda row: str(row["filename"]))
    audited: list[dict[str, float | int | str]] | None = None
    if audit_csv is not None:
        audit_rows = _read_csv(audit_csv)
        paired_by_name = {str(row["filename"]): row for row in normalised}
        if {row["filename"] for row in audit_rows} != set(paired_by_name):
            raise ValueError("Error audit and paired OOF image sets differ")
        audited = []
        for raw in audit_rows:
            name = raw["filename"]
            paired = paired_by_name[name]
            if int(raw["fold"]) != paired["fold"] or raw["fold_gate_selected"] != paired["selected"]:
                raise ValueError(f"Error audit fold choice differs: {name}")
            for metric in METRICS:
                for prefix in ("baseline", "selected"):
                    _close(_number(raw, f"{prefix}_{metric}", minimum=0, maximum=1),
                           float(paired[f"{prefix}_{metric}"]), f"audit {prefix}_{metric}", name)
            checked: dict[str, float | int | str] = {"filename": name, "fold": paired["fold"]}
            for kind in ("fp", "fn"):
                base = _number(raw, f"baseline_{kind}_pixels", minimum=0, maximum=1e9)
                choice = _number(raw, f"selected_{kind}_pixels", minimum=0, maximum=1e9)
                delta = _number(raw, f"paired_{kind}_pixels_delta", minimum=-1e9, maximum=1e9)
                _close(delta, choice - base, f"audit {kind} delta", name)
                checked[f"{kind}_delta"] = delta
            audited.append(checked)
        audited.sort(key=lambda row: str(row["filename"]))
    return normalised, audited


def _frame(title: str, subtitle: str, y_label: str,
           ticks: list[tuple[float, str]]) -> list[str]:
    elements = [_text(44, 38, title, size=22, weight="bold"),
                _text(44, 61, subtitle, size=12, fill="#59656E")]
    for y, label in ticks:
        elements.extend((_line(LEFT, y, LEFT + WIDTH, y, color="#E5E9EC"),
                         _text(LEFT - 10, y + 4, label, size=11, anchor="end")))
    elements.extend((_line(LEFT, TOP, LEFT, BOTTOM, color="#40515C", width=1.5),
                     _line(LEFT, BOTTOM, LEFT + WIDTH, BOTTOM,
                           color="#40515C", width=1.5),
                     _text(25, TOP + HEIGHT / 2, y_label, size=13,
                           anchor="middle", rotate=-90)))
    return elements


def _legend(elements: list[str]) -> None:
    elements.append(_text(790, 113, "Outer fold", size=13, weight="bold"))
    for fold, color in enumerate(FOLD_COLORS):
        y = 137 + fold * 24
        elements.append(f'<circle cx="800" cy="{y}" r="5" fill="{color}"/>')
        elements.append(_text(815, y + 4, f"Fold {fold}", size=12))


def _delta_svg(rows: list[dict[str, float | int | str]]) -> str:
    extent = max(0.05, math.ceil(max(abs(float(row["paired_dice_delta"]))
                                    for row in rows) * 20) / 20)
    def y_pos(value: float) -> float:
        return TOP + (extent - value) / (2 * extent) * HEIGHT
    elements = _frame(
        "LD paired OOF Dice change",
        "Fold-gated selected output minus frozen baseline; descriptive only",
        "Dice difference", [(y_pos(value), f"{value:+.2f}")
                            for value in (-extent, -extent / 2, 0, extent / 2, extent)])
    elements.append(_line(LEFT, y_pos(0), LEFT + WIDTH, y_pos(0),
                          color="#263842", width=1.7))
    for index, row in enumerate(rows):
        x = LEFT + WIDTH * (index + .5) / len(rows)
        delta = float(row["paired_dice_delta"])
        y = y_pos(delta)
        color = FOLD_COLORS[int(row["fold"])]
        elements.append(_line(x, y_pos(0), x, y, color=color, width=1.3))
        label = f'{row["filename"]}: {delta:+.4f} Dice, fold {row["fold"]}'
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                        f'<title>{escape(label)}</title></circle>')
    for index in (0, 5, 11, 17, 23, 29, 35):
        x = LEFT + WIDTH * (index + .5) / len(rows)
        elements.append(_text(x, BOTTOM + 21, str(index + 1), size=11, anchor="middle"))
    elements.append(_text(LEFT + WIDTH / 2, 502, "Image index in filename order",
                          size=13, anchor="middle"))
    _legend(elements)
    elements.append(_text(44, 538, "OOF is exploratory; it must not revise the validation-only method lock.",
                          size=11, fill="#59656E"))
    return _svg("LD paired OOF Dice change", "Per-image selected-minus-baseline Dice.", elements)


def _fold_svg(rows: list[dict[str, float | int | str]]) -> str:
    elements = _frame("LD mean OOF Dice by fold",
                      "Same grouped outer folds, frozen baseline versus fold-gated selection",
                      "Mean Dice", [(TOP + HEIGHT * (1 - score), f"{score:.1f}")
                                    for score in (0, .2, .4, .6, .8, 1)])
    for fold in range(5):
        members = [row for row in rows if row["fold"] == fold]
        x = LEFT + WIDTH * (fold + .5) / 5
        for offset, key, fill in ((-20, "baseline_dice", "#75818A"),
                                  (20, "selected_dice", FOLD_COLORS[fold])):
            score = math.fsum(float(row[key]) for row in members) / len(members)
            y = TOP + (1 - score) * HEIGHT
            elements.append(f'<rect x="{x + offset - 16:.1f}" y="{y:.1f}"'
                            f' width="32" height="{BOTTOM - y:.1f}" fill="{fill}">'
                            f'<title>Fold {fold}: {key}={score:.4f}, n={len(members)}</title>'
                            '</rect>')
        elements.append(_text(x, BOTTOM + 23, f"Fold {fold}", size=12, anchor="middle"))
        elements.append(_text(x, BOTTOM + 40, f"n={len(members)}", size=10,
                              anchor="middle", fill="#59656E"))
    elements.extend((f'<rect x="790" y="105" width="12" height="12" fill="#75818A"/>',
                     _text(809, 116, "Frozen baseline", size=11)))
    # The selected bars use a different color for each fold, so the legend
    # mirrors all five colors instead of implying a single blue series.
    for fold, color in enumerate(FOLD_COLORS):
        elements.append(f'<rect x="{790 + 4 * fold}" y="129" width="4"'
                        f' height="12" fill="{color}"/>')
    elements.append(_text(815, 140, "Fold-gated (fold colors)", size=11))
    elements.append(_text(44, 538, "Per-fold means are descriptive, not a model-selection criterion.",
                          size=11, fill="#59656E"))
    return _svg("LD mean OOF Dice by fold", "Paired mean Dice for all five grouped folds.", elements)


def _foreground_svg(rows: list[dict[str, float | int | str]]) -> str:
    x_min, x_max = -3., 0.  # Percent: 0.001% to 1%, enclosing frozen LD range.
    def x_pos(fraction: float) -> float:
        percent = max(100 * fraction, .001)
        return LEFT + (math.log10(percent) - x_min) / (x_max - x_min) * WIDTH
    elements = _frame("LD OOF Dice versus labelled foreground",
                      "Gray: frozen baseline; colored: fold-gated selected output",
                      "Dice", [(TOP + HEIGHT * (1 - score), f"{score:.1f}")
                               for score in (0, .2, .4, .6, .8, 1)])
    for tick in (.001, .01, .1, 1.):
        x = x_pos(tick / 100)
        elements.append(_line(x, BOTTOM, x, BOTTOM + 6, color="#40515C"))
        elements.append(_text(x, BOTTOM + 23, f"{tick:g}%", size=11, anchor="middle"))
    for row in rows:
        fraction = float(row["foreground_fraction_true"])
        if not 0.00001 <= fraction <= .01:
            raise ValueError(f"Foreground outside chart domain: {row['filename']}")
        x = x_pos(fraction)
        base_y = TOP + (1 - float(row["baseline_dice"])) * HEIGHT
        selected_y = TOP + (1 - float(row["selected_dice"])) * HEIGHT
        elements.append(_line(x, base_y, x, selected_y, color="#A4ADB3", width=1.2))
        label = (f'{row["filename"]}: foreground {100 * fraction:.4f}%, '
                 f'baseline {float(row["baseline_dice"]):.4f}, '
                 f'selected {float(row["selected_dice"]):.4f}')
        elements.append(f'<circle cx="{x:.1f}" cy="{base_y:.1f}" r="3.6"'
                        f' fill="#75818A" fill-opacity="0.6"><title>{escape(label)}</title>'
                        '</circle>')
        elements.append(f'<circle cx="{x:.1f}" cy="{selected_y:.1f}" r="5"'
                        f' fill="{FOLD_COLORS[int(row["fold"])]}" fill-opacity="0.75"'
                        f' stroke="#FFFFFF" stroke-width=".8"><title>{escape(label)}</title>'
                        '</circle>')
    elements.append(_text(LEFT + WIDTH / 2, 505,
                          "Labelled foreground percentage (log scale)", size=13,
                          anchor="middle"))
    _legend(elements)
    elements.append(_text(44, 538, "One paired point per labelled TIFF; log-scale x-axis.",
                          size=11, fill="#59656E"))
    return _svg("LD OOF Dice versus labelled foreground",
                "Paired Dice by foreground percentage for 36 held-out images.", elements)


def _fp_fn_svg(rows: list[dict[str, float | int | str]]) -> str:
    extent = max(100., float(math.ceil(max(abs(float(row[key])) for row in rows
                                        for key in ("fp_delta", "fn_delta")) / 100) * 100))
    def x_pos(value: float) -> float:
        return LEFT + (value + extent) / (2 * extent) * WIDTH
    def y_pos(value: float) -> float:
        return TOP + (extent - value) / (2 * extent) * HEIGHT
    elements = _frame("LD paired FP versus FN change",
                      "Selected minus baseline pixel counts; each point is one OOF image",
                      "FN pixel change", [(y_pos(value), f"{value:+.0f}")
                                           for value in (-extent, 0, extent)])
    elements.append(_line(x_pos(0), TOP, x_pos(0), BOTTOM,
                          color="#40515C", width=1.4))
    elements.append(_line(LEFT, y_pos(0), LEFT + WIDTH, y_pos(0),
                          color="#40515C", width=1.4))
    for value in (-extent, 0, extent):
        x = x_pos(value)
        elements.append(_text(x, BOTTOM + 21, f"{value:+.0f}", size=11, anchor="middle"))
    for row in rows:
        x, y = x_pos(float(row["fp_delta"])), y_pos(float(row["fn_delta"]))
        label = (f'{row["filename"]}: FP {float(row["fp_delta"]):+.0f}, '
                 f'FN {float(row["fn_delta"]):+.0f}')
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"'
                        f' fill="{FOLD_COLORS[int(row["fold"])]}" fill-opacity="0.75">'
                        f'<title>{escape(label)}</title></circle>')
    elements.append(_text(LEFT + WIDTH / 2, 503, "FP pixel change", size=13,
                          anchor="middle"))
    _legend(elements)
    elements.append(_text(44, 538, "Counts are pixel-level and descriptive; no threshold is changed here.",
                          size=11, fill="#59656E"))
    return _svg("LD paired FP versus FN change",
                "Selected-minus-baseline FP and FN pixel counts.", elements)


def render_paired_figures(
    paired_csv: Path, baseline_csv: Path, method_policy: Path,
    destination: Path, audit_csv: Path | None = None,
) -> list[Path]:
    """Write deterministic SVGs to a fresh directory after all input checks."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite LD figures: {destination}")
    rows, audited = _validate_inputs(paired_csv, baseline_csv, method_policy, audit_csv)
    figures = {
        "paired_dice_change.svg": _delta_svg(rows),
        "fold_mean_dice.svg": _fold_svg(rows),
        "dice_vs_foreground_paired.svg": _foreground_svg(rows),
    }
    if audited is not None:
        figures["paired_fp_fn_change.svg"] = _fp_fn_svg(audited)
    destination.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, content in figures.items():
        path = destination / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-oof-csv", type=Path, required=True)
    parser.add_argument("--baseline-oof-csv", type=Path, required=True)
    parser.add_argument("--method-policy", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render_paired_figures(args.paired_oof_csv, args.baseline_oof_csv,
                                      args.method_policy, args.output_dir, args.audit_csv):
        print(path)


if __name__ == "__main__":
    main()
