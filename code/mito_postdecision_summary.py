"""Audit and summarize a locked Mito selected-policy OOF result.

This reads pre-existing validation-only decisions and selected-policy outer
results. It never selects a model, checkpoint, threshold, or training setting.
Rejected candidate folds must reproduce frozen baseline rows exactly. The
generated JSON and SVG contain aggregates only, never biological pixels or
individual image identities. Repeated use of this 36-image pool is exploratory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from html import escape
from pathlib import Path


METRICS = ("dice", "iou", "precision", "recall")
TOLERANCE = 1e-10


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    by_name = {record["filename"]: record for record in records}
    if len(records) != 36 or len(by_name) != 36:
        raise ValueError("Expected 36 unique frozen baseline OOF rows")
    return by_name


def _number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {label}")
    return number


def _close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise ValueError(f"Frozen OOF mismatch for {label}: {actual} vs {expected}")


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty group")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _metric_group(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"image_count": len(rows), "metrics": {}}
    for metric in METRICS:
        baseline = [float(row[f"baseline_{metric}"]) for row in rows]
        selected = [float(row[metric]) for row in rows]
        paired = [after - before for before, after in zip(baseline, selected)]
        result["metrics"][metric] = {
            "baseline": _stats(baseline),
            "selected_policy": _stats(selected),
            "paired_delta": _stats(paired),
        }
    dice_change = [float(row["dice"]) - float(row["baseline_dice"]) for row in rows]
    result["dice_paired_direction_counts"] = {
        "improved": sum(value > TOLERANCE for value in dice_change),
        "worsened": sum(value < -TOLERANCE for value in dice_change),
        "unchanged": sum(abs(value) <= TOLERANCE for value in dice_change),
    }
    return result


def summarize_locked(
    decisions_path: Path, candidate_oof_path: Path, baseline_csv: Path,
    expected_variant: str,
) -> dict[str, object]:
    """Verify provenance and compute only paired, selected-policy aggregates."""

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_oof_path.read_text(encoding="utf-8"))
    baseline = _load_rows(baseline_csv)
    expected_protocol = f"mito_{expected_variant}_validation_gate_v1"
    if (decisions.get("variant") != expected_variant
            or decisions.get("protocol") != expected_protocol
            or candidate.get("variant") != expected_variant
            or candidate.get("protocol") != "mito_validation_gated_candidate_oof_v1"):
        raise ValueError("Candidate files do not match the locked validation protocol")
    folds = decisions.get("decisions", [])
    if (len(folds) != 5 or [item.get("fold") for item in folds] != list(range(5))
            or any(type(item.get("selected_candidate")) is not bool for item in folds)):
        raise ValueError("Expected five ordered, boolean validation-only decisions")
    per_image = candidate.get("per_image", [])
    by_name = {row["filename"]: row for row in per_image}
    if (len(per_image) != 36 or len(by_name) != 36
            or set(by_name) != set(baseline) or candidate.get("image_count") != 36):
        raise ValueError("Selected-policy OOF must contain each frozen image once")
    selected_count = sum(item["selected_candidate"] for item in folds)
    if candidate.get("selected_candidate_folds") != selected_count:
        raise ValueError("Selected-fold count does not match the locked decisions")

    checked_rows: list[dict[str, object]] = []
    for name in sorted(baseline):
        frozen = baseline[name]
        row = by_name[name]
        fold = int(frozen["fold"])
        if (not 0 <= fold < 5 or row.get("fold") != fold
                or type(row.get("selected_candidate")) is not bool
                or row["selected_candidate"] != folds[fold]["selected_candidate"]):
            raise ValueError(f"Outer-fold/decision mismatch: {name}")
        fraction = _number(frozen["foreground_fraction_true"], f"{name} foreground")
        _close(_number(row["foreground_fraction_true"], f"{name} foreground"),
               fraction, f"{name} foreground")
        checked: dict[str, object] = {"fold": fold, "foreground_fraction": fraction}
        for metric in METRICS:
            reference = _number(frozen[metric], f"{name} frozen {metric}")
            reported_baseline = _number(row[f"baseline_{metric}"], f"{name} baseline {metric}")
            selected = _number(row[metric], f"{name} selected {metric}")
            delta = _number(row[f"delta_{metric}"], f"{name} delta {metric}")
            if not 0 <= reference <= 1 or not 0 <= selected <= 1:
                raise ValueError(f"Invalid segmentation metric: {name} {metric}")
            _close(reported_baseline, reference, f"{name} baseline {metric}")
            _close(delta, selected - reference, f"{name} paired {metric}")
            if not row["selected_candidate"]:
                _close(selected, reference, f"{name} unselected {metric}")
            checked[f"baseline_{metric}"] = reference
            checked[metric] = selected
        checked_rows.append(checked)

    # Recompute rather than trusting stored summaries; this cannot affect the
    # already locked decisions or reveal scores of rejected raw candidates.
    all_metrics = _metric_group(checked_rows)
    low_rows = [row for row in checked_rows if row["foreground_fraction"] <= 0.05]
    high_rows = [row for row in checked_rows if row["foreground_fraction"] > 0.05]
    if len(low_rows) != 7 or len(high_rows) != 29:
        raise ValueError("Frozen low-foreground stratification changed")
    fold_summary = []
    for fold in range(5):
        members = [row for row in checked_rows if row["fold"] == fold]
        if not members:
            raise ValueError(f"Empty fold {fold}")
        baseline_dice = statistics.mean(float(row["baseline_dice"]) for row in members)
        selected_dice = statistics.mean(float(row["dice"]) for row in members)
        fold_summary.append({
            "fold": fold,
            "image_count": len(members),
            "selected_candidate": folds[fold]["selected_candidate"],
            "baseline_dice_mean": baseline_dice,
            "selected_policy_dice_mean": selected_dice,
            "paired_dice_mean_delta": selected_dice - baseline_dice,
        })
    existing_folds = candidate.get("folds", [])
    if len(existing_folds) != 5:
        raise ValueError("Stored OOF report is missing fold summaries")
    for fold, stored in zip(fold_summary, existing_folds):
        if (stored.get("fold") != fold["fold"]
                or stored.get("image_count") != fold["image_count"]
                or stored.get("selected_candidate") != fold["selected_candidate"]):
            raise ValueError("Stored fold summary disagrees with locked rows")
        _close(_number(stored["dice"]["mean"], "stored fold Dice"),
               fold["selected_policy_dice_mean"], f"fold {fold['fold']} mean Dice")
    for metric in METRICS:
        _close(_number(candidate["metrics"][metric]["mean"], "stored metric"),
               all_metrics["metrics"][metric]["selected_policy"]["mean"], metric)
        _close(_number(candidate["paired_delta"][metric]["mean"], "stored paired delta"),
               all_metrics["metrics"][metric]["paired_delta"]["mean"], f"paired {metric}")

    precision_change = [float(row["precision"]) - float(row["baseline_precision"])
                        for row in checked_rows]
    recall_change = [float(row["recall"]) - float(row["baseline_recall"])
                     for row in checked_rows]
    report: dict[str, object] = {
        "analysis": "locked_selected_policy_oof_descriptive_v1",
        "variant": expected_variant,
        "scope": "36 already-inspected labelled development images; exploratory",
        "selected_candidate_folds": selected_count,
        "rejected_folds_reuse_frozen_baseline": True,
        "overall": all_metrics,
        "low_foreground_le_5pct": _metric_group(low_rows),
        "higher_foreground_gt_5pct": _metric_group(high_rows),
        "folds": fold_summary,
        "fold_consistency": {
            "improved_mean_dice": sum(row["paired_dice_mean_delta"] > TOLERANCE
                                      for row in fold_summary),
            "worsened_mean_dice": sum(row["paired_dice_mean_delta"] < -TOLERANCE
                                      for row in fold_summary),
            "unchanged_mean_dice": sum(abs(row["paired_dice_mean_delta"]) <= TOLERANCE
                                       for row in fold_summary),
        },
        "precision_recall_tradeoff": {
            "mean_precision_delta": all_metrics["metrics"]["precision"]["paired_delta"]["mean"],
            "mean_recall_delta": all_metrics["metrics"]["recall"]["paired_delta"]["mean"],
            "precision_down_recall_up_image_count": sum(
                precision < -TOLERANCE and recall > TOLERANCE
                for precision, recall in zip(precision_change, recall_change)
            ),
            "precision_up_recall_down_image_count": sum(
                precision > TOLERANCE and recall < -TOLERANCE
                for precision, recall in zip(precision_change, recall_change)
            ),
        },
        "input_sha256": {
            "locked_decisions": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
            "selected_policy_oof": hashlib.sha256(candidate_oof_path.read_bytes()).hexdigest(),
            "frozen_baseline_oof": hashlib.sha256(baseline_csv.read_bytes()).hexdigest(),
        },
    }
    return report


def render_aggregate_svg(report: dict[str, object]) -> str:
    """Draw only summary statistics; no filenames or pixel data enter SVG."""

    overall = report["overall"]["metrics"]
    sparse = report["low_foreground_le_5pct"]["metrics"]
    folds = report["folds"]
    name = escape(str(report["variant"]).replace("_", " ").title())
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#fcfcfb"/>',
        f'<text x="40" y="42" font-family="sans-serif" font-size="25" fill="#142638">{name} · locked OOF policy</text>',
        '<text x="40" y="66" font-family="sans-serif" font-size="13" fill="#536373">36 labelled development images; exploratory comparison, not independent test</text>',
    ]
    headers = ("All images Dice", "≤5% FG Dice", "Worst Dice", "Precision", "Recall")
    values = (
        (overall["dice"]["baseline"]["mean"], overall["dice"]["selected_policy"]["mean"]),
        (sparse["dice"]["baseline"]["mean"], sparse["dice"]["selected_policy"]["mean"]),
        (overall["dice"]["baseline"]["min"], overall["dice"]["selected_policy"]["min"]),
        (overall["precision"]["baseline"]["mean"], overall["precision"]["selected_policy"]["mean"]),
        (overall["recall"]["baseline"]["mean"], overall["recall"]["selected_policy"]["mean"]),
    )
    for index, (label, (baseline, selected)) in enumerate(zip(headers, values)):
        y = 104 + index * 39
        delta = selected - baseline
        color = "#187458" if delta > TOLERANCE else "#b34943" if delta < -TOLERANCE else "#647585"
        parts.extend((
            f'<text x="40" y="{y}" font-family="sans-serif" font-size="16" fill="#142638">{escape(label)}</text>',
            f'<text x="320" y="{y}" font-family="monospace" font-size="16" fill="#425263">{baseline:.4f} → {selected:.4f}</text>',
            f'<text x="590" y="{y}" font-family="monospace" font-size="16" fill="{color}">{delta:+.4f}</text>',
        ))
    parts.extend((
        '<text x="40" y="333" font-family="sans-serif" font-size="18" fill="#142638">Paired mean Dice change by outer fold</text>',
        '<line x1="480" y1="347" x2="480" y2="494" stroke="#8c999f" stroke-width="1"/>',
    ))
    for index, fold in enumerate(folds):
        y = 370 + index * 29
        delta = float(fold["paired_dice_mean_delta"])
        extent = min(abs(delta) * 6000, 180)
        x = 480 if delta >= 0 else 480 - extent
        color = "#187458" if delta > TOLERANCE else "#b34943" if delta < -TOLERANCE else "#a7b0b5"
        parts.extend((
            f'<text x="40" y="{y + 12}" font-family="sans-serif" font-size="14" fill="#425263">Fold {fold["fold"]} · n={fold["image_count"]}</text>',
            f'<rect x="{x:.1f}" y="{y}" width="{max(extent, 2):.1f}" height="15" fill="{color}"/>',
            f'<text x="700" y="{y + 12}" font-family="monospace" font-size="14" fill="#425263">{delta:+.4f}</text>',
        ))
    parts.append(
        '<text x="40" y="523" font-family="sans-serif" font-size="12" fill="#536373">Rejected folds reuse baseline; bars and values are aggregates only.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-decisions", type=Path, required=True)
    parser.add_argument("--candidate-oof-summary", type=Path, required=True)
    parser.add_argument("--frozen-oof-csv", type=Path, required=True)
    parser.add_argument("--expected-variant", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    if args.summary_json.exists() or args.svg.exists():
        raise FileExistsError("Refusing to overwrite a prior post-decision report")
    if args.svg.suffix.lower() != ".svg":
        raise ValueError("Aggregate chart must have the .svg extension")
    report = summarize_locked(
        args.locked_decisions, args.candidate_oof_summary,
        args.frozen_oof_csv, args.expected_variant,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.svg.write_text(render_aggregate_svg(report), encoding="utf-8")
    print(json.dumps({
        "selected_candidate_folds": report["selected_candidate_folds"],
        "mean_dice": report["overall"]["metrics"]["dice"]["selected_policy"]["mean"],
        "paired_mean_dice_delta": report["overall"]["metrics"]["dice"]["paired_delta"]["mean"],
        "summary_json": str(args.summary_json), "svg": str(args.svg),
    }, indent=2))


if __name__ == "__main__":
    main()
