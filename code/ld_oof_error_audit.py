"""Exploratory LD OOF error audit, runnable only after the method policy is locked.

The global policy is reconstructed exclusively from saved inner-validation
evidence and compared with the supplied lock *before* opening any ``outer_test``
artifact, original input TIFF, or derived outer-test label. The later OOF audit
does not select a model, threshold, or hyperparameter. Because the rotated
validation folds contain these same 36 images, this OOF description is not an
independent confirmation of the globally selected method.

Example, after ``ld_method_policy.py`` has written its immutable policy::

    python code/ld_oof_error_audit.py \
      --method-policy experiments/results/ld_method_policy.lock.json \
      --baseline-cv-root runpod_results/ld_baseline_20261012/downloaded/full \
      --candidate-cv-root runpod_results/ld_component_20261012/downloaded/full \
      --baseline-archives runpod_results/ld_baseline_20261012/archives \
      --fold-map experiments/results/ld_five_fold_map_v1.json \
      --label-manifest data/ld_binary_midpoint_v1/LD_output/manifest.json \
      --input-dir /path/to/LD_input \
      --label-dir data/ld_binary_midpoint_v1/LD_output \
      --output-dir outputs/ld_oof_error_audit_YYYYMMDD

The output directory must not exist. Only small CSV/JSON descriptions are
written; image, label, probability, and mask TIFFs remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from ld_component_report import aggregate_gated_candidate
from ld_method_policy import POLICY_PATH, _fold_dir, _sha256, build_method_policy
from ld_validation_error_analysis import analyse_case
from tiff_io import read_tiff_2d


# These strata are descriptive and fixed before viewing OOF errors. LD occupies
# much less of an image than Mito; the first boundary matches the existing
# validation-only LD error audit (0.10% foreground).
STRATA = (
    ("at_most_0_1_percent", 0.001),
    ("above_0_1_to_0_5_percent", 0.005),
    ("above_0_5_percent", math.inf),
)
SCORE_FIELDS = ("dice", "iou", "precision", "recall")
COUNT_FIELDS = (
    "tp_pixels", "fp_pixels", "fn_pixels", "fp_within_2px_of_gt",
    "fp_farther_than_10px_from_gt", "fn_within_2px_of_gt_boundary",
    "fn_farther_than_2px_inside_gt", "fn_farther_than_10px_from_prediction",
    "small_component_count_proxy", "small_component_detected_any_proxy",
    "small_component_detected_half_proxy",
)
OPTIONAL_FIELDS = (
    "small_component_recall_any_proxy", "small_component_recall_half_proxy",
    "fp_intensity_median_normalised", "gt_intensity_median_normalised",
    "tn_intensity_median_normalised",
)


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _safe_name(name: object) -> str:
    if (not isinstance(name, str) or not name or Path(name).name != name
            or Path(name).suffix.lower() not in {".tif", ".tiff"}):
        raise ValueError(f"Unsafe LD TIFF filename: {name!r}")
    return name


def verify_method_policy_before_outer_read(
    policy_path: Path, baseline_root: Path, candidate_root: Path,
    fold_map: Path, label_manifest: Path, label_dir: Path,
) -> dict[str, Any]:
    """Fail closed until a complete, unchanged validation-only policy exists.

    This function never resolves an ``outer_test`` path. It first checks the
    policy file and its frozen control-file hashes without reading TIFF pixels;
    only then does it reconstruct the policy from inner-validation evidence.
    That reconstruction verifies derived-label hashes and validation pixels.
    """
    if not policy_path.is_file():
        raise FileNotFoundError(f"Global validation-only method policy is not locked: {policy_path}")
    saved = _json(policy_path)
    if (saved.get("policy_version") != "ld_global_validation_method_v1"
            or saved.get("target") != "ld"
            or saved.get("selected_method") not in {"baseline", "candidate"}
            or saved.get("selection_basis") != "predeclared_five_fold_inner_validation_only"
            or saved.get("outer_test_used_for_selection") is not False
            or saved.get("image_count") != 36):
        raise ValueError("Global method policy is absent, incomplete, or used outer-test evidence")
    if (saved.get("policy_protocol_sha256") != _sha256(POLICY_PATH)
            or saved.get("fold_map_sha256") != _sha256(fold_map)
            or saved.get("label_manifest_sha256") != _sha256(label_manifest)):
        raise ValueError("Global method policy does not match frozen protocol/data manifests")
    gates = saved.get("fold_gate_records")
    if (not isinstance(gates, list) or len(gates) != 5
            or {record.get("fold") for record in gates if isinstance(record, dict)}
            != set(range(5))):
        raise ValueError("Global method policy lacks five locked fold gates")
    reconstructed = build_method_policy(
        baseline_root, candidate_root, fold_map, label_manifest, label_dir
    )
    if saved != reconstructed:
        raise ValueError("Global method policy differs from reconstructed validation-only evidence")
    return saved


def _archived_outer_hashes(archive: Path, fold: int) -> dict[str, str]:
    """Hash original tar member bytes without extracting or trusting tar paths."""
    prefix = f"fold_{fold}/outer_test/"
    seen: set[tuple[str, ...]] = set()
    outer_hashes: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            raw_name = member.name
            parts = raw_name.rstrip("/").split("/")
            if (not raw_name or raw_name.startswith("/") or "\\" in raw_name
                    or any(part in {"", ".", ".."} for part in parts)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe baseline archive member: {archive}: {raw_name!r}")
            canonical = tuple(part.casefold() for part in parts)
            if canonical in seen:
                raise ValueError(f"Duplicate baseline archive member: {archive}: {raw_name!r}")
            seen.add(canonical)
            if "outer_test" in parts and parts[0] != f"fold_{fold}":
                raise ValueError(f"Unexpected fold's outer-test member: {archive}: {raw_name!r}")
            if not raw_name.startswith(prefix) or member.isdir():
                continue
            relative = raw_name[len(prefix):]
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable baseline archive member: {archive}: {raw_name!r}")
            digest = hashlib.sha256()
            with source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            outer_hashes[relative] = digest.hexdigest()
    if not outer_hashes:
        raise ValueError(f"Baseline archive lacks fold {fold} outer-test files: {archive}")
    return outer_hashes


def verify_baseline_outer_archives(baseline_root: Path, archives: Path) -> None:
    """Bind each extracted outer-test file to a checked original fold archive.

    Call only after the validation-only global method policy has passed. This
    guard rejects changed archive bytes, missing/extra extracted files, and
    mismatched file bytes before the OOF report reads any test metrics.
    """
    for fold in range(5):
        archive = archives / f"fold_{fold}.tar.gz"
        sha_file = archives / f"fold_{fold}.sha256"
        lines = sha_file.read_text(encoding="ascii").splitlines()
        record = re.fullmatch(r"([0-9a-f]{64})  (\S+)", lines[0]) if len(lines) == 1 else None
        if record is None or Path(record.group(2)).name != archive.name:
            raise ValueError(f"Invalid baseline archive SHA256 record: {sha_file}")
        if _sha256(archive) != record.group(1):
            raise ValueError(f"Baseline archive SHA256 mismatch: {archive}")
        archived = _archived_outer_hashes(archive, fold)
        outer_root = _fold_dir(baseline_root, fold) / "outer_test"
        if not outer_root.is_dir() or outer_root.is_symlink():
            raise ValueError(f"Missing or unsafe extracted outer-test directory: {outer_root}")
        extracted: dict[str, Path] = {}
        for path in outer_root.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f"Unsafe extracted outer-test path: {path}")
            if path.is_file():
                extracted[path.relative_to(outer_root).as_posix()] = path
        if set(extracted) != set(archived):
            raise ValueError(f"Extracted outer-test file set differs from archive: fold {fold}")
        for relative, path in extracted.items():
            if _sha256(path) != archived[relative]:
                raise ValueError(f"Extracted outer-test bytes differ from archive: {path}")


def _stratum(fraction: float) -> str:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Foreground fraction must be finite and in [0, 1]")
    for name, maximum in STRATA:
        if fraction <= maximum:
            return name
    raise AssertionError("Unreachable foreground stratum")


def _manifest_records(fold_map: Path, label_manifest: Path) -> tuple[dict[str, int], dict[str, str], dict[str, dict]]:
    frozen, manifest = _json(fold_map), _json(label_manifest)
    fold_ids, input_hashes = frozen.get("fold_ids"), frozen.get("input_sha256")
    records = manifest.get("files")
    if (frozen.get("target") != "ld" or frozen.get("num_folds") != 5
            or not isinstance(fold_ids, dict) or len(fold_ids) != 36
            or any(type(fold) is not int or fold not in range(5) for fold in fold_ids.values())
            or not isinstance(input_hashes, dict) or set(input_hashes) != set(fold_ids)
            or manifest.get("rule_id") != "ld_antialias_midpoint_v1"
            or manifest.get("label_semantics") != {"background": 0, "foreground": 1}
            or not isinstance(records, list) or len(records) != 36):
        raise ValueError("Frozen LD fold map or derived-label manifest is incomplete")
    by_name = {item.get("filename"): item for item in records if isinstance(item, dict)}
    if len(by_name) != 36 or set(by_name) != set(fold_ids):
        raise ValueError("Derived-label records do not cover each frozen image exactly once")
    for name in fold_ids:
        _safe_name(name)
        if (by_name[name].get("source_image_sha256") != input_hashes[name]
                or not isinstance(by_name[name].get("derived_mask_sha256"), str)):
            raise ValueError(f"Original input/derived label provenance differs: {name}")
    return fold_ids, input_hashes, by_name


def _read_locked_case(
    run: Path, name: str, image: np.ndarray, truth: np.ndarray,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Check a saved outer-test probability, both masks, and recorded scores."""
    threshold_record = _json(run / "threshold.json")
    lock = _json(run / "selection.lock.json")
    threshold = float(threshold_record.get("threshold", -1))
    if (not 0 < threshold < 1 or lock.get("threshold") != threshold):
        raise ValueError(f"Invalid validation-selected threshold: {run}")
    outer = run / "outer_test"
    probability = read_tiff_2d(outer / "probabilities" / name)
    result = analyse_case(image, truth, probability, threshold)
    for policy, cutoff in (("validation", threshold), ("fixed_050", 0.5)):
        saved = read_tiff_2d(outer / "predictions" / policy / name)
        if (saved.dtype != np.uint8 or saved.shape != image.shape
                or not np.array_equal(saved, (probability >= cutoff).astype(np.uint8) * 255)):
            raise ValueError(f"Saved {policy} outer-test mask differs from probability: {run}/{name}")
    for field in SCORE_FIELDS:
        recorded = float(expected[field])
        if not math.isclose(float(result[field]), recorded, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Saved outer-test {field} differs from TIFF pixels: {run}/{name}")
    return result


def compare_case(
    fold: int, name: str, gate_selected: str,
    baseline: dict[str, Any], selected: dict[str, Any],
) -> dict[str, Any]:
    """Form one purely descriptive paired row from already-verified cases."""
    if gate_selected not in {"baseline", "candidate"}:
        raise ValueError("Fold gate must be baseline or candidate")
    if not math.isclose(
        float(baseline["foreground_fraction_true"]),
        float(selected["foreground_fraction_true"]), rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("Paired results must use the same truth pixel grid")
    row: dict[str, Any] = {
        "fold": fold, "filename": _safe_name(name), "fold_gate_selected": gate_selected,
        "foreground_stratum": _stratum(float(baseline["foreground_fraction_true"])),
        "foreground_fraction_true": float(baseline["foreground_fraction_true"]),
    }
    for field in (*SCORE_FIELDS, *COUNT_FIELDS, *OPTIONAL_FIELDS):
        row[f"baseline_{field}"] = baseline[field]
        row[f"selected_{field}"] = selected[field]
        if field in (*SCORE_FIELDS, *COUNT_FIELDS):
            row[f"paired_{field}_delta"] = selected[field] - baseline[field]
    return row


def _describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Descriptive metric array must be finite and nonempty")
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()), "max": float(array.max()),
    }


def _describe_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"image_count": 0}
    result: dict[str, Any] = {"image_count": len(rows)}
    for field in SCORE_FIELDS:
        for prefix in ("baseline", "selected", "paired"):
            suffix = "_delta" if prefix == "paired" else ""
            key = f"{prefix}_{field}{suffix}"
            result[key] = _describe([float(row[key]) for row in rows])
    for field in COUNT_FIELDS:
        for prefix in ("baseline", "selected", "paired"):
            suffix = "_delta" if prefix == "paired" else ""
            key = f"{prefix}_{field}{suffix}"
            result[f"{key}_total"] = int(sum(int(row[key]) for row in rows))
    for prefix in ("baseline", "selected"):
        total = result[f"{prefix}_small_component_count_proxy_total"]
        for overlap in ("any", "half"):
            detected = result[f"{prefix}_small_component_detected_{overlap}_proxy_total"]
            result[f"{prefix}_small_component_recall_{overlap}_proxy_micro"] = (
                detected / total if total else None
            )
    return result


def summarise_rows(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 36 or len({row["filename"] for row in rows}) != 36:
        raise ValueError("Exploratory OOF audit requires 36 unique image rows")
    return {
        "target": "ld", "image_count": 36,
        "scope": "exploratory internal five-fold OOF error description",
        "model_selection": "none; fold decisions and thresholds were locked before this audit",
        "independence_limit": (
            "The global method policy used five inner-validation appearances of "
            "these same 36 images. This OOF analysis is descriptive and is not "
            "an untouched or independent confirmation of that policy. File/hash "
            "checks establish content integrity, not wall-clock proof of when "
            "the policy was first written."
        ),
        "global_method_policy_selected": policy["selected_method"],
        "fold_gated_procedure": (
            "Use candidate outer-test prediction only where the pre-test fold "
            "gate selected candidate; otherwise reuse baseline outer-test prediction. "
            "This descriptive fold-gated procedure may differ from the global "
            "production method policy."
        ),
        "foreground_strata": {
            "at_most_0_1_percent": "foreground_fraction_true <= 0.001",
            "above_0_1_to_0_5_percent": "0.001 < foreground_fraction_true <= 0.005",
            "above_0_5_percent": "foreground_fraction_true > 0.005",
        },
        "small_component_proxy": (
            "8-connected semantic GT component <=25 pixels; any-overlap and "
            ">=50%-overlap are proxies, not instance segmentation metrics"
        ),
        "distance_units": "native image pixels",
        "all": _describe_group(rows),
        "by_foreground_stratum": {
            name: _describe_group([row for row in rows if row["foreground_stratum"] == name])
            for name, _ in STRATA
        },
        "by_fold": {
            str(fold): _describe_group([row for row in rows if row["fold"] == fold])
            for fold in range(5)
        },
    }


def audit_oof(
    method_policy: Path, baseline_root: Path, candidate_root: Path, baseline_archives: Path,
    fold_map: Path, label_manifest: Path, input_dir: Path, label_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify policy and source archives, then inspect saved OOF cases."""
    policy = verify_method_policy_before_outer_read(
        method_policy, baseline_root, candidate_root, fold_map, label_manifest, label_dir
    )
    # The global policy must pass before archive member names/bytes or any
    # extracted outer-test file are opened. This second guard binds the
    # extracted baseline evidence to the original checked fold archives.
    verify_baseline_outer_archives(baseline_root, baseline_archives)
    paired_rows, gated = aggregate_gated_candidate(
        baseline_root, candidate_root, fold_map, label_manifest, label_dir
    )
    fold_ids, input_hashes, label_records = _manifest_records(fold_map, label_manifest)
    by_name = {str(row["filename"]): row for row in paired_rows}
    if len(by_name) != 36 or set(by_name) != set(fold_ids):
        raise ValueError("Gated and frozen maps disagree on OOF coverage")
    policy_gates = {int(record["fold"]): record for record in policy["fold_gate_records"]}
    report_gates = {int(record["fold"]): record for record in gated["fold_decisions"]}
    for fold in range(5):
        if (policy_gates[fold]["gate_selected"] != report_gates[fold]["selected"]
                or policy_gates[fold]["gate_sha256"] != report_gates[fold]["decision_lock_sha256"]):
            raise ValueError(f"Fold {fold} gate differs from locked global policy")
    rows: list[dict[str, Any]] = []
    for name in sorted(fold_ids):
        fold = fold_ids[name]
        image_path, label_path = input_dir / name, label_dir / name
        record = label_records[name]
        if (_sha256(image_path) != input_hashes[name]
                or _sha256(label_path) != record["derived_mask_sha256"]):
            raise ValueError(f"Input or derived label differs from frozen SHA256: {name}")
        image, truth = read_tiff_2d(image_path), read_tiff_2d(label_path)
        if (image.ndim != 2 or image.dtype.kind not in "ui" or truth.shape != image.shape
                or truth.dtype != np.uint8 or not np.isin(truth, (0, 1)).all()
                or record.get("shape") != list(image.shape)):
            raise ValueError(f"Input/derived label has invalid native grid or semantics: {name}")
        expected = by_name[name]
        baseline = _read_locked_case(
            _fold_dir(baseline_root, fold), name, image, truth,
            {field: expected[f"baseline_{field}"] for field in SCORE_FIELDS},
        )
        gate_selected = str(expected["selected"])
        if gate_selected == "baseline":
            selected = baseline
        else:
            selected = _read_locked_case(
                _fold_dir(candidate_root, fold), name, image, truth,
                {field: expected[f"selected_{field}"] for field in SCORE_FIELDS},
            )
        rows.append(compare_case(fold, name, gate_selected, baseline, selected))
    return rows, summarise_rows(rows, policy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-policy", type=Path, required=True)
    parser.add_argument("--baseline-cv-root", type=Path, required=True)
    parser.add_argument("--candidate-cv-root", type=Path, required=True)
    parser.add_argument("--baseline-archives", type=Path, required=True)
    parser.add_argument("--fold-map", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing OOF audit: {args.output_dir}")
    rows, summary = audit_oof(
        args.method_policy, args.baseline_cv_root, args.candidate_cv_root,
        args.baseline_archives,
        args.fold_map, args.label_manifest, args.input_dir, args.label_dir,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_oof_error_per_image.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "exploratory_oof_error_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "audit_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "target": "ld", "scope": summary["scope"],
            "method_policy_sha256": _sha256(args.method_policy),
            "fold_map_sha256": _sha256(args.fold_map),
            "label_manifest_sha256": _sha256(args.label_manifest),
            "policy_protocol_sha256": _sha256(POLICY_PATH),
            "image_count": 36,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"LD_EXPLORATORY_OOF_AUDIT_COMPLETE images={len(rows)} output={output}")


if __name__ == "__main__":
    main()
