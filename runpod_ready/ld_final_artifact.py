"""Verify or download a completed all-36 LD final model, without changing a Pod.

The baseline runner checks the exact locally audited policy locks; the candidate
runner recomputes its validation selection. Both use ``verify-tree`` before
publishing. The local ``verify`` and ``fetch`` modes recheck the full archive SHA,
Keras ZIP CRC, locked policy, model manifest, and training record before a new
destination directory is published. No outer-test metrics are opened here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_NAME = "ld_final_model.tar.gz"
SHA_NAME = "ld_final_model.sha256"
READY_NAME = "ld_final_model.ready"
MODEL_NAMES = {"baseline": "final_ld_unet.keras", "candidate": "final_ld_component_unet.keras"}
COMPLETE_MARKERS = {
    "baseline": "LD_FINAL_TRAINING_COMPLETE",
    "candidate": "LD_COMPONENT_FINAL_TRAINING_COMPLETE",
}
CANDIDATE_GLOBAL_CHECKS = {
    "candidate_gate_wins_at_least_4_of_5",
    "all_36_mean_paired_dice_delta_at_least_0_01",
    "each_fold_mean_precision_delta_at_least_minus_0_03",
    "no_individual_dice_delta_below_minus_0_10",
}
LOCKED_BASELINE_FINAL_POLICY_SHA256 = "07b5664fa3a14c7c422d92039067aca1d7946a55bb0de8c82d9f4553420538f6"
LOCKED_BASELINE_GLOBAL_POLICY_SHA256 = "7887edaa54a03ff84a550a8ce38b37362a5075ed0bbba1d45a5a5db43c3a9578"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} has no valid SHA256")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch: {path}")


def validate_policy(method: str, policy_path: Path, fold_map: Path,
                    label_manifest: Path) -> dict:
    if method not in MODEL_NAMES:
        raise ValueError("LD final method must be baseline or candidate")
    policy = read_json(policy_path)
    shared = (policy.get("target") == "ld"
              and policy.get("image_count") == 36
              and policy.get("outer_test_used_for_selection") is False
              and policy.get("fold_map_sha256") == sha256(fold_map)
              and policy.get("label_manifest_sha256") == sha256(label_manifest)
              and type(policy.get("final_epoch_count")) is int
              and 1 <= policy["final_epoch_count"] <= 100
              and isinstance(policy.get("deployment_threshold"), (int, float))
              and not isinstance(policy["deployment_threshold"], bool)
              and math.isfinite(float(policy["deployment_threshold"]))
              and 0 < float(policy["deployment_threshold"]) < 1)
    if not shared:
        raise ValueError("Policy is not locked to this 36-image LD dataset")
    if method == "baseline":
        identity = (policy.get("policy_version") == "ld_validation_median_v1"
                    and policy.get("candidate_id") == "ld_baseline_locked"
                    and policy.get("selection_basis") == "five_fold_inner_validation_only")
    else:
        identity = (policy.get("policy_version") == "ld_global_validation_method_v1"
                    and policy.get("selected_method") == "candidate"
                    and policy.get("candidate_id")
                    == "component_uniform_positive_center_v1"
                    and policy.get("selection_basis")
                    == "predeclared_five_fold_inner_validation_only"
                    and isinstance(policy.get("global_checks"), dict)
                    and set(policy["global_checks"]) == CANDIDATE_GLOBAL_CHECKS
                    and all(value is True for value in policy["global_checks"].values()))
    if not identity:
        raise ValueError(f"Policy does not select the requested {method} method")
    settings = policy.get("scientific_settings")
    if not isinstance(settings, dict) or settings.get("patch_size") != 512 or settings.get("overlap") != 128:
        raise ValueError("Policy lacks the frozen LD scientific settings")
    records = policy.get("fold_validation_records")
    if (not isinstance(records, list) or len(records) != 5
            or {item.get("fold") for item in records if isinstance(item, dict)} != set(range(5))):
        raise ValueError("Policy lacks five locked validation records")
    epochs: list[int] = []
    thresholds: list[float] = []
    for item in records:
        epoch = item.get("validation_best_epoch_count")
        threshold = item.get("validation_selected_threshold")
        history_sha = item.get("training_history_sha256")
        if (type(epoch) is not int or not 1 <= epoch <= 100
                or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(float(threshold)) or not 0 < float(threshold) < 1
                or not isinstance(history_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", history_sha) is None):
            raise ValueError("Policy has an invalid validation epoch, threshold or history hash")
        epochs.append(epoch)
        thresholds.append(float(threshold))
    if (policy["final_epoch_count"] != int(statistics.median(epochs))
            or policy["deployment_threshold"] != float(statistics.median(thresholds))):
        raise ValueError("Policy epoch count or threshold differs from validation medians")
    return policy


def _global_contract(method: str, final_policy: dict, global_policy: dict,
                     fold_map: Path, label_manifest: Path) -> None:
    expected_id = ("ld_baseline_locked" if method == "baseline"
                   else "component_uniform_positive_center_v1")
    if (global_policy.get("policy_version") != "ld_global_validation_method_v1"
            or global_policy.get("target") != "ld"
            or global_policy.get("selected_method") != method
            or global_policy.get("candidate_id") != expected_id
            or global_policy.get("selection_basis")
            != "predeclared_five_fold_inner_validation_only"
            or global_policy.get("outer_test_used_for_selection") is not False
            or global_policy.get("image_count") != 36
            or global_policy.get("fold_map_sha256") != sha256(fold_map)
            or global_policy.get("label_manifest_sha256") != sha256(label_manifest)):
        raise ValueError("Requested LD final method differs from the locked global selection")
    if (final_policy.get("final_epoch_count") != global_policy.get("final_epoch_count")
            or final_policy.get("deployment_threshold")
            != global_policy.get("deployment_threshold")
            or final_policy.get("scientific_settings")
            != global_policy.get("scientific_settings")):
        raise ValueError("Final policy epoch, threshold or settings differ from global selection")
    if method == "candidate" and final_policy != global_policy:
        raise ValueError("Candidate final policy must be the exact global selection policy")


def validate_locked_baseline(
    final_policy_path: Path, global_policy_path: Path, fold_map: Path,
    label_manifest: Path,
) -> dict[str, object]:
    """Check the exact baseline locks already recomputed from local CV evidence."""
    require_hash(final_policy_path, LOCKED_BASELINE_FINAL_POLICY_SHA256,
                 "Locked baseline final policy")
    require_hash(global_policy_path, LOCKED_BASELINE_GLOBAL_POLICY_SHA256,
                 "Locked global method policy")
    final_policy = validate_policy("baseline", final_policy_path, fold_map,
                                   label_manifest)
    global_policy = read_json(global_policy_path)
    _global_contract("baseline", final_policy, global_policy, fold_map,
                     label_manifest)
    return {
        "method": "baseline",
        "final_policy_sha256": LOCKED_BASELINE_FINAL_POLICY_SHA256,
        "global_policy_sha256": LOCKED_BASELINE_GLOBAL_POLICY_SHA256,
        "epochs": final_policy["final_epoch_count"],
        "threshold": final_policy["deployment_threshold"],
    }


def validate_global_selection(
    method: str, final_policy_path: Path, global_policy_path: Path,
    baseline_cv_root: Path, candidate_cv_root: Path, fold_map: Path,
    label_manifest: Path, label_dir: Path, study_root: Path, *, builder=None,
) -> dict[str, object]:
    """Recompute the validation-only choice before any final-fit directory exists."""
    final_policy = validate_policy(method, final_policy_path, fold_map, label_manifest)
    provided = read_json(global_policy_path)
    if builder is None:
        code_dir = study_root / "code"
        if not code_dir.is_dir():
            raise FileNotFoundError(f"LD policy code is missing: {code_dir}")
        sys.path.insert(0, str(code_dir))
        from ld_method_policy import build_method_policy
        builder = build_method_policy
    recomputed = builder(baseline_cv_root, candidate_cv_root,
                         fold_map, label_manifest, label_dir)
    if provided != recomputed:
        raise ValueError("Provided global method policy differs from recomputed validation evidence")
    _global_contract(method, final_policy, recomputed, fold_map, label_manifest)
    return {"method": method, "global_policy_sha256": sha256(global_policy_path),
            "final_policy_sha256": sha256(final_policy_path),
            "epochs": final_policy["final_epoch_count"],
            "threshold": final_policy["deployment_threshold"]}


def verify_tree(tree: Path, method: str, policy_path: Path, fold_map: Path,
                label_manifest: Path, global_policy_path: Path) -> dict[str, object]:
    if method == "baseline":
        validate_locked_baseline(policy_path, global_policy_path, fold_map,
                                 label_manifest)
        policy = read_json(policy_path)
    else:
        policy = validate_policy(method, policy_path, fold_map, label_manifest)
        global_policy = read_json(global_policy_path)
        _global_contract(method, policy, global_policy, fold_map, label_manifest)
    model = tree / "model"
    logs = tree / "logs"
    if not model.is_dir() or not logs.is_dir():
        raise ValueError("Final archive lacks model or logs directory")
    selected = read_json(logs / "selected_method.json")
    policy_digest = sha256(policy_path)
    global_digest = sha256(global_policy_path)
    if (selected.get("method") != method
            or selected.get("policy_sha256") != policy_digest
            or selected.get("global_policy_sha256") != global_digest
            or selected.get("fold_map_sha256") != sha256(fold_map)
            or selected.get("label_manifest_sha256") != sha256(label_manifest)):
        raise ValueError("Runner method/policy record differs from the locked inputs")
    require_hash(logs / "global_method_policy.json", global_digest,
                 "Locked global method policy")
    if (logs / "runner_exit_status.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("Final trainer did not exit successfully")
    preflight_log = ("global_policy_preflight.log" if method == "baseline"
                     else "global_selection_preflight.log")
    for name in ("start_utc.txt", "end_utc.txt", "hostname.txt", "runner.sh",
                 "gpu_usage.csv", "gpu_preflight.log", "nvidia_smi_list.txt",
                 "training.log", preflight_log):
        if not (logs / name).is_file() or (logs / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty final-run log: {name}")
    if method == "baseline":
        preflight_marker = (f"LD_GLOBAL_METHOD_LOCK_PASS method=baseline "
                            f"final_policy_sha256={policy_digest} "
                            f"global_policy_sha256={global_digest}")
    else:
        preflight_marker = (f"LD_GLOBAL_METHOD_SELECTION_PASS method=candidate "
                            f"global_policy_sha256={global_digest}")
    if preflight_marker not in (logs / preflight_log).read_text(encoding="utf-8"):
        raise ValueError("Global method preflight marker is missing")
    if "LD_GPU_PREFLIGHT_PASS" not in (logs / "gpu_preflight.log").read_text(encoding="utf-8"):
        raise ValueError("GPU preflight success marker is missing")
    gpu_lines = (logs / "nvidia_smi_list.txt").read_text(encoding="utf-8").splitlines()
    if len(gpu_lines) != 1 or "NVIDIA A40" not in gpu_lines[0]:
        raise ValueError("Final run did not record exactly one NVIDIA A40")
    if COMPLETE_MARKERS[method] not in (logs / "training.log").read_text(encoding="utf-8"):
        raise ValueError("Final training completion marker is missing")

    require_hash(model / "production_policy.json", policy_digest, "Locked policy")
    config = read_json(model / "final_fit_config.json")
    manifest = read_json(model / "final_model_manifest.json")
    threshold = read_json(model / "threshold.json")
    environment = read_json(model / "environment.json")
    report = read_json(model / "dataset_report.json")
    expected_model = MODEL_NAMES[method]
    if (config.get("target") != "ld" or config.get("image_count") != 36
            or config.get("epochs") != policy["final_epoch_count"]
            or config.get("policy_sha256") != policy_digest
            or config.get("fold_map_sha256") != sha256(fold_map)
            or config.get("label_manifest_sha256") != sha256(label_manifest)
            or config.get("scientific_settings") != policy["scientific_settings"]):
        raise ValueError("Final-fit config differs from locked policy or dataset")
    if method == "candidate" and config.get("candidate_id") != policy["candidate_id"]:
        raise ValueError("Candidate config identity differs from policy")
    source_hashes = config.get("final_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Final-fit source snapshot is incomplete")
    for relative, digest in source_hashes.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts or relative.startswith("."):
            raise ValueError(f"Unsafe source snapshot name: {relative}")
        require_hash(model / "source_snapshot" / name, digest, "Source snapshot")
    if not isinstance(report.get("all_36"), list) or len(report["all_36"]) != 36:
        raise ValueError("Final-fit dataset report does not cover all 36 images")
    if (environment.get("tensorflow") != "2.21.0"
            or environment.get("numpy") != "2.4.6"
            or environment.get("tifffile") != "2026.3.3"
            or not str(environment.get("python", "")).startswith("3.11.")):
        raise ValueError("Final model environment differs from the frozen A40 environment")

    model_path = model / expected_model
    require_hash(model_path, manifest.get("model_sha256"), "Final Keras model")
    try:
        with zipfile.ZipFile(model_path) as keras:
            if not {"metadata.json", "config.json", "model.weights.h5"}.issubset(keras.namelist()):
                raise ValueError("Final Keras model lacks required members")
            bad_member = keras.testzip()
            if bad_member is not None:
                raise ValueError(f"Final Keras model has corrupt member: {bad_member}")
    except zipfile.BadZipFile as exc:
        raise ValueError("Final Keras model is not a valid ZIP archive") from exc
    require_hash(model / "training_history.csv", manifest.get("training_history_sha256"),
                 "Training history")
    with (model / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != policy["final_epoch_count"]:
        raise ValueError("Training history does not contain the locked epoch count")
    for index, row in enumerate(history):
        if (int(row["epoch"]) != index
                or not math.isfinite(float(row["loss"]))
                or not math.isfinite(float(row["soft_dice_coefficient"]))):
            raise ValueError("Training history has invalid epoch or non-finite metric")
    settings = policy["scientific_settings"]
    if (manifest.get("target") != "ld" or manifest.get("status") != "complete"
            or manifest.get("model_file") != expected_model
            or manifest.get("policy_sha256") != policy_digest
            or manifest.get("training_images") != 36
            or manifest.get("epochs_completed") != policy["final_epoch_count"]
            or manifest.get("semantic_labels") != {"background": 0, "foreground": 1}
            or manifest.get("label_rule") != "ld_antialias_midpoint_v1"
            or manifest.get("threshold") != policy["deployment_threshold"]
            or manifest.get("tile_size") != settings["patch_size"]
            or manifest.get("overlap") != settings["overlap"]
            or manifest.get("prediction_batch_size") != settings["prediction_batch_size"]
            or manifest.get("lower_percentile") != settings["lower_percentile"]
            or manifest.get("upper_percentile") != settings["upper_percentile"]
            or threshold.get("threshold") != policy["deployment_threshold"]
            or threshold.get("policy_sha256") != policy_digest):
        raise ValueError("Final model manifest or threshold differs from the locked policy")
    if method == "candidate" and (manifest.get("candidate_id") != policy["candidate_id"]
                                  or manifest.get("positive_center_policy")
                                  != settings["positive_center_policy"]):
        raise ValueError("Candidate model manifest differs from selected method")
    for key in ("final_loss", "final_soft_dice_on_training_patches"):
        if not isinstance(manifest.get(key), (int, float)) or not math.isfinite(manifest[key]):
            raise ValueError(f"Final model manifest has invalid {key}")
    return {"method": method, "model_sha256": manifest["model_sha256"],
            "policy_sha256": policy_digest, "global_policy_sha256": global_digest,
            "epochs": policy["final_epoch_count"],
            "threshold": policy["deployment_threshold"]}


def expected_archive_sha(archive: Path, sha_file: Path) -> str:
    lines = sha_file.read_text(encoding="ascii").splitlines()
    match = re.fullmatch(r"([0-9a-f]{64})  ld_final_model\.tar\.gz", lines[0]) if len(lines) == 1 else None
    if match is None or archive.name != ARCHIVE_NAME:
        raise ValueError("Expected a strict SHA256 record for ld_final_model.tar.gz")
    expected = match.group(1)
    require_hash(archive, expected, "Final archive")
    return expected


def _safe_extract(archive: Path, destination: Path) -> None:
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = member.name
            path = Path(name)
            if (not path.parts or name in seen or path.is_absolute()
                    or ".." in path.parts or "outer_test" in path.parts
                    or "\\" in name or any(part.startswith("._") for part in path.parts)
                    or path.parts[0] not in {"model", "logs"}
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsafe or unexpected final archive member: {name}")
            seen.add(name)
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable final archive member: {name}")
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)


def verify_archive(archive: Path, sha_file: Path, destination: Path, method: str,
                   policy_path: Path, fold_map: Path, label_manifest: Path,
                   global_policy_path: Path) -> dict[str, object]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verified LD final run: {destination}")
    digest = expected_archive_sha(archive, sha_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".ld_final_verify_", dir=destination.parent))
    try:
        _safe_extract(archive, scratch)
        result = verify_tree(scratch, method, policy_path, fold_map,
                             label_manifest, global_policy_path)
        artifact_dir = scratch / "archives"
        artifact_dir.mkdir()
        shutil.copyfile(archive, artifact_dir / ARCHIVE_NAME)
        shutil.copyfile(sha_file, artifact_dir / SHA_NAME)
        (scratch / "verification.json").write_text(
            json.dumps({**result, "archive_sha256": digest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite verified LD final run: {destination}")
        os.rename(scratch, destination)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return {**result, "archive_sha256": digest}


def fetch_archive(host: str, port: int, identity: Path, destination: Path,
                  method: str, policy_path: Path, fold_map: Path,
                  label_manifest: Path, global_policy_path: Path) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH host or port")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verified LD final run: {destination}")
    if not identity.is_file():
        raise FileNotFoundError(f"SSH identity file is missing: {identity}")
    remote = "/workspace/ld_final_20261012"
    options = ["-i", str(identity), "-o", "BatchMode=yes", "-o",
               "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    ssh = ["ssh", *options, "-p", str(port), f"root@{host}"]
    query = (f"test -f {remote}/archives/{READY_NAME} && "
             f"test \"$(cat {remote}/logs/runner_exit_status.txt)\" = 0")
    subprocess.run([*ssh, query], check=True, timeout=35)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ld_final_fetch_", dir=destination.parent) as folder:
        temporary = Path(folder)
        scp = ["scp", *options, "-P", str(port)]
        for name in (SHA_NAME, ARCHIVE_NAME):
            subprocess.run([*scp, f"root@{host}:{remote}/archives/{name}",
                            str(temporary / name)], check=True, timeout=1800)
        return verify_archive(temporary / ARCHIVE_NAME, temporary / SHA_NAME,
                              destination, method, policy_path, fold_map,
                              label_manifest, global_policy_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check-policy", "check-locked-baseline", "check-selection",
                 "verify-tree", "verify", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("--method", choices=tuple(MODEL_NAMES), required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--fold-map", type=Path, required=True)
        command.add_argument("--label-manifest", type=Path, required=True)
        if name != "check-policy":
            command.add_argument("--global-policy", type=Path, required=True)
        if name == "check-selection":
            command.add_argument("--baseline-cv-root", type=Path, required=True)
            command.add_argument("--candidate-cv-root", type=Path, required=True)
            command.add_argument("--label-dir", type=Path, required=True)
            command.add_argument("--study-root", type=Path, required=True)
        elif name == "verify-tree":
            command.add_argument("--tree", type=Path, required=True)
        elif name == "verify":
            command.add_argument("--archive", type=Path, required=True)
            command.add_argument("--sha-file", type=Path, required=True)
            command.add_argument("--destination", type=Path, required=True)
        elif name == "fetch":
            command.add_argument("--host", required=True)
            command.add_argument("--port", type=int, required=True)
            command.add_argument("--identity", type=Path, required=True)
            command.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-policy":
        result = validate_policy(args.method, args.policy, args.fold_map, args.label_manifest)
        print(f"LD_FINAL_POLICY_PASS method={args.method} epochs={result['final_epoch_count']}")
    elif args.command == "check-locked-baseline":
        if args.method != "baseline":
            raise ValueError("The locked-policy preflight supports baseline only")
        result = validate_locked_baseline(
            args.policy, args.global_policy, args.fold_map, args.label_manifest,
        )
        print(f"LD_GLOBAL_METHOD_LOCK_PASS method=baseline "
              f"final_policy_sha256={result['final_policy_sha256']} "
              f"global_policy_sha256={result['global_policy_sha256']}")
    elif args.command == "check-selection":
        result = validate_global_selection(
            args.method, args.policy, args.global_policy,
            args.baseline_cv_root, args.candidate_cv_root,
            args.fold_map, args.label_manifest, args.label_dir, args.study_root,
        )
        print(f"LD_GLOBAL_METHOD_SELECTION_PASS method={result['method']} "
              f"global_policy_sha256={result['global_policy_sha256']}")
    elif args.command == "verify-tree":
        result = verify_tree(args.tree, args.method, args.policy, args.fold_map,
                             args.label_manifest, args.global_policy)
        print(f"LD_FINAL_TREE_PASS model_sha256={result['model_sha256']}")
    else:
        if args.command == "verify":
            result = verify_archive(args.archive, args.sha_file, args.destination,
                                    args.method, args.policy, args.fold_map,
                                    args.label_manifest, args.global_policy)
        else:
            result = fetch_archive(args.host, args.port, args.identity,
                                   args.destination, args.method, args.policy,
                                   args.fold_map, args.label_manifest,
                                   args.global_policy)
        print(f"LD_FINAL_ARCHIVE_VERIFIED method={result['method']} "
              f"archive_sha256={result['archive_sha256']} "
              f"model_sha256={result['model_sha256']}")


if __name__ == "__main__":
    main()
