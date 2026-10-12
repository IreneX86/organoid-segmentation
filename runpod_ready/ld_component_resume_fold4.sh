#!/usr/bin/env bash
# Resume the frozen LD component experiment after folds 0–3 were completed,
# downloaded, and restored. This entry point intentionally cannot train them.
# Restore the original /workspace/ld_study, baseline, and candidate run paths;
# do not alter the frozen source, split, labels, or validation decisions.
set -euo pipefail

STUDY_ROOT=${LD_STUDY_ROOT:-/workspace/ld_study}
BASELINE_ROOT=${LD_BASELINE_ROOT:-/workspace/ld_baseline_20261012/full}
RUN_ROOT=${LD_COMPONENT_RUN_ROOT:?Set the restored original LD_COMPONENT_RUN_ROOT}
FULL_ROOT="$RUN_ROOT/full"
ARCHIVES="$RUN_ROOT/archives"

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH LD_STUDY_ROOT="$STUDY_ROOT" TF_FORCE_GPU_ALLOW_GROWTH=true

# Archive manifests from the original run name absolute paths. Requiring the
# same path also preserves paths inside the locked experiment configurations.
if [[ "$STUDY_ROOT" != /workspace/ld_study \
      || "$BASELINE_ROOT" != /workspace/ld_baseline_20261012/full \
      || "$RUN_ROOT" != /workspace/ld_component_20261012 ]]; then
    echo "Resume requires the original absolute study, baseline, and run paths" >&2
    exit 1
fi
if [[ ! -d "$FULL_ROOT/ld" || ! -d "$ARCHIVES" ]]; then
    echo "Original candidate run or archives were not restored" >&2
    exit 1
fi
if [[ -e "$FULL_ROOT/ld/fold_4" || -e "$RUN_ROOT/fold_4_start_utc.txt" \
      || -e "$RUN_ROOT/fold_4_train.log" || -e "$ARCHIVES/fold_4.tar.gz" \
      || -e "$ARCHIVES/fold_4.sha256" || -e "$ARCHIVES/fold_4.ready" \
      || -e "$RUN_ROOT/gated_cv_report" \
      || -e "$RUN_ROOT/resume_fold4.sh" \
      || -e "$RUN_ROOT/resume_fold4.lock" \
      || -e "$RUN_ROOT/resume_fold4_start_utc.txt" \
      || -e "$ARCHIVES/final_metadata.tar.gz" \
      || -e "$ARCHIVES/final_metadata.ready" ]]; then
    echo "Fold 4 or the final report has already started; refusing to repeat work" >&2
    exit 1
fi
if [[ -f "$RUN_ROOT/runner_exit_status.txt" \
      && "$(cat "$RUN_ROOT/runner_exit_status.txt")" == 0 ]]; then
    echo "Original runner already recorded success; refusing resume" >&2
    exit 1
fi

# The original runner and preflight records must be present. They are included
# unchanged in the final archive so the earlier Pod's provenance is retained.
for file in runner.sh start_utc.txt hostname.txt nvidia_smi_list.txt \
            gpu_preflight.log gpu_usage.csv candidate_unit_tests.log \
            sampler_batch_preflight.log; do
    if [[ ! -s "$RUN_ROOT/$file" ]]; then
        echo "Missing original run record: $file" >&2
        exit 1
    fi
done
cmp -s "$RUN_ROOT/runner.sh" "$STUDY_ROOT/runpod_ready/ld_component_five_fold.sh" || {
    echo "Restored original runner differs from the frozen study source" >&2
    exit 1
}

# This validation never opens fold 4's held-out label bytes. Earlier gates are
# checked against their immutable archive/lock hashes without recomputing them
# from labels. Fold 4's training/validation labels alone are hash-checked.
# Archive SHA, every restored archive member, source/data hashes, checkpoints,
# and immutable gate decisions are verified before any GPU work begins.
LD_COMPONENT_RESUME_RUN_ROOT="$RUN_ROOT" \
LD_COMPONENT_RESUME_BASELINE_ROOT="$BASELINE_ROOT" \
LD_COMPONENT_RESUME_STUDY_ROOT="$STUDY_ROOT" \
PYTHONPATH="$STUDY_ROOT/code" python - <<'PY'
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path

from ld_component_gate import UNCHANGED_SCIENCE_KEYS
from ld_evaluate import _validate_locked_run

run = Path(os.environ["LD_COMPONENT_RESUME_RUN_ROOT"])
study = Path(os.environ["LD_COMPONENT_RESUME_STUDY_ROOT"])
baseline = Path(os.environ["LD_COMPONENT_RESUME_BASELINE_ROOT"])
archives = run / "archives"
candidate = run / "full" / "ld"
fold_map = study / "fold_map.json"
manifest = study / "manifest.json"
labels = study / "LD_output_binary"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


frozen = json.loads(fold_map.read_text())
derived = json.loads(manifest.read_text())
images = frozen.get("input_sha256")
fold_ids = frozen.get("fold_ids")
records = {item["filename"]: item for item in derived.get("files", [])}
if (frozen.get("target") != "ld" or frozen.get("num_folds") != 5
        or not isinstance(images, dict) or len(images) != 36
        or not isinstance(fold_ids, dict) or set(fold_ids) != set(images)
        or set(images) != set(records) or len(records) != 36
        or derived.get("rule_id") != "ld_antialias_midpoint_v1"):
    raise ValueError("Frozen LD input/derived-label manifests are incomplete")
for name in sorted(images):
    if digest(study / "LD_input" / name) != images[name]:
        raise ValueError(f"Restored LD input differs: {name}")
    if fold_ids[name] != 4 and digest(labels / name) != records[name]["derived_mask_sha256"]:
        raise ValueError(f"Restored fold-4 training/validation label differs: {name}")
if {fold_ids[name] for name in fold_ids} != set(range(5)):
    raise ValueError("Frozen LD map does not contain all five folds")

baseline_records = {}
for fold in range(5):
    base = baseline / "ld" / f"fold_{fold}"
    if not (base / "selection.lock.json").is_file() or not (
        base / "outer_test" / "evaluation_manifest.json"
    ).is_file():
        raise ValueError(f"Frozen baseline fold {fold} is incomplete")
    baseline_records[fold] = _validate_locked_run(base, fold_map, manifest)

for fold in range(4):
    name = f"fold_{fold}"
    archive = archives / f"{name}.tar.gz"
    sha_file = archives / f"{name}.sha256"
    pieces = sha_file.read_text(encoding="ascii").strip().split()
    if (len(pieces) != 2 or not re.fullmatch(r"[0-9a-f]{64}", pieces[0])
            or pieces[1] != str(archive) or digest(archive) != pieces[0]):
        raise ValueError(f"Fold {fold} archive SHA256 is invalid")
    expected_files = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = Path(member.name)
            if (relative.is_absolute() or ".." in relative.parts
                    or member.issym() or member.islnk() or member.isdev()):
                raise ValueError(f"Unsafe fold {fold} archive member: {member.name}")
            if not member.isfile():
                if not member.isdir():
                    raise ValueError(f"Unexpected fold {fold} archive member")
                continue
            if relative.parts[0] == name:
                restored = candidate / relative
            elif relative.name == relative.as_posix() and relative.name.startswith(f"{name}_"):
                restored = run / relative
            else:
                raise ValueError(f"Unexpected fold {fold} archive member: {member.name}")
            source = bundle.extractfile(member)
            if source is None or not restored.is_file():
                raise ValueError(f"Fold {fold} archive member was not restored: {member.name}")
            value = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                value.update(block)
            if value.hexdigest() != digest(restored):
                raise ValueError(f"Restored fold {fold} member differs: {member.name}")
            expected_files.add(member.name)
    required = {
        f"{name}/selection.lock.json", f"{name}/decision.lock.json",
        f"{name}/component_unet.keras", f"{name}/training_history.csv",
        f"{name}_train.log", f"{name}_gate.log", f"{name}_evaluate.log",
        f"{name}_start_utc.txt", f"{name}_end_utc.txt",
    }
    if not required <= expected_files:
        raise ValueError(f"Fold {fold} archive lacks a required checkpoint/lock/log")
    fold_dir = candidate / name
    base_dir = baseline / "ld" / name
    config = json.loads((fold_dir / "experiment_config.json").read_text())
    base_config, base_split, _ = baseline_records[fold]
    split = json.loads((fold_dir / "split.json").read_text())
    lock = json.loads((fold_dir / "selection.lock.json").read_text())
    recorded = json.loads((fold_dir / "decision.lock.json").read_text())
    code = config.get("code_sha256")
    expected_partitions = {
        "test": {name for name, assigned in fold_ids.items() if assigned == fold},
        "validation": {name for name, assigned in fold_ids.items()
                       if assigned == (fold + 1) % 5},
        "train": {name for name, assigned in fold_ids.items()
                  if assigned not in (fold, (fold + 1) % 5)},
    }
    if (config.get("target") != "ld" or config.get("fold") != fold
            or config.get("candidate") != "component_uniform_positive_center_v1"
            or config.get("positive_center_policy")
            != "uniform_8_connected_training_component_then_pixel"
            or config.get("positive_patch_probability") != 0.6
            or split.get("fold_ids") != fold_ids or split.get("fold") != fold
            or split != base_split
            or any(set(split.get(part, ())) != names
                   for part, names in expected_partitions.items())
            or any(config.get(key) != base_config.get(key)
                   for key in UNCHANGED_SCIENCE_KEYS)
            or config.get("input_dir") != str(study / "LD_input")
            or config.get("label_dir") != str(labels)
            or base_config.get("input_dir") != config.get("input_dir")
            or base_config.get("label_dir") != config.get("label_dir")
            or not isinstance(code, dict) or not code
            or lock.get("code_sha256") != code):
        raise ValueError(f"Fold {fold} science/split/source identity differs")
    for relative, expected_digest in code.items():
        source_name = Path(relative)
        if (source_name.is_absolute() or ".." in source_name.parts
                or digest(study / source_name) != expected_digest
                or digest(fold_dir / "source_snapshot" / source_name)
                != expected_digest):
            raise ValueError(f"Fold {fold} source differs: {relative}")
    expected_locked_hashes = {
        "fold_map_sha256": digest(fold_map),
        "label_manifest_sha256": digest(manifest),
        "checkpoint_sha256": digest(fold_dir / "component_unet.keras"),
        "threshold_sha256": digest(fold_dir / "threshold.json"),
        "experiment_config_sha256": digest(fold_dir / "experiment_config.json"),
        "split_sha256": digest(fold_dir / "split.json"),
        "validation_artifacts_manifest_sha256": digest(
            fold_dir / "validation_artifacts" / "manifest.json"),
        "training_component_counts_sha256": digest(
            fold_dir / "training_component_counts.json"),
    }
    if (any(lock.get(field) != expected_digest
            for field, expected_digest in expected_locked_hashes.items())
            or config.get("fold_map_sha256") != expected_locked_hashes["fold_map_sha256"]
            or config.get("label_manifest_sha256")
            != expected_locked_hashes["label_manifest_sha256"]
            or recorded.get("fold") != fold
            or recorded.get("candidate_selection_lock_sha256")
            != digest(fold_dir / "selection.lock.json")
            or recorded.get("baseline_selection_lock_sha256")
            != digest(base_dir / "selection.lock.json")
            or recorded.get("fold_map_sha256") != digest(fold_map)
            or recorded.get("label_manifest_sha256") != digest(manifest)
            or recorded.get("outer_test_labels_opened") is not False
            or set(recorded.get("validation_filenames", ()))
            != expected_partitions["validation"]):
        raise ValueError(f"Fold {fold} selection/decision lock changed")
    checks = recorded.get("gate_checks")
    if (not isinstance(checks, dict) or not checks
            or any(type(value) is not bool for value in checks.values())
            or recorded.get("selected")
            != ("candidate" if all(checks.values()) else "baseline")):
        raise ValueError(f"Fold {fold} gate decision is inconsistent")
    if recorded["selected"] == "candidate":
        if not (candidate / name / "outer_test" / "evaluation_manifest.json").is_file():
            raise ValueError(f"Selected candidate fold {fold} lacks evaluation")
    elif recorded["selected"] == "baseline":
        if (candidate / name / "outer_test").exists():
            raise ValueError(f"Baseline-selected fold {fold} opened candidate outer test")
    else:
        raise ValueError(f"Fold {fold} decision has unknown selection")
    print(f"RESTORED_FOLD_{fold}_VERIFIED selected={recorded['selected']}", flush=True)

print("LD_COMPONENT_RESUME_PREREQUISITES_PASS", flush=True)
PY

# An atomic lock prevents another invocation from racing into the same fold.
# Keep it on failure: a partly started fold must be diagnosed, never retried
# automatically or silently overwritten.
mkdir "$RUN_ROOT/resume_fold4.lock"
cp "$STUDY_ROOT/runpod_ready/ld_component_resume_fold4.sh" \
    "$RUN_ROOT/resume_fold4.sh"
sha256sum "$RUN_ROOT/resume_fold4.sh" > "$RUN_ROOT/resume_fold4_script.sha256"
monitor_pid=
finish() {
    status=$?
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ "$status" -ne 0 || ! -f "$RUN_ROOT/resume_fold4_exit_status.txt" ]]; then
        printf '%s\n' "$status" > "$RUN_ROOT/resume_fold4_exit_status.txt"
        date -u +%FT%TZ > "$RUN_ROOT/resume_fold4_end_utc.txt"
    fi
}
trap finish EXIT
date -u +%FT%TZ > "$RUN_ROOT/resume_fold4_start_utc.txt"
hostname > "$RUN_ROOT/resume_fold4_hostname.txt"
nvidia-smi -L > "$RUN_ROOT/resume_fold4_nvidia_smi_list.txt"
if [[ "$(nvidia-smi --list-gpus | wc -l)" -ne 1 ]]; then
    echo "Expected one Pod-visible NVIDIA GPU" >&2
    exit 1
fi
python "$STUDY_ROOT/runpod_ready/ld_gpu_preflight.py" \
    > "$RUN_ROOT/resume_fold4_gpu_preflight.log" 2>&1
grep -qx LD_GPU_PREFLIGHT_PASS "$RUN_ROOT/resume_fold4_gpu_preflight.log"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 10 > "$RUN_ROOT/resume_fold4_gpu_usage.csv" &
monitor_pid=$!

# Keep these scientific commands byte-for-byte equivalent to the original
# runpod_ready/ld_component_five_fold.sh body for a single fold.
fold=4
fold_dir="$FULL_ROOT/ld/fold_$fold"
date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_start_utc.txt"
timeout --signal=TERM --kill-after=60s 80m \
    python "$STUDY_ROOT/code/ld_component_train.py" \
        --input-dir "$STUDY_ROOT/LD_input" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        --output-root "$FULL_ROOT" --fold "$fold" \
    > "$RUN_ROOT/fold_${fold}_train.log" 2>&1
python "$STUDY_ROOT/code/ld_component_gate.py" \
    --baseline-run-dir "$BASELINE_ROOT/ld/fold_$fold" \
    --candidate-run-dir "$fold_dir" \
    --fold-map "$STUDY_ROOT/fold_map.json" \
    --label-manifest "$STUDY_ROOT/manifest.json" \
    --label-dir "$STUDY_ROOT/LD_output_binary" \
    > "$RUN_ROOT/fold_${fold}_gate.log" 2>&1
python "$STUDY_ROOT/code/ld_component_evaluate.py" \
    --baseline-run-dir "$BASELINE_ROOT/ld/fold_$fold" \
    --candidate-run-dir "$fold_dir" \
    --input-dir "$STUDY_ROOT/LD_input" \
    --label-dir "$STUDY_ROOT/LD_output_binary" \
    --label-manifest "$STUDY_ROOT/manifest.json" \
    --fold-map "$STUDY_ROOT/fold_map.json" \
    > "$RUN_ROOT/fold_${fold}_evaluate.log" 2>&1
date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_end_utc.txt"

tar -C "$FULL_ROOT/ld" -czf "$ARCHIVES/fold_${fold}.tar.gz" "fold_$fold" \
    -C "$RUN_ROOT" "fold_${fold}_train.log" "fold_${fold}_gate.log" \
    "fold_${fold}_evaluate.log" "fold_${fold}_start_utc.txt" \
    "fold_${fold}_end_utc.txt"
sha256sum "$ARCHIVES/fold_${fold}.tar.gz" > "$ARCHIVES/fold_${fold}.sha256"
touch "$ARCHIVES/fold_${fold}.ready"
echo "LD_COMPONENT_FOLD_${fold}_COMPLETE"

python "$STUDY_ROOT/code/ld_component_report.py" \
    --baseline-run-root "$BASELINE_ROOT" \
    --candidate-run-root "$FULL_ROOT" \
    --fold-map "$STUDY_ROOT/fold_map.json" \
    --label-manifest "$STUDY_ROOT/manifest.json" \
    --label-dir "$STUDY_ROOT/LD_output_binary" \
    --output-dir "$RUN_ROOT/gated_cv_report" \
    > "$RUN_ROOT/gated_cv_report.log" 2>&1
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

# Preserve the original interrupted runner's markers before writing the
# completed five-fold status expected by the existing archive verifier.
for file in runner_exit_status.txt end_utc.txt; do
    if [[ -e "$RUN_ROOT/$file" ]]; then
        if [[ -e "$RUN_ROOT/pause_original_$file" ]]; then
            echo "Original pause status was already copied: $file" >&2
            exit 1
        fi
        cp -p "$RUN_ROOT/$file" "$RUN_ROOT/pause_original_$file"
    fi
done
printf '0\n' > "$RUN_ROOT/runner_exit_status.txt"
printf '0\n' > "$RUN_ROOT/resume_fold4_exit_status.txt"
date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
cp "$RUN_ROOT/end_utc.txt" "$RUN_ROOT/resume_fold4_end_utc.txt"
pause_records=()
for file in pause_original_runner_exit_status.txt pause_original_end_utc.txt; do
    if [[ -f "$RUN_ROOT/$file" ]]; then
        pause_records+=("$file")
    fi
done
tar -C "$RUN_ROOT" -czf "$ARCHIVES/final_metadata.tar.gz" \
    start_utc.txt end_utc.txt runner_exit_status.txt hostname.txt runner.sh \
    nvidia_smi_list.txt gpu_preflight.log \
    gpu_usage.csv \
    candidate_unit_tests.log sampler_batch_preflight.log \
    gated_cv_report gated_cv_report.log \
    fold_0_start_utc.txt fold_0_end_utc.txt \
    fold_1_start_utc.txt fold_1_end_utc.txt \
    fold_2_start_utc.txt fold_2_end_utc.txt \
    fold_3_start_utc.txt fold_3_end_utc.txt \
    fold_4_start_utc.txt fold_4_end_utc.txt \
    resume_fold4_start_utc.txt resume_fold4_end_utc.txt \
    resume_fold4_exit_status.txt resume_fold4_hostname.txt \
    resume_fold4_nvidia_smi_list.txt resume_fold4_gpu_preflight.log \
    resume_fold4_gpu_usage.csv resume_fold4.sh resume_fold4_script.sha256 \
    "${pause_records[@]}"
sha256sum "$ARCHIVES/final_metadata.tar.gz" > "$ARCHIVES/final_metadata.sha256"
echo LD_COMPONENT_FIVE_FOLD_COMPLETE
touch "$ARCHIVES/final_metadata.ready"
