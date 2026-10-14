#!/usr/bin/env bash
# Prospectively locked LD focal-BCE plus soft-Dice loss experiment on one RunPod GPU.
# All five validation selections and five strictly fold-local validation-only
# method decisions are locked before this script opens any outer-test mask.
set -euo pipefail

STUDY_ROOT=${LD_STUDY_ROOT:-/workspace/ld_study}
BASELINE_REFERENCE_DIR=${LD_BASELINE_REFERENCE_DIR:-$STUDY_ROOT/baseline_validation_reference}
RUN_ROOT=${LD_FOCAL_RUN_ROOT:?Set a fresh LD_FOCAL_RUN_ROOT}
FULL_ROOT="$RUN_ROOT/full"
ARCHIVES="$RUN_ROOT/archives"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing LD focal run: $RUN_ROOT" >&2
    exit 1
fi
for fold in 0 1 2 3 4; do
    if [[ ! -s "$BASELINE_REFERENCE_DIR/fold_$fold.json" ]]; then
        echo "Missing immutable same-fold baseline validation reference $fold" >&2
        exit 1
    fi
done
for path in \
    "$STUDY_ROOT/LD_input" "$STUDY_ROOT/LD_output_binary" \
    "$STUDY_ROOT/manifest.json" "$STUDY_ROOT/fold_map.json" \
    "$STUDY_ROOT/code/ld_focal_train.py" "$STUDY_ROOT/code/ld_focal_evaluate.py" \
    "$STUDY_ROOT/code/ld_focal_gate.py" \
    "$STUDY_ROOT/code/ld_focal_cv_report.py" \
    "$STUDY_ROOT/runpod_ready/ld_focal_five_fold.sh" \
    "$STUDY_ROOT/runpod_ready/ld_gpu_preflight.py" \
    "$STUDY_ROOT/experiments/ld_focal_dice_protocol.md"; do
    if [[ ! -e "$path" ]]; then
        echo "Required frozen input or experiment source is missing: $path" >&2
        exit 1
    fi
done

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
# Reuse the proven A40 TensorFlow 2.21 pip CUDA/cuDNN library discovery.
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH LD_STUDY_ROOT="$STUDY_ROOT" TF_FORCE_GPU_ALLOW_GROWTH=true

mkdir -p "$FULL_ROOT" "$ARCHIVES"
cp "$STUDY_ROOT/runpod_ready/ld_focal_five_fold.sh" "$RUN_ROOT/runner.sh"
cp "$STUDY_ROOT/experiments/ld_focal_dice_protocol.md" "$RUN_ROOT/protocol.md"
cp "$STUDY_ROOT/code/ld_focal_gate.py" "$RUN_ROOT/ld_focal_gate.py"
(cd "$RUN_ROOT" && sha256sum runner.sh protocol.md ld_focal_gate.py) \
    > "$RUN_ROOT/source_sha256.txt"
(cd "$BASELINE_REFERENCE_DIR" && \
    sha256sum fold_0.json fold_1.json fold_2.json fold_3.json fold_4.json) \
    > "$RUN_ROOT/baseline_reference_sha256.txt"

monitor_pid=
on_exit() {
    status=$?
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$RUN_ROOT/runner_exit_status.txt"
    date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
}
trap on_exit EXIT

# Digest files use relative archive names so the downloaded archive can be
# verified locally with `sha256sum -c` from its own directory.
publish_archive() {
    local name=$1
    shift
    local partial="$ARCHIVES/$name.tar.gz.partial"
    tar -czf "$partial" "$@"
    mv "$partial" "$ARCHIVES/$name.tar.gz"
    (cd "$ARCHIVES" && sha256sum "$name.tar.gz" > "$name.sha256")
    touch "$ARCHIVES/$name.ready"
}

date -u +%FT%TZ > "$RUN_ROOT/start_utc.txt"
hostname > "$RUN_ROOT/hostname.txt"
python --version > "$RUN_ROOT/python_version.txt" 2>&1
command -v python > "$RUN_ROOT/python_executable.txt"
python -m pip list --format=freeze > "$RUN_ROOT/pip_packages.txt"
nvidia-smi > "$RUN_ROOT/nvidia_smi_start.txt"
nvidia-smi -L > "$RUN_ROOT/nvidia_smi_list.txt"
if [[ "$(nvidia-smi --list-gpus | wc -l)" -ne 1 ]]; then
    echo "Expected precisely one Pod-visible NVIDIA GPU" >&2
    exit 1
fi
python "$STUDY_ROOT/runpod_ready/ld_gpu_preflight.py" \
    > "$RUN_ROOT/gpu_preflight.log" 2>&1
grep -qx LD_GPU_PREFLIGHT_PASS "$RUN_ROOT/gpu_preflight.log"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 10 > "$RUN_ROOT/gpu_usage.csv" &
monitor_pid=$!

# Phase 1: fitting may read only each fold's train and inner-validation masks.
# No outer-test evaluation occurs in this loop. Every completed checkpoint and
# validation selection is published in a separately checksummed fold archive.
for fold in 0 1 2 3 4; do
    fold_dir="$FULL_ROOT/ld/fold_$fold"
    if [[ -e "$fold_dir" ]]; then
        echo "Refusing to overwrite candidate fold $fold" >&2
        exit 1
    fi
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_start_utc.txt"
    timeout --signal=TERM --kill-after=60s 80m \
        python "$STUDY_ROOT/code/ld_focal_train.py" \
            --input-dir "$STUDY_ROOT/LD_input" \
            --label-dir "$STUDY_ROOT/LD_output_binary" \
            --label-manifest "$STUDY_ROOT/manifest.json" \
            --fold-map "$STUDY_ROOT/fold_map.json" \
            --output-root "$FULL_ROOT" --fold "$fold" \
        > "$RUN_ROOT/fold_${fold}_train.log" 2>&1
    if [[ ! -s "$fold_dir/selection.lock.json" \
          || ! -s "$fold_dir/validation_artifacts/manifest.json" \
          || ! -s "$fold_dir/focal_unet.keras" \
          || -e "$fold_dir/outer_test" ]]; then
        echo "Fold $fold did not finish its validation-only selection cleanly" >&2
        exit 1
    fi
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_validation_end_utc.txt"
    publish_archive "fold_${fold}_training" \
        -C "$FULL_ROOT/ld" "fold_$fold" \
        -C "$RUN_ROOT" "fold_${fold}_train.log" \
        "fold_${fold}_start_utc.txt" "fold_${fold}_validation_end_utc.txt"
    echo "LD_FOCAL_FOLD_${fold}_SELECTION_LOCKED"
done

# Phase 2: each gate reads only this fold's inner-validation data and the
# matching compact frozen-baseline reference. Lock all five independent gate
# decisions before any candidate outer-test metric is computed. Validation
# sets rotate, so pooling them to choose a given fold would leak that fold's
# held-out images through another fold's validation set.
for fold in 0 1 2 3 4; do
    fold_dir="$FULL_ROOT/ld/fold_$fold"
    if [[ ! -s "$fold_dir/selection.lock.json" \
          || ! -s "$fold_dir/validation_artifacts/manifest.json" \
          || -e "$fold_dir/outer_test" ]]; then
        echo "Fold-local gate blocked: incomplete or already evaluated fold $fold" >&2
        exit 1
    fi
    python "$STUDY_ROOT/code/ld_focal_gate.py" compare \
        --baseline-reference "$BASELINE_REFERENCE_DIR/fold_$fold.json" \
        --candidate-run-dir "$fold_dir" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        --output-file "$fold_dir/decision.lock.json" \
        > "$RUN_ROOT/fold_${fold}_gate.log" 2>&1
    if [[ ! -s "$fold_dir/decision.lock.json" ]]; then
        echo "Fold $fold validation-only decision was not locked" >&2
        exit 1
    fi
    publish_archive "fold_${fold}_gate" \
        -C "$FULL_ROOT/ld" "fold_$fold/decision.lock.json" \
        -C "$RUN_ROOT" "fold_${fold}_gate.log"
    echo "LD_FOCAL_FOLD_${fold}_VALIDATION_GATE_LOCKED"
done

# Phase 3: one final, descriptive outer-test evaluation per locked fold.
# Even a rejected candidate is scored once for exploratory paired analysis;
# these results must never revise any fold-local decision lock.
for fold in 0 1 2 3 4; do
    fold_dir="$FULL_ROOT/ld/fold_$fold"
    if [[ ! -s "$fold_dir/decision.lock.json" \
          || ! -s "$fold_dir/selection.lock.json" \
          || -e "$fold_dir/outer_test" ]]; then
        echo "Outer-test evaluation blocked for fold $fold" >&2
        exit 1
    fi
    python "$STUDY_ROOT/code/ld_focal_evaluate.py" \
        --run-dir "$fold_dir" \
        --all-runs-root "$FULL_ROOT" \
        --input-dir "$STUDY_ROOT/LD_input" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        > "$RUN_ROOT/fold_${fold}_evaluate.log" 2>&1
    if [[ ! -s "$fold_dir/outer_test/evaluation_manifest.json" ]]; then
        echo "Fold $fold outer-test evaluation was incomplete" >&2
        exit 1
    fi
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_end_utc.txt"
    publish_archive "fold_${fold}_evaluation" \
        -C "$FULL_ROOT/ld" "fold_$fold/outer_test" \
        -C "$RUN_ROOT" "fold_${fold}_evaluate.log" "fold_${fold}_end_utc.txt"
    echo "LD_FOCAL_FOLD_${fold}_EVALUATED"
done

python "$STUDY_ROOT/code/ld_focal_cv_report.py" \
    --run-root "$FULL_ROOT" --output-dir "$RUN_ROOT/cv_report" \
    > "$RUN_ROOT/cv_report.log" 2>&1
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
printf '0\n' > "$RUN_ROOT/runner_exit_status.txt"
date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
publish_archive final_metadata \
    -C "$RUN_ROOT" \
    runner.sh protocol.md ld_focal_gate.py source_sha256.txt \
    baseline_reference_sha256.txt \
    start_utc.txt end_utc.txt runner_exit_status.txt hostname.txt \
    python_version.txt python_executable.txt pip_packages.txt \
    nvidia_smi_start.txt nvidia_smi_list.txt gpu_preflight.log gpu_usage.csv \
    cv_report cv_report.log \
    fold_0_gate.log fold_1_gate.log fold_2_gate.log \
    fold_3_gate.log fold_4_gate.log \
    fold_0_start_utc.txt fold_0_validation_end_utc.txt fold_0_end_utc.txt \
    fold_1_start_utc.txt fold_1_validation_end_utc.txt fold_1_end_utc.txt \
    fold_2_start_utc.txt fold_2_validation_end_utc.txt fold_2_end_utc.txt \
    fold_3_start_utc.txt fold_3_validation_end_utc.txt fold_3_end_utc.txt \
    fold_4_start_utc.txt fold_4_validation_end_utc.txt fold_4_end_utc.txt \
    -C "$FULL_ROOT/ld" \
    fold_0/decision.lock.json fold_1/decision.lock.json \
    fold_2/decision.lock.json fold_3/decision.lock.json \
    fold_4/decision.lock.json \
    -C "$BASELINE_REFERENCE_DIR" \
    fold_0.json fold_1.json fold_2.json fold_3.json fold_4.json
echo LD_FOCAL_FIVE_FOLD_COMPLETE
