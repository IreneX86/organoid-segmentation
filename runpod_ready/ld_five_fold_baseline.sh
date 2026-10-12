#!/usr/bin/env bash
# Run the predeclared LD baseline on one RunPod GPU. Outer-test evaluation is
# performed only after each fold's validation-only selection lock is written.
set -euo pipefail

STUDY_ROOT=/workspace/ld_study
RUN_ROOT=/workspace/ld_baseline_20261012
FULL_ROOT="$RUN_ROOT/full"
ARCHIVES="$RUN_ROOT/archives"
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod

# TensorFlow 2.21's pip CUDA/cuDNN libraries are present in this Conda env.
# Exposing their lib directories matches the successful Mito A40 environment.
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH LD_STUDY_ROOT="$STUDY_ROOT" TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing LD run: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$FULL_ROOT" "$ARCHIVES"
monitor_pid=
finish() {
    status=$?
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$RUN_ROOT/runner_exit_status.txt"
    date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
}
trap finish EXIT
date -u +%FT%TZ > "$RUN_ROOT/start_utc.txt"
hostname > "$RUN_ROOT/hostname.txt"
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

# A one-step/one-epoch fit checks that model compilation, labels, training and
# validation run end to end. It is separate from all five formal folds and is
# never evaluated on outer-test labels or included in OOF results.
python "$STUDY_ROOT/code/ld_train.py" \
    --input-dir "$STUDY_ROOT/LD_input" \
    --label-dir "$STUDY_ROOT/LD_output_binary" \
    --label-manifest "$STUDY_ROOT/manifest.json" \
    --fold-map "$STUDY_ROOT/fold_map.json" \
    --output-root "$RUN_ROOT/smoke" --fold 0 --steps-per-epoch 1 --epochs 1 \
    > "$RUN_ROOT/fit_smoke.log" 2>&1
echo LD_FIT_SMOKE_PASS

for fold in 0 1 2 3 4; do
    fold_dir="$FULL_ROOT/ld/fold_$fold"
    if [[ -e "$fold_dir" ]]; then
        echo "Refusing to overwrite fold $fold" >&2
        exit 1
    fi
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_start_utc.txt"
    timeout --signal=TERM --kill-after=60s 80m \
        python "$STUDY_ROOT/code/ld_train.py" \
            --input-dir "$STUDY_ROOT/LD_input" \
            --label-dir "$STUDY_ROOT/LD_output_binary" \
            --label-manifest "$STUDY_ROOT/manifest.json" \
            --fold-map "$STUDY_ROOT/fold_map.json" \
            --output-root "$FULL_ROOT" --fold "$fold" \
        > "$RUN_ROOT/fold_${fold}_train.log" 2>&1
    python "$STUDY_ROOT/code/ld_evaluate.py" \
        --run-dir "$fold_dir" \
        --input-dir "$STUDY_ROOT/LD_input" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        > "$RUN_ROOT/fold_${fold}_evaluate.log" 2>&1
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_end_utc.txt"

    # One complete archive per fold, with logs and SHA256, can be downloaded
    # while later folds train. The .ready marker is created only after tar and
    # digest succeed. TIFF and model artifacts stay outside Git.
    tar -C "$FULL_ROOT/ld" -czf "$ARCHIVES/fold_${fold}.tar.gz" "fold_$fold" \
        -C "$RUN_ROOT" "fold_${fold}_train.log" "fold_${fold}_evaluate.log" \
        "fold_${fold}_start_utc.txt" "fold_${fold}_end_utc.txt"
    sha256sum "$ARCHIVES/fold_${fold}.tar.gz" > "$ARCHIVES/fold_${fold}.sha256"
    touch "$ARCHIVES/fold_${fold}.ready"
    echo "LD_FOLD_${fold}_COMPLETE"
done

python "$STUDY_ROOT/code/ld_cv_report.py" \
    --run-root "$FULL_ROOT" --output-dir "$RUN_ROOT/cv_report" \
    > "$RUN_ROOT/cv_report.log" 2>&1
tar -C "$RUN_ROOT" -czf "$ARCHIVES/final_metadata.tar.gz" \
    start_utc.txt hostname.txt nvidia_smi_start.txt nvidia_smi_list.txt \
    gpu_preflight.log gpu_usage.csv fit_smoke.log cv_report cv_report.log \
    fold_0_start_utc.txt fold_0_end_utc.txt \
    fold_1_start_utc.txt fold_1_end_utc.txt \
    fold_2_start_utc.txt fold_2_end_utc.txt \
    fold_3_start_utc.txt fold_3_end_utc.txt \
    fold_4_start_utc.txt fold_4_end_utc.txt
sha256sum "$ARCHIVES/final_metadata.tar.gz" > "$ARCHIVES/final_metadata.sha256"
touch "$ARCHIVES/final_metadata.ready"
echo LD_FIVE_FOLD_BASELINE_COMPLETE
