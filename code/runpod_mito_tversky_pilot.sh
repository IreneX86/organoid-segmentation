#!/usr/bin/env bash
# Validation-only pilot for the independent BCE+Tversky Mito loss factor.
# This never reads outer-test labels; completed folds require later five-fold CV.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/clean/Organoid}"
RUN_ROOT="${RUN_ROOT:-/workspace/mito_ablation_runs_20261001}"
BASELINE_ROOT="$PROJECT_ROOT/baseline/mito_five_fold_20261001"
FROZEN_MAP="$BASELINE_ROOT/fold_0/split.json"
VARIANT_ROOT="$RUN_ROOT/tversky_06"

if [[ -e "$VARIANT_ROOT" ]]; then
    echo "Refusing to reuse experiment directory: $VARIANT_ROOT" >&2
    exit 1
fi
mkdir -p "$VARIANT_ROOT" "$RUN_ROOT/logs"

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH PROJECT_ROOT

# A short two-fold pilot fits the remaining cost allowance. It cannot be used
# for candidate selection or OOF claims; no outer-test export is run here.
for fold in 0 1; do
    prefix="$RUN_ROOT/logs/tversky_06_fold_${fold}"
    date -u +%FT%TZ > "${prefix}.start_utc"
    nvidia-smi -L > "${prefix}.nvidia_smi"
    python -c 'import ctypes; status=ctypes.CDLL("libcuda.so.1").cuInit(0); print("cuInit_result=",status); assert status == 0' \
        > "${prefix}.cuda_driver.log" 2>&1
    python /workspace/gpu_preflight.py > "${prefix}.gpu_preflight.log" 2>&1
    grep -qx RUNPOD_GPU_PREFLIGHT_PASS "${prefix}.gpu_preflight.log"

    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv -l 10 > "${prefix}.gpu_usage.csv" &
    monitor_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30s 25m \
        python "$PROJECT_ROOT/code/train.py" --project-root "$PROJECT_ROOT" \
        --target mito --fold "$fold" --num-folds 5 \
        --expected-mito-fold-map "$FROZEN_MAP" --defer-mito-outer-test \
        --require-new-run-dir --output-root "$VARIANT_ROOT" \
        --mito-loss bce_tversky_06 \
        > "${prefix}.train.log" 2> "${prefix}.train.stderr.log"
    training_status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$training_status" > "${prefix}.exit_status"
    date -u +%FT%TZ > "${prefix}.end_utc"
    if [[ "$training_status" -ne 0 ]]; then
        echo "PILOT_FOLD_FAILED: fold $fold status $training_status" >&2
        exit "$training_status"
    fi
    echo "VALIDATION_ONLY_PILOT_COMPLETE: tversky_06 fold $fold"
done
echo "TVERSKY_PILOT_FINISHED"
