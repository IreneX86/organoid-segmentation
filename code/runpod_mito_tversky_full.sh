#!/usr/bin/env bash
# Complete the independent five-fold Mito BCE+Tversky experiment on one A40.
# Outer-test labels remain untouched until all validation-only choices lock.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/clean/Organoid}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-$(dirname "$PROJECT_ROOT")/gpu_preflight.py}"
RUN_ROOT="${RUN_ROOT:-/workspace/mito_tversky_full_runs}"
BASELINE_ROOT="$PROJECT_ROOT/baseline/mito_five_fold_20261001"
FROZEN_MAP="$BASELINE_ROOT/fold_0/split.json"
VARIANT_ROOT="$RUN_ROOT/tversky_06"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to reuse experiment directory: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$VARIANT_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/archives"

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH PROJECT_ROOT

# Stop before starting a fold if the run has insufficient time for that fold,
# verification, and transfer. The Pod must be removed separately after local
# archives have been downloaded and checked.
budget_end=$(($(date +%s) + 185 * 60))
printf '%s\n' "$budget_end" > "$RUN_ROOT/budget_end_epoch.txt"

for fold in 0 1 2 3 4; do
    if (( budget_end - $(date +%s) < 45 * 60 )); then
        echo "COST_GUARD_STOP: insufficient time for fold $fold" >&2
        exit 75
    fi
    prefix="$RUN_ROOT/logs/tversky_06_fold_${fold}"
    date -u +%FT%TZ > "${prefix}.start_utc"
    hostname > "${prefix}.hostname"
    nvidia-smi -L > "${prefix}.nvidia_smi"
    python -c 'import ctypes; status=ctypes.CDLL("libcuda.so.1").cuInit(0); print("cuInit_result=",status); assert status == 0' \
        > "${prefix}.cuda_driver.log" 2>&1
    python "$PREFLIGHT_SCRIPT" > "${prefix}.gpu_preflight.log" 2>&1
    grep -qx RUNPOD_GPU_PREFLIGHT_PASS "${prefix}.gpu_preflight.log"

    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv -l 10 > "${prefix}.gpu_usage.csv" &
    monitor_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30s 40m \
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
        echo "TRAINING_FAILED: tversky_06 fold $fold status $training_status" >&2
        exit "$training_status"
    fi

    # Per-fold archives make it possible to safeguard finished checkpoints
    # while the next fold trains. No source TIFF or mask is included.
    tar -czf "$RUN_ROOT/archives/tversky_fold${fold}.tar.gz" \
        -C "$RUN_ROOT" "tversky_06/mito/fold_$fold" "logs/tversky_06_fold_${fold}.start_utc" \
        "logs/tversky_06_fold_${fold}.end_utc" "logs/tversky_06_fold_${fold}.exit_status" \
        "logs/tversky_06_fold_${fold}.hostname" "logs/tversky_06_fold_${fold}.nvidia_smi" \
        "logs/tversky_06_fold_${fold}.cuda_driver.log" \
        "logs/tversky_06_fold_${fold}.gpu_preflight.log" \
        "logs/tversky_06_fold_${fold}.gpu_usage.csv" \
        "logs/tversky_06_fold_${fold}.train.log" \
        "logs/tversky_06_fold_${fold}.train.stderr.log"
    sha256sum "$RUN_ROOT/archives/tversky_fold${fold}.tar.gz" \
        > "$RUN_ROOT/archives/tversky_fold${fold}.sha256"
    echo "VALIDATION_ONLY_COMPLETE: tversky_06 fold $fold"
done

python "$PROJECT_ROOT/code/mito_candidate_gate.py" \
    --baseline-root "$BASELINE_ROOT" --candidate-root "$VARIANT_ROOT" \
    --variant tversky_06 > "$VARIANT_ROOT/gate.log" 2>&1

# The locked decision file alone determines which candidate folds can be
# evaluated on outer test. Rejected folds reuse the frozen baseline reports.
for fold in 0 1 2 3 4; do
    if python - "$VARIANT_ROOT/candidate_decisions.json" "$fold" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    decisions = json.load(handle)["decisions"]
sys.exit(0 if decisions[int(sys.argv[2])]["selected_candidate"] else 1)
PY
    then
        python "$PROJECT_ROOT/code/mito_export_predictions.py" \
            --project-root "$PROJECT_ROOT" \
            --run-dir "$VARIANT_ROOT/mito/fold_$fold" \
            --candidate-decisions "$VARIANT_ROOT/candidate_decisions.json" \
            --write-test-report \
            > "$RUN_ROOT/logs/tversky_06_fold_${fold}.export.log" 2>&1
    fi
done
touch "$VARIANT_ROOT/COMPLETE"
echo "TVERSKY_EXPERIMENT_COMPLETE"
