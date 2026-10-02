#!/usr/bin/env bash
# Run one isolated, validation-gated Mito hard-negative experiment on one Pod.
# The script never reads candidate outer-test masks before decisions are locked.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
PREFLIGHT_SCRIPT="${PREFLIGHT_SCRIPT:-/workspace/gpu_preflight.py}"
RUN_ROOT="${RUN_ROOT:-/workspace/mito_hard_negative_runs}"
BASELINE_ROOT="$PROJECT_ROOT/baseline/mito_five_fold_20261001"
BASELINE_MODELS="$PROJECT_ROOT/baseline_models"
FROZEN_MAP="$BASELINE_ROOT/fold_0/split.json"
VARIANT_ROOT="$RUN_ROOT/hard_negative"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing existing output directory: $RUN_ROOT" >&2
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

date -u +%FT%TZ > "$RUN_ROOT/start_utc"
nvidia-smi -L > "$RUN_ROOT/logs/nvidia_smi"
python -c 'import ctypes; s=ctypes.CDLL("libcuda.so.1").cuInit(0); print("cuInit_result=",s); assert s == 0' \
    > "$RUN_ROOT/logs/cuda_driver.log" 2>&1
python "$PREFLIGHT_SCRIPT" > "$RUN_ROOT/logs/gpu_preflight.log" 2>&1
grep -qx RUNPOD_GPU_PREFLIGHT_PASS "$RUN_ROOT/logs/gpu_preflight.log"

# The caller provides the budget deadline measured from Pod creation. A fold
# starts only with enough time left for training and result preservation.
budget_end="${BUDGET_END_EPOCH:?Set the absolute Pod cost-guard deadline}"
printf '%s\n' "$budget_end" > "$RUN_ROOT/budget_end_epoch.txt"
for fold in 0 1 2 3 4; do
    if (( budget_end - $(date +%s) < 65 * 60 )); then
        echo "COST_GUARD_STOP: insufficient time for fold $fold" >&2
        exit 75
    fi
    prefix="$RUN_ROOT/logs/hard_negative_fold_${fold}"
    date -u +%FT%TZ > "${prefix}.start_utc"
    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv -l 15 > "${prefix}.gpu_usage.csv" &
    monitor_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30s 55m \
        python "$PROJECT_ROOT/code/train.py" --project-root "$PROJECT_ROOT" \
        --target mito --fold "$fold" --num-folds 5 \
        --expected-mito-fold-map "$FROZEN_MAP" --defer-mito-outer-test \
        --require-new-run-dir --output-root "$VARIANT_ROOT" \
        --mito-hard-negative-fraction 0.2 \
        > "${prefix}.train.log" 2> "${prefix}.train.stderr.log"
    status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$status" > "${prefix}.exit_status"
    date -u +%FT%TZ > "${prefix}.end_utc"
    if [[ "$status" -ne 0 ]]; then
        echo "TRAINING_FAILED: hard_negative fold $fold exit $status" >&2
        exit "$status"
    fi
    tar -czf "$RUN_ROOT/archives/hard_negative_fold${fold}.tar.gz" \
        -C "$RUN_ROOT" "hard_negative/mito/fold_$fold" \
        "logs/hard_negative_fold_${fold}.start_utc" \
        "logs/hard_negative_fold_${fold}.end_utc" \
        "logs/hard_negative_fold_${fold}.exit_status" \
        "logs/hard_negative_fold_${fold}.gpu_usage.csv" \
        "logs/hard_negative_fold_${fold}.train.log" \
        "logs/hard_negative_fold_${fold}.train.stderr.log"
    sha256sum "$RUN_ROOT/archives/hard_negative_fold${fold}.tar.gz" \
        > "$RUN_ROOT/archives/hard_negative_fold${fold}.sha256"
    echo "VALIDATION_ONLY_COMPLETE: hard_negative fold $fold"
done

python "$PROJECT_ROOT/code/mito_round2_gate.py" \
    --project-root "$PROJECT_ROOT" --baseline-root "$BASELINE_ROOT" \
    --baseline-model-root "$BASELINE_MODELS" --candidate-root "$VARIANT_ROOT" --variant hard_negative \
    > "$VARIANT_ROOT/gate.log" 2>&1
echo "VALIDATION_DECISIONS_LOCKED"

for fold in 0 1 2 3 4; do
    if python - "$VARIANT_ROOT/candidate_decisions.json" "$fold" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    decision = json.load(handle)["decisions"][int(sys.argv[2])]
sys.exit(0 if decision["selected_candidate"] else 1)
PY
    then
        python "$PROJECT_ROOT/code/mito_export_predictions.py" \
            --project-root "$PROJECT_ROOT" \
            --run-dir "$VARIANT_ROOT/mito/fold_$fold" \
            --candidate-decisions "$VARIANT_ROOT/candidate_decisions.json" \
            --write-test-report \
            > "$RUN_ROOT/logs/hard_negative_fold_${fold}.export.log" 2>&1
        tar -czf "$RUN_ROOT/archives/hard_negative_selected_fold${fold}.tar.gz" \
            -C "$RUN_ROOT" "hard_negative/mito/fold_$fold/test_per_image.csv" \
            "hard_negative/mito/fold_$fold/test_summary.json" \
            "hard_negative/mito/fold_$fold/threshold_policy_per_image.csv" \
            "hard_negative/mito/fold_$fold/predictions" \
            "logs/hard_negative_fold_${fold}.export.log"
        sha256sum "$RUN_ROOT/archives/hard_negative_selected_fold${fold}.tar.gz" \
            > "$RUN_ROOT/archives/hard_negative_selected_fold${fold}.sha256"
    fi
done

python "$PROJECT_ROOT/code/mito_candidate_report.py" \
    --baseline-root "$BASELINE_ROOT" --candidate-root "$VARIANT_ROOT" \
    > "$VARIANT_ROOT/report.log" 2>&1
tar -czf "$RUN_ROOT/archives/hard_negative_final.tar.gz" \
    -C "$RUN_ROOT" "hard_negative/candidate_decisions.json" \
    "hard_negative/candidate_oof_summary.json" "hard_negative/gate.log" \
    "hard_negative/report.log" "logs/gpu_preflight.log" "logs/cuda_driver.log" \
    "logs/nvidia_smi" "start_utc" "budget_end_epoch.txt"
sha256sum "$RUN_ROOT/archives/hard_negative_final.tar.gz" \
    > "$RUN_ROOT/archives/hard_negative_final.sha256"
date -u +%FT%TZ > "$RUN_ROOT/end_utc"
touch "$RUN_ROOT/COMPLETE"
echo "HARD_NEGATIVE_EXPERIMENT_COMPLETE"
