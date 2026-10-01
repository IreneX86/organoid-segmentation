#!/usr/bin/env bash
# Run two independent, validation-gated Mito ablations on one verified A40 Pod.
# Data and prior baseline outputs are read-only; all new files stay in RUN_ROOT.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
RUN_ROOT="${RUN_ROOT:-/workspace/mito_ablation_runs}"
BASELINE_ROOT="$PROJECT_ROOT/baseline/mito_five_fold_20261001"
FROZEN_MAP="$BASELINE_ROOT/fold_0/split.json"
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod

for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH PROJECT_ROOT

mkdir -p "$RUN_ROOT/logs"
# Reserve at least 25 minutes to archive and download results before the
# previous US$2 cost guard could be reached at US$0.49/hour.
budget_end=$(($(date +%s) + 180 * 60))
printf '%s\n' "$budget_end" > "$RUN_ROOT/budget_end_epoch.txt"

for variant in sampling_035 tversky_06; do
    variant_root="$RUN_ROOT/$variant"
    if [[ -e "$variant_root" ]]; then
        echo "Refusing to reuse experiment directory: $variant_root" >&2
        exit 1
    fi
    # Do not start the second full candidate if there is insufficient time to
    # run its five folds and still retrieve checkpoints within the cost cap.
    if [[ "$variant" == tversky_06 ]] && (( budget_end - $(date +%s) < 80 * 60 )); then
        echo "COST_GUARD_SKIP_TVERSKY: too little budgeted time remains"
        break
    fi
    mkdir -p "$variant_root"
    for fold in 0 1 2 3 4; do
        if (( budget_end - $(date +%s) < 35 * 60 )); then
            echo "COST_GUARD_STOP: insufficient time for another fold" >&2
            exit 75
        fi
        prefix="$RUN_ROOT/logs/${variant}_fold_${fold}"
        date -u +%FT%TZ > "${prefix}.start_utc"
        hostname > "${prefix}.hostname"
        nvidia-smi -L > "${prefix}.nvidia_smi"
        python -c 'import ctypes; status=ctypes.CDLL("libcuda.so.1").cuInit(0); print("cuInit_result=",status); assert status == 0' \
            > "${prefix}.cuda_driver.log" 2>&1
        python /workspace/gpu_preflight.py > "${prefix}.gpu_preflight.log" 2>&1
        grep -qx RUNPOD_GPU_PREFLIGHT_PASS "${prefix}.gpu_preflight.log"

        if [[ "$variant" == sampling_035 ]]; then
            factor_args=(--mito-positive-patch-probability 0.35)
        else
            factor_args=(--mito-loss bce_tversky_06)
        fi
        nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
            --format=csv -l 10 > "${prefix}.gpu_usage.csv" &
        monitor_pid=$!
        set +e
        timeout --signal=TERM --kill-after=30s 35m \
            python "$PROJECT_ROOT/code/train.py" --project-root "$PROJECT_ROOT" \
            --target mito --fold "$fold" --num-folds 5 \
            --expected-mito-fold-map "$FROZEN_MAP" --defer-mito-outer-test \
            --require-new-run-dir --output-root "$variant_root" \
            "${factor_args[@]}" \
            > "${prefix}.train.log" 2> "${prefix}.train.stderr.log"
        training_status=$?
        set -e
        kill "$monitor_pid" 2>/dev/null || true
        printf '%s\n' "$training_status" > "${prefix}.exit_status"
        date -u +%FT%TZ > "${prefix}.end_utc"
        if [[ "$training_status" -ne 0 ]]; then
            echo "TRAINING_FAILED: $variant fold $fold status $training_status" >&2
            exit "$training_status"
        fi
        echo "VALIDATION_ONLY_COMPLETE: $variant fold $fold"
    done

    python "$PROJECT_ROOT/code/mito_candidate_gate.py" \
        --baseline-root "$BASELINE_ROOT" --candidate-root "$variant_root" \
        --variant "$variant" > "$variant_root/gate.log" 2>&1

    # Only folds selected by the validation-only gate touch outer-test labels.
    for fold in 0 1 2 3 4; do
        if python - "$variant_root/candidate_decisions.json" "$fold" <<'PY'
import json
import sys
decisions = json.load(open(sys.argv[1], encoding="utf-8"))["decisions"]
sys.exit(0 if decisions[int(sys.argv[2])]["selected_candidate"] else 1)
PY
        then
            run_dir="$variant_root/mito/fold_$fold"
            python "$PROJECT_ROOT/code/mito_export_predictions.py" \
                --project-root "$PROJECT_ROOT" --run-dir "$run_dir" \
                --candidate-decisions "$variant_root/candidate_decisions.json" \
                --write-test-report \
                > "$RUN_ROOT/logs/${variant}_fold_${fold}.export.log" 2>&1
        fi
    done
    touch "$variant_root/COMPLETE"
    echo "EXPERIMENT_COMPLETE: $variant"
done

echo "MITO_ABLATION_RUN_FINISHED"
