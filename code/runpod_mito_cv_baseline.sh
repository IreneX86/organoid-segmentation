#!/usr/bin/env bash
# Reproduce baseline Mito folds 1–4 on one isolated Runpod GPU.
# The scientific training command remains code/train.py with its original defaults.
set -euo pipefail

PROJECT_ROOT=/workspace
RUN_ROOT=/workspace/cv_runs/mito
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod

# Match the CUDA library visibility that passed the A40 fold-0 GPU smoke test.
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH PROJECT_ROOT

mkdir -p "$RUN_ROOT"
for fold in 1 2 3 4; do
    run_dir="$RUN_ROOT/fold_$fold"
    if [[ -e "$run_dir/baseline_unet.keras" || -e "$run_dir/TRAINING_START" ]]; then
        echo "Refusing to overwrite an existing fold: $run_dir" >&2
        exit 1
    fi
    mkdir -p "$run_dir"
    date -u +%FT%TZ > "$run_dir/start_utc.txt"
    hostname > "$run_dir/hostname.txt"
    nvidia-smi -L > "$run_dir/nvidia_smi_start.txt"
    python -c 'import ctypes; result=ctypes.CDLL("libcuda.so.1").cuInit(0); print("cuInit_result=",result); assert result == 0' \
        > "$run_dir/cuda_driver_preflight.log" 2>&1
    python /workspace/gpu_preflight.py > "$run_dir/gpu_preflight.log" 2>&1
    grep -qx RUNPOD_GPU_PREFLIGHT_PASS "$run_dir/gpu_preflight.log"

    # The marker is written only after the fail-closed GPU checks succeed.
    touch "$run_dir/TRAINING_START"
    nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv -l 5 > "$run_dir/gpu_usage.csv" &
    monitor_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30s 80m \
        python "$PROJECT_ROOT/code/train.py" --project-root "$PROJECT_ROOT" \
        --target mito --fold "$fold" --num-folds 5 --epochs 100 \
        --output-root /workspace/cv_runs \
        > "$run_dir/train.log" 2> "$run_dir/train.stderr.log"
    training_status=$?
    set -e
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$training_status" > "$run_dir/train_exit_status.txt"
    if [[ "$training_status" -ne 0 ]]; then
        echo "Fold $fold training failed with exit $training_status" >&2
        exit "$training_status"
    fi

    # Both masks come from the same checkpoint. Test labels are used only to
    # score these already fixed policies, never to choose a threshold.
    python "$PROJECT_ROOT/code/mito_export_predictions.py" \
        --project-root "$PROJECT_ROOT" --run-dir "$run_dir" \
        > "$run_dir/export.log" 2>&1
    date -u +%FT%TZ > "$run_dir/end_utc.txt"
    tar -C "$RUN_ROOT" -czf "/workspace/cv_runs/fold_$fold.tar.gz" "fold_$fold"
    sha256sum "/workspace/cv_runs/fold_$fold.tar.gz" > "/workspace/cv_runs/fold_$fold.sha256"
    touch "/workspace/cv_runs/fold_$fold.ready"
    echo "FOLD_${fold}_COMPLETE"
done
echo ALL_BASELINE_FOLDS_COMPLETE
