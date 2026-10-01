#!/usr/bin/env bash
# Run a bounded Mito fold-0 GPU smoke test; never touch prior Cetus outputs.
# Requires setup_env.sh and the prepared project archive to have been unpacked.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
WORKSPACE="${WORKSPACE:-/workspace}"
source "$WORKSPACE/miniforge3/etc/profile.d/conda.sh"
conda activate organoid-runpod

# Match the CUDA library lookup that passed the Cetus TensorFlow GPU test.
NVIDIA_LIB_ROOT="$(python -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_paths()["purelib"]) / "nvidia")')"
if [[ ! -d "$NVIDIA_LIB_ROOT" ]]; then
    echo "TensorFlow pip NVIDIA libraries are missing" >&2
    exit 1
fi
for lib_dir in "$NVIDIA_LIB_ROOT"/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$WORKSPACE/runpod_runs/$RUN_ID"
mkdir -m 700 "$RUN_DIR"
export PROJECT_ROOT

echo "run_id=$RUN_ID"
echo "output_directory=$RUN_DIR"
date -u
nvidia-smi
echo "cpu_count=$(getconf _NPROCESSORS_ONLN)" | tee "$RUN_DIR/host_resources.log"
free -h | tee -a "$RUN_DIR/host_resources.log"
GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" != "1" ]]; then
    echo "Expected one GPU exposed to this single-GPU Pod, found $GPU_COUNT" >&2
    exit 1
fi
python "$PROJECT_ROOT/runpod_ready/gpu_preflight.py" | tee "$RUN_DIR/gpu_preflight.log"

# CSV sampled every five seconds records utilization and peak memory during
# training. The trap stops the monitor even if training fails or times out.
nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 5 > "$RUN_DIR/gpu_usage.csv" &
MONITOR_PID=$!
(
    while :; do
        printf '%s,' "$(date -u +%FT%TZ)"
        awk '/MemTotal:|MemAvailable:/ {printf "%s=%s_kB,", $1, $2}' /proc/meminfo
        uptime | sed 's/^/load=/'
        sleep 5
    done
) > "$RUN_DIR/cpu_ram_usage.csv" &
HOST_MONITOR_PID=$!
trap 'kill "$MONITOR_PID" "$HOST_MONITOR_PID" 2>/dev/null || true' EXIT

# A 20-minute hard cap includes full-image validation. Three short epochs yield
# observed per-step and per-epoch timings without launching a full 100-epoch run.
echo "smoke_start_utc=$(date -u +%FT%TZ)" | tee "$RUN_DIR/timing.log"
set +e
timeout --signal=TERM --kill-after=30s 20m \
    python "$PROJECT_ROOT/code/train.py" \
    --project-root "$PROJECT_ROOT" \
    --target mito --fold 0 --num-folds 5 \
    --epochs 3 --steps-per-epoch 100 \
    --output-root "$RUN_DIR/artifacts" \
    > "$RUN_DIR/train.log" 2> "$RUN_DIR/train.stderr.log"
status=$?
set -e
echo "smoke_end_utc=$(date -u +%FT%TZ)" | tee -a "$RUN_DIR/timing.log"
echo "smoke_exit_status=$status" | tee -a "$RUN_DIR/timing.log"

# These outputs allow actual GPU peak, utilization, batch time, and epoch time
# to be calculated before approving a full training run.
if [[ "$status" -ne 0 ]]; then
    echo "Smoke test failed or reached its 20-minute limit; see $RUN_DIR" >&2
    exit "$status"
fi
test -s "$RUN_DIR/artifacts/mito/fold_0/training_history.csv"
test -s "$RUN_DIR/artifacts/mito/fold_0/baseline_unet.keras"
echo "RUNPOD_MITO_SMOKE_PASS"
