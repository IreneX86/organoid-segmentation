#!/usr/bin/env bash
# Run one frozen Nuc internal prototype after the separate environment setup.
# This is same-batch model development, not an independent test or CV run.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
RUN_ROOT="${RUN_ROOT:-/workspace/nuc_internal_v1_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"

monitor_pid=""
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
    printf '%s\n' "$status" > "$RUN_ROOT/exit_status.txt"
    exit "$status"
}
trap finish EXIT

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
cd "$PROJECT_ROOT"
date -u +%FT%TZ > "$RUN_ROOT/start_utc.txt"
hostname > "$RUN_ROOT/hostname.txt"
nvidia-smi > "$RUN_ROOT/nvidia_smi_start.txt"
python runpod_ready/nuc_gpu_preflight.py 2>&1 | tee "$RUN_ROOT/preflight.log"

# The collector is observational only. It cannot alter training or GPU selection.
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 15 > "$RUN_ROOT/gpu_usage.csv" 2>&1 &
monitor_pid=$!

echo "NUC_INTERNAL_TRAINING_START $(date -u +%FT%TZ)" | tee "$RUN_ROOT/training_state.txt"
python code/nuc_train_internal.py \
    --project-root "$PROJECT_ROOT" \
    --split-manifest "$PROJECT_ROOT/experiments/results/nuc_internal_split_v1.json" \
    --output-dir "$RUN_ROOT/model" \
    2>&1 | tee "$RUN_ROOT/train.log"

echo "NUC_INTERNAL_TRAINING_COMPLETE $(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/training_state.txt"
tar -C "$RUN_ROOT" -czf "$RUN_ROOT/results.tar.gz" \
    model preflight.log train.log training_state.txt gpu_usage.csv \
    nvidia_smi_start.txt start_utc.txt hostname.txt
python - "$RUN_ROOT/results.tar.gz" <<'PY' > "$RUN_ROOT/results.tar.gz.sha256"
import hashlib
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest(), path.name)
PY
echo "NUC_INTERNAL_ARCHIVE_READY $RUN_ROOT/results.tar.gz"
