#!/usr/bin/env bash
# Run exactly one predeclared Nuc contrast candidate and its locked stress audit.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
RUN_ROOT="${RUN_ROOT:-/workspace/nuc_low_contrast_$(date -u +%Y%m%dT%H%M%SZ)}"
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

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 15 > "$RUN_ROOT/gpu_usage.csv" 2>&1 &
monitor_pid=$!

echo "NUC_LOW_CONTRAST_TRAINING_START $(date -u +%FT%TZ)" \
    | tee "$RUN_ROOT/training_state.txt"
python code/nuc_train_internal.py \
    --project-root "$PROJECT_ROOT" \
    --split-manifest "$PROJECT_ROOT/experiments/results/nuc_internal_split_v1.json" \
    --output-dir "$RUN_ROOT/model" \
    --low-contrast-probability 0.5 \
    --low-contrast-sigma 64 \
    --low-contrast-alpha-min 0.5 \
    2>&1 | tee "$RUN_ROOT/train.log"

echo "NUC_LOW_CONTRAST_STRESS_START $(date -u +%FT%TZ)" \
    | tee -a "$RUN_ROOT/training_state.txt"
python code/nuc_low_contrast_stress.py \
    --project-root "$PROJECT_ROOT" \
    --split-manifest "$PROJECT_ROOT/experiments/results/nuc_internal_split_v1.json" \
    --baseline-model "$PROJECT_ROOT/reference/nuc_baseline.keras" \
    --candidate-model "$RUN_ROOT/model/nuc_unet.keras" \
    --output-dir "$RUN_ROOT/stress" \
    2>&1 | tee "$RUN_ROOT/stress.log"

echo "NUC_LOW_CONTRAST_TRAINING_COMPLETE $(date -u +%FT%TZ)" \
    | tee -a "$RUN_ROOT/training_state.txt"
tar -C "$RUN_ROOT" -czf "$RUN_ROOT/results.tar.gz" \
    model stress preflight.log train.log stress.log training_state.txt \
    gpu_usage.csv nvidia_smi_start.txt start_utc.txt hostname.txt
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
echo "NUC_LOW_CONTRAST_ARCHIVE_READY $RUN_ROOT/results.tar.gz"
