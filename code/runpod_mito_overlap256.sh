#!/usr/bin/env bash
# Independent inference-only comparison after the architecture run finishes.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/Organoid}"
RUN_ROOT="${RUN_ROOT:-/workspace/mito_overlap256_runs}"
BASELINE_ROOT="$PROJECT_ROOT/baseline/mito_five_fold_20261001"
BASELINE_MODELS="$PROJECT_ROOT/baseline_models"
BUDGET_END_EPOCH="${BUDGET_END_EPOCH:?Set the absolute Pod budget deadline}"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing existing overlap output: $RUN_ROOT" >&2
    exit 1
fi
if (( BUDGET_END_EPOCH - $(date +%s) < 45 * 60 )); then
    echo "COST_GUARD_SKIP_OVERLAP: insufficient time" >&2
    exit 75
fi
mkdir -p "$RUN_ROOT/archives"
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH

common=(--project-root "$PROJECT_ROOT" --baseline-root "$BASELINE_ROOT"
        --baseline-model-root "$BASELINE_MODELS"
        --output-root "$RUN_ROOT/overlap256")
date -u +%FT%TZ > "$RUN_ROOT/start_utc"
python "$PROJECT_ROOT/code/mito_overlap_trial.py" select "${common[@]}" \
    > "$RUN_ROOT/select.log" 2>&1
echo "OVERLAP_VALIDATION_DECISIONS_LOCKED"
python "$PROJECT_ROOT/code/mito_overlap_trial.py" evaluate "${common[@]}" \
    > "$RUN_ROOT/evaluate.log" 2>&1
date -u +%FT%TZ > "$RUN_ROOT/end_utc"
tar -czf "$RUN_ROOT/archives/overlap256_final.tar.gz" \
    -C "$RUN_ROOT" overlap256 select.log evaluate.log start_utc end_utc
sha256sum "$RUN_ROOT/archives/overlap256_final.tar.gz" \
    > "$RUN_ROOT/archives/overlap256_final.sha256"
touch "$RUN_ROOT/COMPLETE"
echo "OVERLAP_EXPERIMENT_COMPLETE"
