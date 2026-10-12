#!/usr/bin/env bash
# Prospective LD single-factor candidate. The baseline output is read-only.
# No candidate outer-test mask is opened before its fold-local gate is locked.
set -euo pipefail

STUDY_ROOT=${LD_STUDY_ROOT:-/workspace/ld_study}
BASELINE_ROOT=${LD_BASELINE_ROOT:-/workspace/ld_baseline_20261012/full}
RUN_ROOT=${LD_COMPONENT_RUN_ROOT:?Set a fresh LD_COMPONENT_RUN_ROOT}
FULL_ROOT="$RUN_ROOT/full"
ARCHIVES="$RUN_ROOT/archives"

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH LD_STUDY_ROOT="$STUDY_ROOT" TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing candidate run: $RUN_ROOT" >&2
    exit 1
fi
for fold in 0 1 2 3 4; do
    if [[ ! -f "$BASELINE_ROOT/ld/fold_$fold/selection.lock.json" \
          || ! -f "$BASELINE_ROOT/ld/fold_$fold/outer_test/evaluation_manifest.json" ]]; then
        echo "Complete frozen baseline fold $fold is required" >&2
        exit 1
    fi
done
mkdir -p "$FULL_ROOT" "$ARCHIVES"
cp "$STUDY_ROOT/runpod_ready/ld_component_five_fold.sh" "$RUN_ROOT/runner.sh"
monitor_pid=
finish() {
    status=$?
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
    fi
    # The success marker is placed inside the final archive before it is
    # published. On any packaging failure, overwrite it with the real status;
    # a final .ready marker will then be absent.
    if [[ "$status" -ne 0 || ! -f "$RUN_ROOT/runner_exit_status.txt" ]]; then
        printf '%s\n' "$status" > "$RUN_ROOT/runner_exit_status.txt"
        date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
    fi
}
trap finish EXIT
date -u +%FT%TZ > "$RUN_ROOT/start_utc.txt"
hostname > "$RUN_ROOT/hostname.txt"
nvidia-smi -L > "$RUN_ROOT/nvidia_smi_list.txt"
if [[ "$(nvidia-smi --list-gpus | wc -l)" -ne 1 ]]; then
    echo "Expected one Pod-visible NVIDIA GPU" >&2
    exit 1
fi
python "$STUDY_ROOT/runpod_ready/ld_gpu_preflight.py" \
    > "$RUN_ROOT/gpu_preflight.log" 2>&1
grep -qx LD_GPU_PREFLIGHT_PASS "$RUN_ROOT/gpu_preflight.log"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 10 > "$RUN_ROOT/gpu_usage.csv" &
monitor_pid=$!
python -m unittest discover -s "$STUDY_ROOT/code/tests" \
    -p 'test_ld_component_candidate.py' \
    > "$RUN_ROOT/candidate_unit_tests.log" 2>&1
LD_STUDY_ROOT="$STUDY_ROOT" PYTHONPATH="$STUDY_ROOT/code" python - <<'PY' \
    > "$RUN_ROOT/sampler_batch_preflight.log" 2>&1
import numpy as np
import os
from pathlib import Path
from data_split import split_from_fold_ids
from ld_component_sequence import ComponentUniformPatchSequence
from ld_component_train import _read_frozen_map
from pipeline_config import PipelineConfig
from tiff_io import pair_tiff_directories

root = Path(os.environ["LD_STUDY_ROOT"])
pairs = pair_tiff_directories(root / "LD_input", root / "LD_output_binary")
fold_ids = _read_frozen_map(root / "fold_map.json", {pair.filename for pair in pairs}, 5)
split = split_from_fold_ids(pairs, fold_ids, 0, 5)
config = PipelineConfig()
batch = ComponentUniformPatchSequence(
    split.train, batch_size=1, patch_size=512, steps_per_epoch=1,
    lower_percentile=config.lower_percentile,
    upper_percentile=config.upper_percentile,
    positive_patch_probability=0.6, random_seed=42, augment=False,
)
images, masks = batch[0]
assert images.shape == (1, 512, 512, 1)
assert masks.shape == images.shape
assert np.isfinite(images).all()
assert set(np.unique(masks)).issubset({0.0, 1.0})
print("LD_COMPONENT_BATCH_PREFLIGHT_PASS")
PY
grep -qx LD_COMPONENT_BATCH_PREFLIGHT_PASS "$RUN_ROOT/sampler_batch_preflight.log"

for fold in 0 1 2 3 4; do
    fold_dir="$FULL_ROOT/ld/fold_$fold"
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_start_utc.txt"
    timeout --signal=TERM --kill-after=60s 80m \
        python "$STUDY_ROOT/code/ld_component_train.py" \
            --input-dir "$STUDY_ROOT/LD_input" \
            --label-dir "$STUDY_ROOT/LD_output_binary" \
            --label-manifest "$STUDY_ROOT/manifest.json" \
            --fold-map "$STUDY_ROOT/fold_map.json" \
            --output-root "$FULL_ROOT" --fold "$fold" \
        > "$RUN_ROOT/fold_${fold}_train.log" 2>&1
    python "$STUDY_ROOT/code/ld_component_gate.py" \
        --baseline-run-dir "$BASELINE_ROOT/ld/fold_$fold" \
        --candidate-run-dir "$fold_dir" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        > "$RUN_ROOT/fold_${fold}_gate.log" 2>&1
    python "$STUDY_ROOT/code/ld_component_evaluate.py" \
        --baseline-run-dir "$BASELINE_ROOT/ld/fold_$fold" \
        --candidate-run-dir "$fold_dir" \
        --input-dir "$STUDY_ROOT/LD_input" \
        --label-dir "$STUDY_ROOT/LD_output_binary" \
        --label-manifest "$STUDY_ROOT/manifest.json" \
        --fold-map "$STUDY_ROOT/fold_map.json" \
        > "$RUN_ROOT/fold_${fold}_evaluate.log" 2>&1
    date -u +%FT%TZ > "$RUN_ROOT/fold_${fold}_end_utc.txt"

    # Publish a fold archive only after its gate/evaluation phase finishes.
    tar -C "$FULL_ROOT/ld" -czf "$ARCHIVES/fold_${fold}.tar.gz" "fold_$fold" \
        -C "$RUN_ROOT" "fold_${fold}_train.log" "fold_${fold}_gate.log" \
        "fold_${fold}_evaluate.log" "fold_${fold}_start_utc.txt" \
        "fold_${fold}_end_utc.txt"
    sha256sum "$ARCHIVES/fold_${fold}.tar.gz" > "$ARCHIVES/fold_${fold}.sha256"
    touch "$ARCHIVES/fold_${fold}.ready"
    echo "LD_COMPONENT_FOLD_${fold}_COMPLETE"
done

python "$STUDY_ROOT/code/ld_component_report.py" \
    --baseline-run-root "$BASELINE_ROOT" \
    --candidate-run-root "$FULL_ROOT" \
    --fold-map "$STUDY_ROOT/fold_map.json" \
    --label-manifest "$STUDY_ROOT/manifest.json" \
    --label-dir "$STUDY_ROOT/LD_output_binary" \
    --output-dir "$RUN_ROOT/gated_cv_report" \
    > "$RUN_ROOT/gated_cv_report.log" 2>&1
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
printf '0\n' > "$RUN_ROOT/runner_exit_status.txt"
date -u +%FT%TZ > "$RUN_ROOT/end_utc.txt"
tar -C "$RUN_ROOT" -czf "$ARCHIVES/final_metadata.tar.gz" \
    start_utc.txt end_utc.txt runner_exit_status.txt hostname.txt runner.sh \
    nvidia_smi_list.txt gpu_preflight.log \
    gpu_usage.csv \
    candidate_unit_tests.log sampler_batch_preflight.log \
    gated_cv_report gated_cv_report.log \
    fold_0_start_utc.txt fold_0_end_utc.txt \
    fold_1_start_utc.txt fold_1_end_utc.txt \
    fold_2_start_utc.txt fold_2_end_utc.txt \
    fold_3_start_utc.txt fold_3_end_utc.txt \
    fold_4_start_utc.txt fold_4_end_utc.txt
sha256sum "$ARCHIVES/final_metadata.tar.gz" > "$ARCHIVES/final_metadata.sha256"
echo LD_COMPONENT_FIVE_FOLD_COMPLETE
touch "$ARCHIVES/final_metadata.ready"
