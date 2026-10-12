#!/usr/bin/env bash
# Final LD all-36 fit on the existing single-A40 RunPod Pod. Never deletes a Pod.
set -euo pipefail

STUDY_ROOT=/workspace/ld_study
RUN_ROOT=/workspace/ld_final_20261012
MODEL_DIR="$RUN_ROOT/model"
LOGS="$RUN_ROOT/logs"
ARCHIVES="$RUN_ROOT/archives"
METHOD="${LD_FINAL_METHOD:?Set LD_FINAL_METHOD=baseline or candidate}"
POLICY="${LD_FINAL_POLICY:?Set LD_FINAL_POLICY to the chosen locked policy JSON}"
GLOBAL_POLICY="${LD_GLOBAL_METHOD_POLICY:?Set LD_GLOBAL_METHOD_POLICY to the locked global method policy JSON}"
FOLD_MAP="$STUDY_ROOT/fold_map.json"
LABEL_MANIFEST="$STUDY_ROOT/manifest.json"
BASELINE_CV_ROOT="${LD_BASELINE_CV_ROOT:-/workspace/ld_baseline_20261012/full}"
CANDIDATE_CV_ROOT="${LD_CANDIDATE_CV_ROOT:-/workspace/ld_component_20261012/full}"
VERIFIER="$STUDY_ROOT/runpod_ready/ld_final_artifact.py"

if [[ "$METHOD" != baseline && "$METHOD" != candidate ]]; then
    echo "LD_FINAL_METHOD must be baseline or candidate" >&2
    exit 1
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing LD final run: $RUN_ROOT" >&2
    exit 1
fi
for path in "$POLICY" "$GLOBAL_POLICY" "$FOLD_MAP" "$LABEL_MANIFEST" "$VERIFIER" \
            "$STUDY_ROOT/LD_input" "$STUDY_ROOT/LD_output_binary"; do
    if [[ ! -e "$path" ]]; then
        echo "Required final-fit input is missing: $path" >&2
        exit 1
    fi
done
if [[ "$METHOD" == candidate ]]; then
    for fold in 0 1 2 3 4; do
        for cv_root in "$BASELINE_CV_ROOT" "$CANDIDATE_CV_ROOT"; do
            if [[ ! -f "$cv_root/ld/fold_$fold/selection.lock.json" ]]; then
                echo "Missing locked CV fold $fold: $cv_root" >&2
                exit 1
            fi
        done
    done
    TRAINER="$STUDY_ROOT/code/ld_component_train_final.py"
else
    TRAINER="$STUDY_ROOT/code/ld_train_final.py"
fi
if [[ ! -f "$TRAINER" ]]; then
    echo "Final-fit trainer is missing: $TRAINER" >&2
    exit 1
fi

source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
for lib_dir in /workspace/miniforge3/envs/organoid-runpod/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then
        LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH LD_STUDY_ROOT="$STUDY_ROOT" TF_FORCE_GPU_ALLOW_GROWTH=true

# The baseline policies were recomputed from both complete CV trees locally,
# then frozen by SHA256. Check those exact locks on the Pod without uploading
# validation predictions or checkpoints. Candidate retains CV recomputation.
if [[ "$METHOD" == baseline ]]; then
    preflight_log=global_policy_preflight.log
    preflight_result="$(python "$VERIFIER" check-locked-baseline \
        --method baseline --policy "$POLICY" --global-policy "$GLOBAL_POLICY" \
        --fold-map "$FOLD_MAP" --label-manifest "$LABEL_MANIFEST")"
else
    preflight_log=global_selection_preflight.log
    preflight_result="$(python "$VERIFIER" check-selection --method candidate --policy "$POLICY" \
        --global-policy "$GLOBAL_POLICY" \
        --baseline-cv-root "$BASELINE_CV_ROOT" \
        --candidate-cv-root "$CANDIDATE_CV_ROOT" \
        --fold-map "$FOLD_MAP" --label-manifest "$LABEL_MANIFEST" \
        --label-dir "$STUDY_ROOT/LD_output_binary" --study-root "$STUDY_ROOT")"
fi

mkdir -p "$LOGS" "$ARCHIVES"
monitor_pid=
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ "$status" -ne 0 ]]; then
        printf '%s\n' "$status" > "$LOGS/runner_exit_status.txt"
        date -u +%FT%TZ > "$LOGS/end_utc.txt"
    fi
    exit "$status"
}
trap finish EXIT

date -u +%FT%TZ > "$LOGS/start_utc.txt"
hostname > "$LOGS/hostname.txt"
printf '%s\n' "$preflight_result" > "$LOGS/$preflight_log"
cp "${BASH_SOURCE[0]}" "$LOGS/runner.sh"
cp "$VERIFIER" "$LOGS/artifact_verifier.py"
cp "$GLOBAL_POLICY" "$LOGS/global_method_policy.json"
python - "$METHOD" "$POLICY" "$GLOBAL_POLICY" "$FOLD_MAP" "$LABEL_MANIFEST" \
    "$LOGS/selected_method.json" "$BASELINE_CV_ROOT" "$CANDIDATE_CV_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

method, policy, global_policy, fold_map, label_manifest, target, baseline_cv, candidate_cv = sys.argv[1:]
digest = lambda name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
record = {
    "method": method,
    "policy_path": str(Path(policy).resolve()),
    "policy_sha256": digest(policy),
    "global_policy_sha256": digest(global_policy),
    "fold_map_sha256": digest(fold_map),
    "label_manifest_sha256": digest(label_manifest),
}
if method == "candidate":
    record["baseline_cv_root"] = str(Path(baseline_cv).resolve())
    record["candidate_cv_root"] = str(Path(candidate_cv).resolve())
with Path(target).open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

nvidia-smi -L > "$LOGS/nvidia_smi_list.txt"
gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" != 1 ]] || ! grep -q 'NVIDIA A40' "$LOGS/nvidia_smi_list.txt"; then
    echo "Expected exactly one visible NVIDIA A40" >&2
    exit 1
fi
python "$STUDY_ROOT/runpod_ready/ld_gpu_preflight.py" \
    > "$LOGS/gpu_preflight.log" 2>&1
grep -qx LD_GPU_PREFLIGHT_PASS "$LOGS/gpu_preflight.log"

nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 10 > "$LOGS/gpu_usage.csv" &
monitor_pid=$!

command=(python "$TRAINER"
    --input-dir "$STUDY_ROOT/LD_input"
    --label-dir "$STUDY_ROOT/LD_output_binary"
    --label-manifest "$LABEL_MANIFEST"
    --fold-map "$FOLD_MAP"
    --policy "$POLICY"
    --output-dir "$MODEL_DIR")
if [[ "$METHOD" == candidate ]]; then
    command+=(--baseline-cv-root "$BASELINE_CV_ROOT"
              --candidate-cv-root "$CANDIDATE_CV_ROOT")
fi
timeout --signal=TERM --kill-after=60s 150m "${command[@]}" \
    > "$LOGS/training.log" 2>&1

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
date -u +%FT%TZ > "$LOGS/end_utc.txt"
printf '0\n' > "$LOGS/runner_exit_status.txt"
python "$VERIFIER" verify-tree --tree "$RUN_ROOT" --method "$METHOD" \
    --policy "$POLICY" --global-policy "$GLOBAL_POLICY" --fold-map "$FOLD_MAP" \
    --label-manifest "$LABEL_MANIFEST" > "$LOGS/artifact_prepublication.log" 2>&1
grep -q '^LD_FINAL_TREE_PASS ' "$LOGS/artifact_prepublication.log"

# Publish only a finished all-36 model and audit logs. No CV or outer-test data
# is included. A failed run/packaging step cannot create a ready marker.
tar -C "$RUN_ROOT" -czf "$ARCHIVES/.ld_final_model.tar.gz.partial" model logs
tar -tzf "$ARCHIVES/.ld_final_model.tar.gz.partial" > /dev/null
mv "$ARCHIVES/.ld_final_model.tar.gz.partial" "$ARCHIVES/ld_final_model.tar.gz"
(
    cd "$ARCHIVES"
    sha256sum ld_final_model.tar.gz > ld_final_model.sha256
    sha256sum --check ld_final_model.sha256
)
touch "$ARCHIVES/ld_final_model.ready"
echo "LD_FINAL_ALL36_ARCHIVE_READY method=$METHOD archive=$ARCHIVES/ld_final_model.tar.gz"
