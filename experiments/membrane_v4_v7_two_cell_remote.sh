#!/usr/bin/env bash
set -euo pipefail

# Remote one-Pod application of the frozen v4/v7 local11 inference to the two
# remaining complete CF stacks. Each source/model is SHA-checked by Python.
root=/workspace/membrane_v4_v7_batch_20261026
cd "$root"
if [[ -e batch_complete.marker || -e batch_failed.marker ]]; then
    echo 'Batch already finished; refusing a duplicate' >&2
    exit 2
fi
date -u
hostname
nvidia-smi
if [[ ! -f .venv_setup.done ]]; then
    python3 -m venv --system-site-packages .venv
    .venv/bin/python -m pip install 'cellpose==4.2.1.1' > setup.log 2>&1
    touch .venv_setup.done
fi
.venv/bin/python -m pip freeze > environment.txt
for cell in cell_21420_NZ_100 cell_21421_NZ_100; do
    if [[ -e "$cell/exit_status.txt" || -d "$cell/results" ]]; then
        echo "Prior result exists for $cell; refusing overwrite" >&2
        exit 2
    fi
    echo "BEGIN_CELL $cell $(date -u +%FT%TZ)"
    set +e
    MEMBRANE_CF_ROOT="$root/$cell" .venv/bin/python -u "$cell/infer_cf100.py" > "$cell/inference.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$cell/exit_status.txt"
    if [[ "$status" -ne 0 ]]; then
        touch batch_failed.marker
        echo "FAILED_CELL $cell status=$status"
        exit "$status"
    fi
    tar -czf "$cell/results.tar.gz" -C "$cell" results inference.log exit_status.txt input_manifest.json z_intensity_profile.json
    sha256sum "$cell/results.tar.gz" > "$cell/results.tar.gz.sha256"
    echo "CELL_ARCHIVE_READY $cell $(date -u +%FT%TZ)"
done
touch batch_complete.marker
echo "MEMBRANE_V4_V7_BATCH_COMPLETE $(date -u +%FT%TZ)"
