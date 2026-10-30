#!/usr/bin/env bash
# Run only inside one dedicated Runpod GPU Pod after a local dataset archive
# has passed the eight-stack ER pseudo-label verifier.
set -euo pipefail

bundle="${1:?dataset .tar required}"
run_root="${2:?unique run root required}"
if [[ -e "$run_root" ]]; then
    echo "Refusing to overwrite run directory: $run_root" >&2
    exit 2
fi
mkdir -p "$run_root/data" "$run_root/results"
status_file="$run_root/exit_status.txt"
trap 'printf "%s\n" "$?" > "$status_file"' EXIT

echo "HOST=$(hostname) START=$(date -u +%FT%TZ)"
nvidia-smi

# Reuse the previously validated isolated TensorFlow 2.21 / CUDA pip-library
# environment. No ER source images are in the training bundle.
export WORKSPACE=/workspace
bash /workspace/setup_env.sh
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for cuda_lib in "$site_packages"/nvidia/*/lib; do
    if [[ -d "$cuda_lib" ]]; then
        export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
    fi
done

tar -xf "$bundle" -C "$run_root/data"
cd /workspace/Organoid/code
python er_fluo_student.py train \
    --data-root "$run_root/data" \
    --output-root "$run_root/results/student" \
    --max-epochs 40 2>&1 | tee "$run_root/train.log"

tar -czf "$run_root/results.tar.gz" -C "$run_root" results train.log
sha256sum "$run_root/results.tar.gz" > "$run_root/results.tar.gz.sha256"
echo "ER_FLUO_RUN_COMPLETE $(date -u +%FT%TZ)"
