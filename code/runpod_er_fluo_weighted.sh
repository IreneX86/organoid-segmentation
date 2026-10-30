#!/usr/bin/env bash
# One-factor ER student experiment: full old ER target, weighted BCE + Dice.
# Test is locked out; only validation is evaluated in this run.
set -euo pipefail

bundle="${1:?dataset .tar.gz required}"
run_root="${2:?unique run root required}"
if [[ -e "$run_root" ]]; then
    echo "Refusing to overwrite run directory: $run_root" >&2
    exit 2
fi
mkdir -p "$run_root/data" "$run_root/results"
trap 'printf "%s\n" "$?" > "$run_root/exit_status.txt"' EXIT

echo "HOST=$(hostname) START=$(date -u +%FT%TZ)"
nvidia-smi
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
    --max-epochs 40 \
    --loss-variant weighted_bce_dice_5 \
    --validation-only 2>&1 | tee "$run_root/train.log"

tar -czf "$run_root/results.tar.gz" -C "$run_root" results train.log
sha256sum "$run_root/results.tar.gz" > "$run_root/results.tar.gz.sha256"
echo "ER_FLUO_WEIGHTED_VALIDATION_COMPLETE $(date -u +%FT%TZ)"
