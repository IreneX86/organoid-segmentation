#!/usr/bin/env bash
# Dedicated one-GPU Pod only. Archive partial results even on technical failure.
set -euo pipefail
root=${ER_RUN_ROOT:-/workspace/er_partial_student_20261029}
mkdir -p "$root"
exec 9>"$root/runner.lock"
flock -n 9 || exit 90
[[ ! -e "$root/started" ]] || exit 91
date -u +%FT%TZ > "$root/started"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$root/exit_status.txt"
    tar -czf "$root/results.tar.gz" -C "$root" results exit_status.txt started supervisor.log
    sha256sum "$root/results.tar.gz" > "$root/results.tar.gz.sha256"
    exit "$status"
}
mkdir -p "$root/results"
trap finish EXIT
nvidia-smi
python3 - <<'PY'
import ctypes
driver=ctypes.CDLL('libcuda.so.1')
status=driver.cuInit(0)
print('cuInit=',status,flush=True)
assert status==0
PY
WORKSPACE=/workspace bash /workspace/setup_env.sh
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
python -m pip install --no-cache-dir 'pillow>=10,<13'
site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for lib in "$site"/nvidia/*/lib; do
    [[ ! -d "$lib" ]] || export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
done
bash /workspace/Organoid/code/runpod_er_gpu_preflight.sh
cd /workspace/Organoid/code
timeout --signal=INT --kill-after=60s 7200 python -u er_partial_student.py train \
    --data /workspace/er_partial_data --output "$root/results/student"
echo ER_PARTIAL_RUN_COMPLETE
