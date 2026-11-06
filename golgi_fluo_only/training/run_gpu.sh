#!/usr/bin/env bash
# Runpod entry point: bash run.sh  (from /workspace/golgi_student, after decrypting the payload)
set -Eeuo pipefail
root=/workspace/golgi_student; cd "$root"
finish() {
  s=$?; echo "$s" > exit_status.txt; date -u +%FT%TZ > finished_at.txt
  [[ -d output ]] && tar -cf results.tar output run.log && sha256sum results.tar > results.tar.sha256
  echo "RUN_EXIT=$s"
}
trap finish EXIT
nvidia-smi || true
sha256sum -c payload/SHA256SUMS.txt
if [[ ! -f /workspace/miniforge3/etc/profile.d/conda.sh ]]; then
  curl -fL --retry 3 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /workspace/mf.sh
  bash /workspace/mf.sh -b -p /workspace/miniforge3
fi
source /workspace/miniforge3/etc/profile.d/conda.sh
conda env list | awk '{print $1}' | grep -qx golgi || conda create -y -n golgi python=3.11
conda activate golgi
python -m pip install --no-cache-dir 'tensorflow[and-cuda]==2.21.0' scipy scikit-image scikit-learn
sp=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(IFS=:; d=("$sp"/nvidia/*/lib); echo "${d[*]}"):${LD_LIBRARY_PATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert len(g)==1, g; print('GPU_OK', g, tf.__version__)"
python -u payload/code/fluo_golgi_student_v4.py
