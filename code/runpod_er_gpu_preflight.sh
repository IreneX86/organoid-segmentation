#!/usr/bin/env bash
set -euo pipefail
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate organoid-runpod
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for cuda_lib in "$site_packages"/nvidia/*/lib; do
    if [[ -d "$cuda_lib" ]]; then
        export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
    fi
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import numpy as np
import tensorflow as tf
import tifffile
import sys

print('python', sys.version.split()[0], 'tensorflow', tf.__version__,
      'numpy', np.__version__, 'tifffile', tifffile.__version__, flush=True)
physical = tf.config.list_physical_devices('GPU')
if len(physical) != 1:
    raise RuntimeError(f'Expected one GPU in the one-GPU Pod, saw {physical}')
tf.config.set_soft_device_placement(False)
tf.config.experimental.set_memory_growth(physical[0], True)
logical = tf.config.list_logical_devices('GPU')
if len(logical) != 1:
    raise RuntimeError(f'Expected one logical GPU, saw {logical}')
with tf.device('/GPU:0'):
    value = tf.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
print('matmul_device', value.device, 'checksum', float(tf.reduce_sum(value)), flush=True)
if 'GPU:0' not in value.device.upper() or float(tf.reduce_sum(value)) != 262144.0:
    raise RuntimeError('GPU matrix operation did not run on the assigned GPU')
print('ER_GPU_PREFLIGHT_PASS', flush=True)
PY
