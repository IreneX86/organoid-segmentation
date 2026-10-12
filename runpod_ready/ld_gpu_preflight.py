"""Verify one usable RunPod GPU and the fixed LD dataset before any fit."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile


def main() -> None:
    print(f"python={sys.executable} {sys.version.split()[0]}", flush=True)
    print(f"tensorflow={tf.__version__} numpy={np.__version__} "
          f"tifffile={tifffile.__version__}", flush=True)
    print(f"tensorflow_build={tf.sysconfig.get_build_info()}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
          flush=True)
    print(f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}",
          flush=True)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("LD baseline requires Python 3.11")
    if (tf.__version__, np.__version__, tifffile.__version__) != (
        "2.21.0", "2.4.6", "2026.3.3"
    ):
        raise RuntimeError("LD baseline dependency versions differ from the frozen environment")

    result = ctypes.CDLL("libcuda.so.1").cuInit(0)
    print(f"cuInit={result}", flush=True)
    if result != 0:
        raise RuntimeError("CUDA driver initialization failed")
    physical = tf.config.list_physical_devices("GPU")
    print(f"physical_gpus={physical}", flush=True)
    if len(physical) != 1:
        raise RuntimeError("Expected precisely one RunPod-assigned GPU")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.set_memory_growth(physical[0], True)
    logical = tf.config.list_logical_devices("GPU")
    print(f"logical_gpus={logical}", flush=True)
    if len(logical) != 1:
        raise RuntimeError("Expected precisely one logical GPU")
    with tf.device("/GPU:0"):
        result_tensor = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
        checksum = float(tf.reduce_sum(result_tensor).numpy())
    print(f"matmul_device={result_tensor.device} checksum={checksum}", flush=True)
    if "GPU:0" not in result_tensor.device.upper() or checksum != 32768.0:
        raise RuntimeError("Matrix multiplication did not run on the assigned GPU")

    root = Path(os.environ.get("LD_STUDY_ROOT", "/workspace/ld_study"))
    inputs = sorted((root / "LD_input").glob("ld_*.tif"))
    labels = sorted((root / "LD_output_binary").glob("ld_*.tif"))
    if len(inputs) != 36 or len(labels) != 36:
        raise RuntimeError(f"Expected 36 inputs and labels, got {len(inputs)}/{len(labels)}")
    if [path.name for path in inputs] != [path.name for path in labels]:
        raise RuntimeError("LD input and derived-label filenames differ")
    image = tifffile.imread(inputs[0])
    label = tifffile.imread(labels[0])
    if image.shape != (1200, 1200) or label.shape != image.shape:
        raise RuntimeError("LD preflight found an unexpected pixel grid")
    if label.dtype != np.uint8 or not np.isin(label, [0, 1]).all():
        raise RuntimeError("LD preflight requires the frozen uint8 0/1 derived masks")
    print("LD_GPU_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
