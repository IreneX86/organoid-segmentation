"""Fail closed before a paid Nuc training run on a single Runpod A40."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/Organoid"))
FROZEN_SPLIT_SHA256 = "eb1e1be161865e3e92cf93e30dcac7a8287462a670a4068aa97f17a525c8bc54"


def main() -> None:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    print(f"nvidia_smi_gpus={gpu_info}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"python={sys.executable}; version={sys.version.split()[0]}", flush=True)
    print(f"tensorflow={tf.__version__}; numpy={np.__version__}; tifffile={tifffile.__version__}", flush=True)
    print(f"tensorflow_build_info={tf.sysconfig.get_build_info()}", flush=True)
    if len(gpu_info) != 1 or "A40" not in gpu_info[0]:
        raise RuntimeError("Expected exactly one NVIDIA A40 exposed to this Pod")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Expected Python 3.11")
    if (tf.__version__, np.__version__, tifffile.__version__) != (
        "2.21.0", "2.4.6", "2026.3.3"
    ):
        raise RuntimeError("Pinned Python package versions differ from the validated environment")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    status = driver.cuInit(0)
    print(f"cuInit(0)={status}", flush=True)
    if status != 0:
        raise RuntimeError("CUDA Driver API initialization failed")

    physical = tf.config.list_physical_devices("GPU")
    print(f"physical_gpus={physical}", flush=True)
    if len(physical) != 1:
        raise RuntimeError("TensorFlow must see exactly one physical GPU")
    tf.config.experimental.set_memory_growth(physical[0], True)
    tf.config.set_soft_device_placement(False)
    logical = tf.config.list_logical_devices("GPU")
    print(f"logical_gpus={logical}", flush=True)
    if len(logical) != 1:
        raise RuntimeError("TensorFlow must see exactly one logical GPU")
    with tf.device("/GPU:0"):
        value = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
        checksum = float(tf.reduce_sum(value).numpy())
    print(f"matmul_device={value.device}; checksum={checksum}", flush=True)
    if "GPU:0" not in value.device.upper() or checksum != 32768.0:
        raise RuntimeError("GPU matrix multiplication did not complete on GPU:0")

    sys.path.insert(0, str(PROJECT_ROOT / "code"))
    from nuc_train_internal import validate_frozen_split
    from tiff_io import pair_tiff_directories

    data_root = PROJECT_ROOT / "label_ref" / "Nuc"
    pairs = pair_tiff_directories(data_root / "Nuc_input", data_root / "Nuc_output")
    split_path = PROJECT_ROOT / "experiments" / "results" / "nuc_internal_split_v1.json"
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != FROZEN_SPLIT_SHA256:
        raise RuntimeError("Nuc split manifest differs from the frozen local protocol")
    _, training, validation, report = validate_frozen_split(split_path, pairs)
    empty = sorted(row["filename"] for row in report if row["foreground_fraction"] == 0)
    print(f"nuc_pairs={len(pairs)}; train={len(training)}; validation={len(validation)}; empty={empty}", flush=True)
    if ((len(pairs), len(training), len(validation)) != (167, 133, 34)
            or empty != [f"nuc_{index:03d}.tif" for index in range(2, 6)]):
        raise RuntimeError("Nuc data split differs from the frozen 167/133/34 protocol")
    print("NUC_GPU_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
