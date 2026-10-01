"""Fail closed unless one usable NVIDIA GPU is visible to TensorFlow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tifffile


def main() -> None:
    print(f"python_executable={sys.executable}", flush=True)
    print(f"python_version={sys.version.split()[0]}", flush=True)
    print(f"tensorflow_version={tf.__version__}", flush=True)
    print(f"numpy_version={np.__version__}", flush=True)
    print(f"tifffile_version={tifffile.__version__}", flush=True)
    print(f"tensorflow_build_info={tf.sysconfig.get_build_info()}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Expected Python 3.11")
    if (tf.__version__, np.__version__, tifffile.__version__) != (
        "2.21.0", "2.4.6", "2026.3.3"
    ):
        raise RuntimeError("Package versions differ from the Cetus reference environment")

    physical = tf.config.list_physical_devices("GPU")
    print(f"physical_gpus={physical}", flush=True)
    if len(physical) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {len(physical)}")
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.set_memory_growth(physical[0], True)
    logical = tf.config.list_logical_devices("GPU")
    print(f"logical_gpus={logical}", flush=True)
    if len(logical) != 1:
        raise RuntimeError(f"Expected exactly one logical GPU, found {len(logical)}")
    with tf.device("/GPU:0"):
        product = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
        checksum = float(tf.reduce_sum(product).numpy())
    print(f"matmul_device={product.device}", flush=True)
    print(f"matmul_checksum={checksum}", flush=True)
    if "GPU:0" not in product.device.upper() or checksum != 32768.0:
        raise RuntimeError("GPU matrix multiplication did not complete as expected")

    project_root = Path(os.environ.get("PROJECT_ROOT", "/workspace/Organoid"))
    input_dir = project_root / "label_ref/Mito/mito_input"
    label_dir = project_root / "label_ref/Mito/mito_output"
    images = sorted(input_dir.glob("*.tif"))
    masks = sorted(label_dir.glob("*.tif"))
    if len(images) != 36 or len(masks) != 36:
        raise RuntimeError(f"Expected 36 image/mask TIFFs, found {len(images)}/{len(masks)}")
    if [p.name for p in images] != [p.name for p in masks]:
        raise RuntimeError("Input and mask filenames do not match")
    image = tifffile.imread(images[0])
    mask = tifffile.imread(masks[0])
    print(f"sample_shape={image.shape}; mask_shape={mask.shape}", flush=True)
    if image.shape != (1200, 1200) or mask.shape != image.shape:
        raise RuntimeError("Sample image/mask geometry differs from the expected 1200x1200")
    print("RUNPOD_GPU_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
