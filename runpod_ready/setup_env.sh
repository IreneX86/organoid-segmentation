#!/usr/bin/env bash
# Prepare an isolated Python 3.11 environment on a Runpod Ubuntu x86_64 Pod.
# Run only after a Pod has been explicitly approved and created.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MINIFORGE_DIR="$WORKSPACE/miniforge3"
INSTALLER="$WORKSPACE/Miniforge3-Linux-x86_64.sh"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Expected an x86_64 Pod" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required in the Pod image" >&2
    exit 1
fi
mkdir -p "$WORKSPACE"
if [[ ! -f "$MINIFORGE_DIR/etc/profile.d/conda.sh" ]]; then
    curl -fL --retry 3 \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
fi

# The shell hook is required: `conda activate` is not available from PATH alone.
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx organoid-runpod; then
    conda create -y -n organoid-runpod python=3.11
fi
conda activate organoid-runpod

# Match the versions observed in the Cetus organoid environment. TensorFlow's
# `and-cuda` supplies CUDA/cuDNN; the Mito hard-negative ablation uses SciPy.
python -m pip install --no-cache-dir \
    'tensorflow[and-cuda]==2.21.0' \
    'numpy==2.4.6' \
    'tifffile==2026.3.3' \
    'scipy==1.16.2'
python -m pip check
python --version
python -m pip show tensorflow numpy tifffile scipy | grep -E '^(Name|Version):'
