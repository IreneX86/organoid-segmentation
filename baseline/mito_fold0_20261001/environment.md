# Baseline execution environment

- Runpod Secure Cloud, one NVIDIA A40 GPU, official image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- Python 3.11.16; TensorFlow 2.21.0 installed as `tensorflow[and-cuda]`;
  NumPy 2.4.6; tifffile 2026.3.3.
- TensorFlow build info: CUDA 12.5.1, cuDNN 9; pip NVIDIA runtime libraries
  were added to `LD_LIBRARY_PATH` from the environment's `nvidia/*/lib` dirs.
- The container exposed one A40. CUDA Driver API `cuInit(0)` returned success;
  TensorFlow found one physical/logical GPU and performed matrix multiplication
  on `/device:GPU:0` before training.
- Command, with all other parameters at `code/train.py` defaults:

```bash
python code/train.py --project-root /workspace/Organoid \
  --target mito --fold 0 --num-folds 5 --epochs 100 \
  --output-root /workspace/full_runs/mito_fold0_full_20261001T0153Z/artifacts
```

The local development machine and external deployment paths may differ.
`code/pipeline_config.py` resolves input paths relative to `--project-root`.
No TIFF or mask is stored in this repository.
