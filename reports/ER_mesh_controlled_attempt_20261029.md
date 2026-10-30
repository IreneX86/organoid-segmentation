# ER mesh-window controlled pilot: infrastructure failure

The preregistered two-arm comparison has not run. Secure Cloud A40 Pod `[pod-id-redacted]` in EU-SE-1 could not create its container: NVIDIA reported `[GPU requires reset]`. No project files were uploaded, no training began and no new checkpoint or scientific result was generated.

The Pod was stopped and terminated immediately. The account showed zero GPUs, vCPUs, storage and endpoints afterward; displayed balance remained US$4.72 (rounded, not an exact invoice).

Prepared artifacts: [protocol](../docs/er_mesh_controlled_protocol_20261029.md), [training script](../experiments/er_mesh_controlled_train.py). The ignored local payload contains 56 Fluo/partial-label NPZ files and the calibrated source checkpoint, no raw ER TIFFs. Syntax and payload hash/content checks passed. Original images, labels and models remain unchanged.

Resume the same controlled experiment on a healthy A40 host. Do not interpret this infrastructure error as model failure or change scientific settings in response.
