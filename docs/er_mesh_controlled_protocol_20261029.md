# ER mesh-window controlled pilot — preregistered 2026-10-29

Compare original 32 interline training crops (control) with 16 mesh windows (mesh), using the same four training specimens. Original eight validation crops remain unchanged and are exploratory, previously seen development data, not independent test data.

Both arms start from the same train-only BN-calibrated checkpoint, freeze BN statistics/affine weights and disable dropout. Adam 0.0003; seed 20261029; batch 4 (one uniformly sampled crop from each specimen); 500 updates, no augmentation. Equal updates do not imply equal supervised pixels. Mesh retains 320px context but loss applies only to its 64px window. No new labels, no ER input, no threshold search. Final checkpoint fixed at step 500; validation evaluated only afterward at 0.5.

Compare all eight paired validation outputs, known-pixel pseudo-label Dice/IoU/precision/recall, predicted area and ignored-positive fraction. Pseudo-label agreement cannot establish biological accuracy or generalization. Broad masks or poor mesh structure preclude production approval irrespective of Dice. No repeated tuning following this pilot.

One Secure Cloud A40, no persistent volume; target total new cost below US$1. Per-arm guard10 minutes; training wrapper1300 seconds. Download and SHA256-verify both checkpoints and outputs before deleting Pod. Original TIFFs, labels and prior models unchanged.
