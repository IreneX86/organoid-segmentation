# ER single-training-crop fitting diagnostic

This is a training-capacity diagnostic, not a candidate model or validation experiment.

- Crop selection: lexicographically first existing training record, `cell_20260814_23298_NZ_50_z007_y240_x560.npz`. No selection using prediction quality.
- Starting weights: preserved train-only BN-calibrated interline candidate.
- Input/partial labels/normalization unchanged. One Fluo crop only; no ER source or validation data uploaded.
- Freeze BN statistics and affine weights; call network with `training=False` (dropout off) while differentiating other trainable weights.
- Adam0.0003; seed20261029; fixed500 gradient updates with10-minute training guard, threshold0.5. No augmentation, validation selection or parameter search.
- Record known-pixel Dice/IoU/precision/recall and ignored-positive area before/after. Ignored pixels are not treated as biological negatives.
- Save diagnostic checkpoint/probabilities. Never replace or deploy production weights.
- A high fit score establishes capacity to memorize known labels only. It does not establish ER recovery or accuracy on unknown pixels. A poor fit prompts technical diagnosis, not a conclusion that Fluo cannot encode ER.
- One Secure A40, ephemeral50GB disk, no network volume. Budget ceilingUS$1; terminate by2026-10-29T10:00Z after preserving outputs. No second Pod automatically.
