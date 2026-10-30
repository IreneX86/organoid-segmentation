# ER enclosed-gap negative candidates — 2026-10-30

## Scope and result
Prepared a new derived16-window dataset; no training, inference, validation access or paid resources. Original TIFFs, prior masks and checkpoints remain unchanged. These are weak-negative proposals, NOT certified labels.

Added1832 provisional background pixels, preserving all25759 positive pixels and7379 existing weak negatives. Negative supervision now9211 pixels. Overall ignored window pixels32398→30566. Among the8546 previously audited enclosed-gap pixels, ignored fraction falls88.32%→66.89%; most uncertainty remains intentionally unsupervised. Per-specimen counts are in the adjacent metrics JSON.

## Fixed rule
Within each pre-existing64×64 training window (320×320 context), examine only enclosed pseudo-mask gaps of4–256pixels. Candidate pixels must be originally ignored, more than1.5px from the gap boundary, and darker than the25th percentile of the locally labelled rim by more than2 robust noise-proxy units after sigma1 ER smoothing. Require a local trough relative to sigma3 smoothing and a raw-intensity bright-structure guard. Retain candidate components of≥2pixels. Noise proxy is estimated from the original320px training ER reference only. Fluo is not used to invent a target or added to the model as ER input.

No parameter search was performed. The smaller boundary guard addresses small enclosed gaps that the earlier distance>3px interline method left unknown; it is a new experimental assumption, not an established biological threshold. It can still suppress faint missed ER. Do not relabel all gaps as background.

## Review and validation
Fixed representative overview contains the first2 windows per training specimen, not windows chosen for attractive outcomes. Candidate cyan pixels largely follow dark ER troughs, with boundaries retained as ignore. Fluo often lacks the same obvious contrast; correspondence remains a limitation. Reviewed previews cannot certify every candidate pixel or establish improvement.

Synthetic checks: dark enclosed core selected; an unlabelled bright line protected; uniform ER rejected; positives preserved. All16 source hashes and saved-label readbacks pass; loss-window restriction, unchanged Fluo, old positive/negative preservation verified. One manifest-write duplicate-key implementation error occurred before completion; its partial files were preserved separately, code corrected and run repeated with the SAME rules. No scientific retry/tuning.

## Next controlled experiment, prepared conceptually only
Compare prior mesh labels against these new mesh+gap proposals, with identical source checkpoint,16input windows,500updates,seed,stableBN/dropout settings and fixed0.5threshold. Prior matched mesh arm may be reused only if hardware/determinism comparability is accounted for; the safest paired run uses both arms on the same GPU. No extra cropped samples, architecture/loss changes or validation-driven threshold tuning. Known-pixel validation pseudo-Dice alone is insufficient; broad masks must still be rejected. Do not train a production model from these proposals.

## Files
- `experiments/er_gap_negative_candidates.py`
- `outputs/er_gap_negative_candidates_20261030/manifest.json`, `protocol.json`,16NPZs and16individual reviews
- `outputs/er_gap_negative_candidates_20261030/eight_gap_candidates.png`
- New cost US$0. No Pod created.
