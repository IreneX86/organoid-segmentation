# ER versus Fluo paired input diagnostic — 2026-10-30

## Main finding
The same U-Net can learn the ER-derived binary target substantially better from ER input than from Fluo input in this small controlled experiment. The current Fluo-only pipeline remains unsuitable for production. This supports a cross-channel generalization problem under the tested conditions, not a claim that Fluo-to-ER prediction is fundamentally impossible.

## Design
See [locked protocol](../docs/er_channel_control_protocol_20261030.md). Forty original 320×320 crops: 32 training crops from four specimens and eight validation crops from one separate specimen. Same unchanged user-confirmed binary labels, identical freshly initialized base16 U-Net weights and sampling sequence, deterministic TF operations, seed20261030, Adam0.0003, batch4, 1000fixedupdates per arm, existing training BN/dropout behavior, no augmentation. Input channel is the sole arm difference; p1–p99.8 normalization applied independently by the same rule. Final checkpoint and threshold0.5, no model/threshold selection. These are matched fresh runs, not a direct causal comparison to the earlier500-update frozen-BN model.

## Results
| Metric | Fluo input | ER input |
|---|---:|---:|
| Training mean Dice | 0.8466 | 0.9049 |
| Validation mean Dice | 0.1642 | 0.7425 |
| Validation mean IoU | 0.0913 | 0.5916 |
| Validation mean precision | 0.2519 | 0.8396 |
| Validation mean recall | 0.1248 | 0.6722 |
| Validation predicted foreground fraction | 0.0790 | 0.1207 |

ER-input Dice is higher in8/8validationcrops; these correlated crops are not eight independent specimens. GPU training took53.05s(Fluo) and49.71s(ER), excluding setup, evaluation and transfer.

## Interpretation and limits
The Fluo model fits training crops (Dice0.847) but fails on the separate validation specimen (0.164). ER-input control performs substantially better, making complete inability of this architecture/loss to fit the target an unlikely sole explanation. Fluo-domain variation, weak/ambiguous cross-channel correspondence and limited training diversity remain plausible causes. Pixel-level channel registration and Z-plane correspondence have not been independently demonstrated by this control; pairing files/crop coordinates is not proof of perfect biological alignment. No single cause is proven.

Normalized positive-minus-background mean intensity contrast is0.073(train)/0.076(validation) for Fluo and0.286/0.262forER. This is descriptive, not a claim of measured SNR. Labels themselves are derived fromER, so the ERpositivecontrol is partly circular and its score does not establish biological correctness. The validation specimen was repeatedly inspected; no independent test or biological accuracy claim.

Probability diagnostic on the PREVIOUS binary model (separate experiment) showed lowering0.5→0.1only raised recall0.116→0.195 while FP/crop increased5268→10173. Thus simple threshold lowering is inadequate; see [probability report](ER_binary_probability_20261030.md).

## Visual review
`outputs/er_channel_control_20261030/all8_channel_control.png`: Fluo input / ER input / identical binary target / Fluo prediction / ER prediction. ER predictions recover much of the network, but still miss fine detail; Fluo predictions remain fragmented/off-target. Preserve both as diagnostic checkpoints only.

## Saved artifacts and cleanup
`outputs/er_channel_control_20261030/extracted/results/` contains initial checkpoint, both trained checkpoints,80train+validationprobabilityNPZs, metrics/history/environment records and preflight information. Archive SHA256, both checkpoint SHA256 and ZIP integrity, all prediction dimensions/finite values, label agreement passed.
Fluo checkpoint SHA256: e8e6714c8126881138ac706ebacf7b6429a9001f3f5f8d193a9feacad974cb43
ER checkpoint SHA256: 8192497cb86785a5d5b31e89f3a7f21bf2053f8e23b38ec03c956df58e48c67e

SecureCloudRTXA6000 Pod [pod-id-redacted] deleted after verified download. Console GPUs0/vCPUs0/storage0/endpoints0, spend$0/hour. Rounded displayed balance$4.53→$4.47, approximate cost$0.06, underUS$1ceiling. No originals or prior models changed.

## Recommended next step
Do a read-only cross-channel alignment audit (XY offset and Z correspondence) and compare training/validation Fluo appearance in the paired crops before another GPU experiment. Do not choose registration transforms by maximizing validation label Dice. If alignment is trustworthy, consider a larger/diverse set of reliably visible paired structures with specimen-grouped evaluation; do not indiscriminately mark unconfirmed areas as negative or increase model size merely to fit training crops. No further training launched.
