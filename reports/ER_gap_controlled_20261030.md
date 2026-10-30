# ER gap-negative controlled ablation — 2026-10-30 Sydney

## Decision

REJECT for production. Both500-update runs completed, but additional gap negatives did not yield reliably thin ER masks. All8 fixed validation previews still show broad fused positive regions. No model, label or threshold was promoted.

## Comparison

Same16trainingFluo windows, same input order/normalization, calibrated sourcecheckpoint, frozenBN/dropoutoff, Adam0.0003,batch4,seed20261029,500fixedupdates. Only1832pixels changed ignore→provisionalnegative. No ER input. Same8unchanged validation crops,threshold0.5. Finalstepcheckpoint only; no tuning/retraining. Inputs/label-differences verified locally and remotely.

| Mean metric | Original mesh | Mesh + gap negatives |
|---|---:|---:|
| dice | 0.7434 | 0.7748 |
| iou | 0.5950 | 0.6348 |
| precision | 0.7262 | 0.7273 |
| recall | 0.7686 | 0.8326 |
| positive_fraction | 0.6345 | 0.6795 |
| ignored_positive_fraction | 0.6180 | 0.6641 |

Paired pseudo-Dice increases in5/8crops,mean+0.0314,not a consistent eight-crop improvement. Higher recall accompanies increased predicted area63.45%→67.95%; visual broad masks persist. Ignored-area predictions cannot be classified as true/false without reference. This fails the intended reduction of broad sheet-like predictions.

## Limitations

These are partialpseudo-label metrics from ONE previously viewed validation specimen, not independent biological accuracy or8independent subjects. New negatives may suppress faintER. No ground-truth claim. Fixedseed was used but deterministic GPU kernels were not enforced: the originalmesh control here scores0.7434 versus0.7704 in the prior allocation. Thus exact repeatability and a small numerical gain are not established; use the same-allocation paired comparison and visual failure, not cross-run score ranking. No repeated seeds were run just to obtain a favorable outcome.

Training time (updates only): originalmesh18.32s,gap13.43s. Final sampled-batch losses0.000155/0.003805 are not whole-training accuracy. Low training loss and poor validation morphology are compatible with limited-supervision overfitting and channel mismatch, but do not isolate the cause.

## Next informative diagnostic

Stop scanning gap ratios or thresholds. A potentially useful next control is ER-reference input→same labels on the same frozen specimen split, solely to distinguish target/objective limitations from cross-channel Fluo information limitations. If evenER cannot produce thin masks under the same supervision, address labels/objective/fitting first; ifER works butFluo fails, investigate correspondence and information availability. Such a control would NOT replace the intended Fluo-only application. It has not been run, and its outcome cannot prove impossibility from one small study.

## Preservation and compute

SecureCloudRTXA6000 48GB,US-TX-1,Pod[pod-id-redacted]. Python3.11,TF2.21.0,TFbuildCUDA12.5.1/cuDNN9. DriverandoneGPUcomputationchecks passed. Exit0,complete marker,archiveSHA256,bothcheckpointSHA256/ZIP,andall16finite320pxoutputchecksPASS. Results downloaded beforePodstopped/deleted. OriginalTIFFs,masks andpriorcheckpointunchanged. No sourceERchannel uploaded.

## Files

- Full models,archive,logs andall8comparison: `outputs/er_gap_controlled_20261030/`.
- Figure: `all8_mesh_vs_gap.png` (Fluo / partial labels / originalmesh / mesh+gap).
- Small metrics: `reports/ER_gap_controlled_metrics_20261030.json`.
- Protocol: `docs/er_gap_controlled_protocol_20261030.md`.
- Scripts: `experiments/er_gap_controlled_train.py`, `experiments/er_gap_controlled_review.py`.
- No TIFFs/checkpoints/logs/credentials pushed.

## Final cleanup
BillingverifiedGPUs0,vCPUs0,storage0,endpoints0,spend$0/hour. Balance$4.66→$4.59;approxnewcost$0.07 (roundeddisplaydifference,notexactinvoice). No additional training/resources.
