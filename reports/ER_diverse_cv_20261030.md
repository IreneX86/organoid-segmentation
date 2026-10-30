# ER Fluo-only U-Net: expanded-crop paired fivefold study

Date: 2026-10-30. **Completed; reject this expanded-pool configuration for production. No previous model or label was replaced.**

## Question and locked design
Does adding diverse existing ER-labelled Fluo crops improve out-of-specimen prediction under a fixed compute budget? [Prospective protocol](../docs/er_diverse_roi_cv_protocol_20261030.md).

Five specimens, 40 immutable original 320×320 anchors and 114 newly sampled pseudo-labelled crops. Each fold trains on four whole specimens and evaluates the same eight original anchors from the fifth. Control uses 32 training anchors; expanded uses those anchors plus new training-specimen crops (122–125 crops total). The held-out specimen's new crops are not used. All 40 anchors have exactly one paired OOF result. No historical external specimen is added.

Both arms use Fluo alone, identical initial U-Net weights within each fold, base16, Adam0.0003, 1,000 fixed updates, batch4 (one crop per training specimen), the existing BCE+Dice loss, normal BN/dropout, no augmentation, identical p1–p99.8 per-crop normalization and threshold0.5. Final checkpoint only; no selection or rerun using held-out outcomes. Ten fits all completed. Pure training-loop time 472.49s (7.87min); setup/evaluation/download are additional. TensorFlow2.21.0 on one Secure Cloud RTX A6000.

## Paired OOF results
Each specimen has eight evaluation anchors, so pooled crop mean equals mean of five specimen means. SD definitions are explicit; crops within a specimen are correlated.

| Metric | Original crop pool | Expanded crop pool |
|---|---:|---:|
| OOF mean Dice | 0.1914 | 0.0908 |
| Across-specimen Dice SD (population) | 0.0267 | 0.0428 |
| Across-crop Dice SD (population) | 0.0668 | 0.0669 |
| Median crop Dice | 0.1901 | 0.0722 |
| Minimum crop Dice | 0.0317 | 0.0000 |
| Maximum crop Dice | 0.3237 | 0.2599 |
| Mean IoU | 0.1073 | 0.0489 |
| Mean precision | 0.2620 | 0.2562 |
| Mean recall | 0.1773 | 0.0653 |
| Mean training-anchor Dice | 0.8372 | 0.5846 |

| Fold | Held-out specimen | Original Dice | Expanded Dice | Change |
|---|---|---:|---:|---:|
| 0 | cell_20260814_23298_NZ_50 | 0.2251 | 0.1535 | -0.0716 |
| 1 | cell_20260814_23299_NZ_200 | 0.2041 | 0.0426 | -0.1615 |
| 2 | cell_20260904_23961_NZ_100 | 0.1971 | 0.1245 | -0.0726 |
| 3 | cell_20260904_23962_NZ_170 | 0.1864 | 0.0489 | -0.1375 |
| 4 | cell_20260904_23965_NZ_140 | 0.1443 | 0.0845 | -0.0598 |

Mean paired Dice change -0.1006; median change -0.1081. Only 2/40 crops improve; **0/5 specimen means improve**. There is only one <=5% foreground anchor: Dice 0.0317→0.0464, too little evidence for a low-foreground subgroup claim. No significance claim or crop-level independence assumption.

## Observed errors and plausible explanation
The representative fixed-first-anchor panels show fragmented, misplaced predictions in the control and markedly sparser predictions after expansion; the complete fivefold gallery preserves all40 comparisons. Recall falls 0.1773→0.0653, while precision is nearly unchanged (0.2620→0.2562). This is increased under-detection, not a useful precision/recall improvement.

Two measured changes matter for interpretation:
- Old anchors average **13.43% positive pixels**; new crops average **6.78%** (medians14.38% and6.06%). Expansion changes foreground density and pseudo-label difficulty as well as diversity. No assertion that new labels are complete or biologically correct.
- At the same1,000 updates, an original anchor is sampled about125times in control (1,000/8 per specimen) but roughly32–33times in expanded (1,000/30–32). Training-anchor Dice falls0.8372→0.5846. Thus insufficient exposure/optimization of the larger pool is plausible; this study does not prove that more training data is intrinsically harmful.

The original-pool model has a large training/held-out gap (0.8372vs0.1914), consistent with weak cross-specimen generalization. The expanded model also generalizes poorly and has poorer training fit. New automatic labels can inherit missed ER and treat it as background. Data composition, optimization budget and label completeness are unresolved contributors, not separately proven causes. Prior matched ER-input control fit the ER-derived targets far better than Fluo input, and the prior alignment audit found no consistent small global shift; neither establishes biological impossibility of Fluo-to-ER prediction.

## Limits and decision
These are previously inspected specimens and ER-derived pseudo-labels, not new independent manual ground truth. Selecting positive-rich crops limits whole-image false-positive assessment. Five specimens—not40 independent biological replicates—underlie the comparison. OOF outcomes are reported after all fits, not used to alter training, checkpoints, thresholds or labels.

**Reject automatic expansion as implemented under this fixed budget.** Keep all checkpoints as experimental diagnostics only; no production promotion and no threshold adjustment. Both arms remain inadequate for reliable ER segmentation.

Next: before paying for another run, audit completeness and Fluo visibility of a fixed sample of new training crops and quantify how much old-anchor exposure was diluted. If another training comparison is justified, preregister an exposure-controlled sampling/convergence design using training-only or inner-validation information. Do not use these five outer-fold scores to select more epochs, reweighting, or thresholds. Do not blindly enlarge the network or run a parameter sweep.

## Artifacts, checks and cleanup
Local root: `outputs/er_diverse_cv_20261030/`.
- `extracted/results/fold0` … `fold4`: initial weights, both final checkpoints, locks, histories, per-image metrics and held-out probability/input/label NPZs.
- `results.tar.gz` and SHA256, `transfer_verified.json`: exit0 and archive verification.
- `review_metrics.json`, `paired_oof.csv`, `data_diagnostics.json`: independently checked metrics and sampling diagnostics.
- `five_fold_overview.png`: one predetermined anchor per fold, columns **Fluo / fixed pseudo-label / original-pool prediction / expanded-pool prediction**. White=positive, black=background. `fold0_all8.png` … `fold4_all8.png` show all anchors.
- Checkpoint SHA256 and ZIP integrity:10/10 PASS; input/label agreement, same paired initialization, source-manifest hash, training/test separation, all1,000-step histories, finite probabilities, and re-derived TP/FP/FN:PASS.
- TensorFlow logged a layout-optimizer warning involving dropout; training proceeded and all finite-value/end-state checks passed. Logs are retained; no science-changing workaround applied.

Pod `[pod-id-redacted]` stopped then **terminated after local verification**. Runpod console confirms GPUs0, vCPUs0, storage0GB, endpoints0, **$0.00/hour**. Displayed balance $4.46→$4.31: approximately **US$0.15**, rounded balance difference rather than an itemized invoice, belowUS$1 cap. `runpod_cleanup.png` is saved proof. No further GPU job, training, TIFF/label modification or public upload was performed.
