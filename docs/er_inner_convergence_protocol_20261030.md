# ER inner-validation convergence protocol — frozen; completed 2026-10-30

Question: does the expanded training pool benefit from adequate optimization, with stopping chosen without outer-test labels? This is an exploratory study after prior results, not independent confirmatory evidence.

## Frozen design
Five specimens sorted by ID. Fold i reserves specimen i for outer evaluation and specimen (i+1) mod5 for inner validation. Remaining three specimens train. Exact file lists and SHA256 are locked in outputs/er_inner_convergence_20261030/protocol.json. Every inner/outer set has eight old anchors; new crops from either held-out specimen are excluded. Control24trainingcrops; expanded91–94. No changes to existing data.

Both arms start from identical per-fold weights, seed20261031+i, base16U-Net, Adam0.0003, BCE+Dice, existing BN/dropout, noaugmentation, p1–p99.8 normalization, threshold0.5. Batch4 contains one crop per training specimen plus a fourth rotating among the three; this change from four training specimens is shared by both arms. Therefore prior 4-training-specimen scores are context only, not the matched comparison.

Evaluate fixed inner anchors every250updates; candidates are eligible only from step1000. Minimum1,000; maximum4,000updates. A mean inner Dice gain strictly greater than0.002 replaces the selected checkpoint and resets patience. Four evaluations without that gain stop training; ties retain earlier weights. Save step1000 separately. No threshold search. This stopping rule is a pragmatic prespecified choice, not a biological significance cutoff.

Selection runner never loads its fold's outer images/labels. All ten selection decisions must be saved and hashed before any separate outer evaluation. No rerun for poor metrics, no best-fold selection, no production promotion. The outer specimens have been previously inspected, so future OOF remains exploratory regardless of computational isolation.

## Checks and artifacts
experiments/er_inner_convergence_prepare.py freezes lists; experiments/er_inner_convergence.py performs GPU-only selection. Local checks passed: source protocol hash, train/inner/outer specimen isolation, original-pool subset, selection boundary/tie behavior and Python compilation. GPU execution is not yet tested. Original models, TIFFs and labels remain unchanged. Newly generated labels are still pseudo-labels and may miss real ER.

## Resource plan
One reliable Secure Cloud GPU; no endpoint/network volume. Maximum40,000updates, roughly32minutes of training extrapolated from prior A6000 measurements, plus setup/evaluation/transfer. Approximate expected bill US$0.4–0.7 at the prior ~$0.597/hour rate, not a live quote; recheck actual price before provisioning. Proposed new ceiling US$1, with external wall-clock guard and download reserve. Download/hash-check all checkpoints and decisions before deleting Pod, then confirm zero ongoing spend. This study has not started; previous run's US$1 ceiling was specific to that completed study and is not silently reused.

## Execution authorization
User explicitly approved the proposed new US$1 ceiling. One Secure A6000 Pod [pod-id-redacted] created2026-10-30 around05:05UTC at ~$0.597/hour. Hard cleanup06:25UTC; no secondPod. Locked scientific protocol unchanged.

## Completion (protocol above retained)
All ten models completed; locked decisions preceded outer evaluation. Control/expanded mean Dice0.2072/0.1753, no production promotion. Twenty checkpoints and results verified locally before Pod deletion; zero ongoing spend, approximate cost$0.24. See [completed report](../reports/ER_inner_convergence_20261030.md). Earlier prepared/not-started statements describe pre-execution status only.
