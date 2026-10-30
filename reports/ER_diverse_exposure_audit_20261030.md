# ER expanded-crop exposure and label-visibility audit — 2026-10-30

Read-only follow-up to the failed paired fivefold study. No GPU, inference, training, relabelling or source changes. The audit uses frozen training manifests, sampler seeds and training histories; it does not select cases by outer-test prediction quality.

## Confirmed findings
The exact NumPy sampler was replayed for each fold/arm, using the recorded per-specimen groups and seeds. Each arm drew 4,000 crops total (1,000 batches of four).

- Original anchors were seen a mean125times each in control, versus32.53times in expanded. Expanded individual anchors received18–45visits depending fold and random sampling.
- Mean sampled positive pixel fraction fell13.42%→8.49%. The earlier6.78% figure describes NEW crops alone, not the actual mixed training batches.
- Expanded average logged training loss fell1.1195 at step500 to0.7179 at step1000; control0.6845→0.3891. These are noisy sampled-batch values, not a fixed-set convergence measure. They support investigating optimization exposure; they do not prove a longer run would generalize better.
- Mean per-crop normalized Fluo positive-minus-background contrast is0.0736 in old anchors and0.1219 in new candidates. This statistic does NOT support a blanket explanation that new labelled positives are dimmer. It is not SNR, semantic correspondence, or a measure of label correctness.

## Visual audit
`five_specimen_label_review.png` displays ten fixed examples: median-positive-density old and new crop per specimen, chosen without any student predictions or scores. Rows follow sorted specimen order. Leftthree columns:oldFluo/ER/ER+label; rightthree:newFluo/ER/ER+label. Yellow is the unchanged positive target.

New examples contain more exterior/dark area and sparse or partial structures. Some visible ER texture lies outside the mask in both old and new examples. Different Fluo structures have unequal ER-channel support. This shows a potential completeness/semantic issue, but without trusted exhaustive masks it cannot establish exactly which negative pixels are false negatives. We did not automatically convert them to positives or mark arbitrary Fluo texture as ER. Original masks remain unchanged.

## Conclusion and next experiment design
The failed fixed-compute expansion confounds increased diversity with fewer repetitions of each original anchor and a changed foreground distribution. It cannot demonstrate that additional data are intrinsically harmful. It also does not demonstrate that longer training will fix the much larger cross-specimen gap.

The next informative TRAINING question is whether the expanded pool reaches useful training/inner-validation convergence when exposure is sufficient, with architecture/loss/normalization/threshold unchanged. Choose stopping only using an inner specimen held out from the outer-training specimens, with a fixed maximum step budget; freeze that procedure before outer evaluation. Keep a matched original-pool control under the same procedure. Previously inspected outer specimens remain exploratory, never a fresh independent test. Do not select step count by these existing outer scores. Do not simultaneously add class weights, alter sampling or relabel candidates.

Before executing that study, document an inner split and fixed stopping rule, and inspect candidate-label completeness within training data. No new training has been launched by this audit. Prior partial-supervision and threshold experiments are not silently repeated. No production promotion.

## Reproducibility
Script `experiments/er_diverse_exposure_audit.py`; output `outputs/er_diverse_exposure_audit_20261030/` contains audit.json, all154crop_stats.csv, all10sampling_replay.csv, and five_specimen_label_review.png. Inputs are existing immutable manifests/crops and completed training histories. This audit certifies sampling arithmetic, not biological label accuracy. Incremental GPU cost US$0.
