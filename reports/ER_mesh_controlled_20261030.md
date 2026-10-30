# ER mesh-window controlled pilot — 2026-10-30 (Sydney)

## Conclusion

Both fixed 500-update arms completed successfully. Mesh-window sampling improved agreement with partial validation pseudo-labels relative to the matched control, but predictions remain broad, connected white regions rather than resolved ER filaments. REJECT both outputs for production. Do not promote the mesh checkpoint or automatically generate new training labels from it.

## Fixed comparison

Original32 interline crops versus16 mesh windows (4 per training specimen). Same four training specimens, same train-only BN-calibrated source checkpoint, Adam0.0003, batch4 (one crop/specimen), seed20261029,500 updates, frozen BN statistics/affine, dropoutoff,no augmentation. Mesh loss only in64px windows,320px Fluo context. Same8 validation crops, fixed threshold0.5, no checkpoint selection or parameter changes after results. Equal updates do not equal supervised pixels; window selection and supervised support differ together.

Only Fluo enters the network. ER derived labels remain incomplete, provisional pseudo-labels. Validation is8 crops from ONE previously viewed specimen, not eight independent biological replicates or an independent test set.

## Mean known-pixel pseudo-label agreement

| Metric | Original-crop control | Mesh-window arm |
|---|---:|---:|
| dice | 0.6080 | 0.7704 |
| iou | 0.4402 | 0.6294 |
| precision | 0.7629 | 0.7308 |
| recall | 0.5120 | 0.8180 |
| positive_fraction | 0.3812 | 0.6500 |
| ignored_positive_fraction | 0.3634 | 0.6302 |

All8 paired pseudo-Dice changes are positive; mean change+0.1624. This is increased recall accompanied by slightly lower precision and greater positive area, not demonstrated finer ER reconstruction. Positive fraction rises38.12%→65.00%; ignored-positive fraction36.34%→63.02%. No accuracy claim is possible on ignored pixels. Visual inspection of all8 masks shows thicker/sheet-like regions in the mesh arm, including nearly filled regions.

The earlier calibrated source achieved approximately0.8198 pseudo-Dice under the same validation labels; both new fixed-final checkpoints are below that historical reference. That is contextual, not a third equal-training-budget arm.

## Fitting and limitations

GPU training portions: control24.21s,mesh19.64s; including checkpoint/evaluation25.41s and20.48s. Final sampled-batch loss0.05288(control) and0.000117(mesh) is not whole-training-set accuracy. Very low mesh training loss with coarse validation predictions is consistent with memorizing limited supervision, but does not isolate overfitting from label ambiguity or Fluo/ER information mismatch.

The experiment does not prove Fluo lacks all ER information. It shows this small-window partial-supervision scheme does not produce acceptable fine ER masks. Existing large ignored areas, mostly positive supervision, source-label width, and imperfect cross-channel structural correspondence remain unresolved.

## Recommendation

Do not repeat positive-crop expansion or threshold scans. The next informative step is a small, fixed line/inter-line evaluation audit with reliable reviewed pixels in aligned Fluo/ER ROIs, explicitly separating genuinely invisible ER structures from visible boundaries. Keep uncertain pixels ignored; do not turn all ignored pixels into background. Without reliable line/inter-line reference, further pseudo-Dice gains cannot establish real ER fidelity. No additional training launched.

## Runtime and safety

One Secure Cloud RTX A6000 48GB (EU-SE-1),Pod [pod-id-redacted],$0.59GPU+$0.007ephemeral disk/hour. User explicitly authorized alternative GPU. Python3.11,TensorFlow2.21.0,TF build CUDA12.5.1/cuDNN9. cuInit and one-GPU TensorFlow matrix test passed.

Archive SHA256 matched; both checkpoint SHA256 and ZIP integrity passed;16 prediction files have finite320×320 outputs and identical validation inputs/labels across arms. Exit0 and completion marker verified. Original TIFFs,labels and prior models unchanged. No raw ER channel or credentials uploaded.

Pod stopped and terminated only after verified download. Billing confirms GPU0,vCPU0,storage0,endpoints0,spend$0/hour. Displayed balance$4.72→$4.66; approximate cost$0.06 (rounded balance difference, not exact invoice). No delayed deployment subscription created.

## Artifacts

- Local results/checkpoints/archive: `outputs/er_mesh_controlled_gpu_20261030/` (ignored from Git).
- Full comparison: `outputs/er_mesh_controlled_gpu_20261030/all8_control_vs_mesh.png`.
- Small metrics: `reports/ER_mesh_controlled_metrics_20261030.json`.
- Scripts: `experiments/er_mesh_controlled_train.py`, `experiments/er_mesh_controlled_review.py`.
- Protocol: `docs/er_mesh_controlled_protocol_20261029.md`.
- No data/checkpoints/logs pushed.
