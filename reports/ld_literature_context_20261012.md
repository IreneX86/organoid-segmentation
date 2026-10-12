# LD segmentation: literature context

This note records evidence considered before LD model selection. The project
uses 36 separate 2D fluorescence TIFF inputs with sparse, manually
annotated lipid-droplet foreground. Evidence from another imaging modality or
dataset is a reason to test an idea, not evidence that it will improve this
dataset's out-of-fold (OOF) performance.

| Primary source | What it supports | Limit and implication here |
| --- | --- | --- |
| [Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)](https://arxiv.org/abs/1505.04597) | U-Net combines local detail and broader context; augmentation helps when annotated biomedical images are scarce. Its overlap-tile strategy permits full-resolution segmentation of images larger than GPU memory. | The paper did not evaluate this LD fluorescence dataset. Start with an independent 2D U-Net and preserve each TIFF's pixel grid; it does not establish an expected Dice score or justify a larger architecture. |
| [Dejgaard and Presley, *New automated single-cell technique for segmentation and quantitation of lipid droplets* (2014)](https://pubmed.ncbi.nlm.nih.gov/25246387/) | In fluorescence microscopy, very small LDs can appear as diffraction-limited spots, while clustered LDs complicate separation. The authors note that watershed can be sensitive to noise and that smoothing may sacrifice resolution. | Their quantification method is not a validation of this U-Net. Audit small-droplet false negatives and clustered regions; do not discard tiny predicted components or apply watershed by default. Semantic coverage and individual-droplet counting are different endpoints. |
| [Sudre et al., *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations* (2017)](https://arxiv.org/abs/1707.03237) | Rare foreground can make segmentation optimization difficult. Loss behavior depends on imbalance, learning rate, and sampled patches. | The study is not LD-specific and does not show that generalized Dice will outperform BCE+Dice here. Keep the baseline fixed; test a loss or sampling change only as a separately declared, controlled experiment using training/validation data for selection. |
| [Sheneman et al., *Deep learning classification of lipid droplets in quantitative phase images* (2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0249196) | A U-Net can produce useful LD semantic masks, but severe imbalance can also lead to all-background predictions; qualitative mask quality may matter alongside overlap scores. | Quantitative-phase imaging differs from this project's fluorescence images, and the study used a different dataset and annotation process. Monitor foreground fraction, empty predictions, precision, and recall; do not transfer its scores, thresholds, or model weights. |
| [Maier-Hein et al., *Metrics Reloaded: recommendations for image analysis validation* (2024)](https://www.nature.com/articles/s41592-023-02151-z) | Metric choice should reflect the biological endpoint. Dice/IoU can be unstable or incomplete for small structures; non-independent images can make aggregate results misleading. | Report per-image five-fold OOF Dice, IoU, precision, and recall, plus foreground- and droplet-size-stratified error checks. Keep suspected near-duplicate views in one fold. Object-level counting metrics require an explicitly defined object endpoint and annotation convention. |

No primary study found here establishes a universal cutoff for converting
Photoshop antialiased grayscale LD masks into biological ground truth. The
project's midpoint conversion is therefore a **pre-model operational label
definition**, supported by source-mask encoding and local boundary/tiny-object
quality checks, not by model performance. Preserve original annotations and
freeze the derived-label rule before CV. Any later change needs a new label
version and evaluation plan; the same outer-test scores must not be used to
select a new label threshold.

A more complex architecture would be warranted only after the baseline error
analysis identifies a repeated, biologically relevant failure mode across
multiple images/folds, the change has a specific mechanism to address it, and
a fair paired CV evaluation shows stable improvement without a substantial
precision, recall, or variance regression. A single weak fold or training-set
Dice is not sufficient evidence.
