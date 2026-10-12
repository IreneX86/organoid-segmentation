# LD binary label definition (locked before model training)

**Status:** label rule selected from manual-mask encoding and spatial QC, before
any LD model training or outer-fold evaluation. This rule is not selected by
cross-validation Dice and must not be changed after seeing model results.

## Source and scientific question

The 36 same-name, single-channel `LD_input`/`LD_output` TIFF pairs are
1200×1200 `uint16`. The annotator confirmed that Photoshop generated gray
pixels through antialiasing. Thirty-three masks contain multiple gray levels
and approach the 16-bit white level; three masks (`ld_003`, `ld_006`,
`ld_021`) contain only 0/255 despite their `uint16` storage. The task is
binary **semantic** LD coverage, not a distinct ID for each droplet.

## Fixed conversion

For each source mask with zero-valued background and observed maximum `M`,
set derived foreground to 1 where `source >= ceil(M / 2)` and background to 0
elsewhere. Save a separate same-name `uint8` 0/1 TIFF. Require `M=255` or
`60000 <= M <= 65535`; any other scale fails for manual review. Preserve the
original TIFFs exactly. The conversion script records both source-file SHA256
hashes, `M`, the inclusive threshold, source and derived pixel counts, and
the derived-file SHA256 for every image.

The midpoint is an **operational 50% coverage convention** for antialiased
edges, not a claim that every partially covered pixel is biologically
background. Per-file scaling is necessary because a fixed 32768 cutoff would
erase the three 0/255 masks. The rule is applied equally to every image,
regardless of CV fold or future prediction quality.

## Pre-training sensitivity and tiny-object check

The following diagnostic rules were compared on the 36 manual masks **before**
training. None was selected using model predictions or held-out metrics.

| Rule | Foreground pixels across 36 masks |
| --- | ---: |
| Any nonzero value | 107,730 |
| At least 25% of each mask's `M` | 104,278 |
| **At least 50% of each mask's `M` (selected)** | **102,605** |
| At least 75% of each mask's `M` | 100,977 |

Relative to the nonzero rule, the midpoint removes 5,125 faint pixels
(4.76%). Of these, 97.37% are within one pixel and 99.63% within two pixels
of retained foreground, consistent with a narrow antialiased fringe.
Only eight nonzero 8-connected components disappear entirely: 12 pixels in
total, largest four pixels, occurring in `ld_010`, `ld_013`, `ld_022`, and
`ld_023`. Local 48-pixel crop overlays were reviewed; these appear as isolated
or edge-adjacent faint pixels rather than clearly resolved, complete droplets,
but their biological status cannot be proven from the images alone. They are
flagged for later expert review. No manual mask was changed or selectively
patched. A stricter 95% rule would leave only 91,064 pixels and can remove
small gray-only regions; it was rejected on label-preservation grounds.

Under the selected convention, median image foreground fraction is 0.1674%
(range 0.00729%–0.4846%). This extreme sparsity motivates full-image
validation and explicit small-object/false-positive analysis after baseline.

Representative input/mask/contour overlays are kept **locally outside Git**
at `outputs/ld_label_qc/`; raw biological pixels are not included in this
repository. The aggregate audit is in
[`reports/ld_label_audit_20261012.md`](../reports/ld_label_audit_20261012.md).

## Scientific and reproducibility limits

- This conversion resolves file-encoding differences, not biological
  uncertainty in the original annotations.
- Model performance must never be used to revise this label rule on the same
  five outer folds. A future annotation revision requires a separately
  versioned dataset and evaluation plan.
- All 36 pairs are development data. No independent LD external ground truth
  is currently available.

Source on antialiasing behavior: [Adobe Photoshop selection-edge
documentation](https://helpx.adobe.com/photoshop/desktop/make-selections/refine-modify-selections/refine-and-soften-selection-edges.html).
