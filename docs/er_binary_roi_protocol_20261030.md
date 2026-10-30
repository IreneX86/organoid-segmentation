# User-defined complete binary ROI pilot — 2026-10-30

The user explicitly states gray is background and yellow is positive ER. In the existing40 selected320×320crops, map oldlabel1→1 and oldlabel0/255→0. Preserve original files and all original positives. Do not apply this mapping outside selected crops/full stacks. Use original full-crop labels, NOT the16window labels that artificially ignore positive context outside64px loss support.

Retain32traincrops from4specimens and8validationcrops from1separate specimen. Convert both by the identical user semantic rule before training, without prediction-based edits. This changes evaluation semantics: old partial-label Dice is NOT comparable. Automatic source-mask positives remain pseudo-labels, not exhaustive independently traced manual reference. Whole validation specimen has been repeatedly examined; results exploratory, not independent generalization.

InputFluo only. Start prior train-onlyBNcalibratedsource; existingUNet, maskedBCE+Dice nowappliedtoALLcroppixels,Adam0.0003,batch4(onecrop/specimen),seed20261029,500fixedupdates,BNfrozenincludingaffine,dropoutoff,noaugmentation,threshold0.5. No selection/earlystopping or thresholdsearch. Save sourcepredictions and finalpredictions on identicalnewbinaryvalidationlabels, plus training predictions for descriptive fit/generalization gap. Do not promote automatically.

One compatible SecureCloudGPU,newcapUS$1; no networkvolume/endpoint/autorecharge. GPUpreflight,10mintrainingguard,downloadSHAverifybeforedelete. No rawERupload,originalsunchanged.
