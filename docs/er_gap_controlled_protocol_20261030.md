# Fixed mesh-gap supervision ablation — 2026-10-30

Control:16 original mesh windows. Candidate:identical16windows plus1832 provisional enclosed-gap negative pixels. Original positive labels and existing negatives unchanged; only ignored→negative pixels can differ. Both arms use identical Fluo inputs,normalization,crop order,source checkpoint,sampling seed20261029,500updates,batch4(one crop/specimen),Adam0.0003,frozenBNincludingaffine,dropoutoff,noaugmentation,maskedBCE+Dice. No ER channel input.

Eight original validation crops from one previously viewed specimen remain fixed. Final-step checkpoint only,threshold0.5,no model/threshold selection. Report allpairedcrops and partialpseudoDice/IoU/P/R,positivearea/ignoredpositivearea and visible broad masks. No independent accuracy/generalization claim. No retraining or parameter sweep based on outcomes. Candidate weak negatives may include faint ER; improvement in pseudoDice alone cannot approve production.

Use one Secure Cloud compatible GPU at a time, authorized alternative to A40, total new spending capUS$1. Upload only croppedFluo/partiallabels/code/sourcecheckpoint,no rawER orcredentials. Perarm10min guard and fulltraining1300seconds. Download/hashverify both checkpoints and all results before terminate; no volumes/endpoints/autorecharge.
