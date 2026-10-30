"""F: pseudo-label Dice ceiling for a Fluo-only model under reporter mosaicism.
R = tissue within 15 px of any pseudo-label (proxy for reporter-positive cells).
Hypothetical 'oracle' marks the pseudo-label exactly inside R and the SAME density
of ER in reporter-negative tissue (where real ER exists but is unlabelled because
the reporter is absent). Against the pseudo-label its recall=1, precision=|R|/|T|.
This is an arithmetic ceiling under a stated assumption, not a measurement."""
import json, numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from common import SPECIMENS, BATCH, load_plane, sample_planes, tissue_mask
out={}
for s in SPECIMENS:
    fr=[];dice=[]
    for z in sample_planes(s):
        F,P=load_plane(s,z,"fluo"),load_plane(s,z,"pl")>0; T=tissue_mask(F); P&=T
        R=binary_dilation(P,disk(15))&T; f=R.sum()/T.sum(); fr.append(f); dice.append(2*f/(1+f))
    out[s]=dict(batch=BATCH[s],labelled_neighbourhood_frac=float(np.mean(fr)),oracle_dice=float(np.mean(dice)))
    print(s,BATCH[s],round(np.mean(fr),3),round(np.mean(dice),3))
json.dump(out,open("out/f_oracle_bound.json","w"),indent=1)
