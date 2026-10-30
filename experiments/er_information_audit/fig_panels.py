"""Fixed-rule visual panel: middle sampled plane, 320px crop centred on tissue centroid.
Columns: Fluo | ER | ER + pseudo-label (yellow) | Fluo fine detail | ER fine detail.
No selection by any model output."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, center_of_mass
from common import SPECIMENS, BATCH, load_plane, sample_planes, tissue_mask, robust_norm
C=320; fig,ax=plt.subplots(len(SPECIMENS),5,figsize=(15,3.1*len(SPECIMENS)))
for i,s in enumerate(SPECIMENS):
    z=sample_planes(s)[3]; E,F,P=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z,"pl")>0
    m=tissue_mask(F); cy,cx=map(int,center_of_mass(m)); cy=min(max(cy,C//2),1200-C//2); cx=min(max(cx,C//2),1200-C//2)
    sl=(slice(cy-C//2,cy+C//2),slice(cx-C//2,cx+C//2))
    f,e=robust_norm(F,m)[sl],robust_norm(E,m)[sl]
    hp=lambda x:gaussian_filter(x,1)-gaussian_filter(x,4)
    ov=np.dstack([e,e,e]).clip(0,1); ov[P[sl]]=[1,0.85,0]
    for j,(im,t) in enumerate([(f,"Fluo (deploy input)"),(e,"ER channel (train only)"),(ov,"ER + pseudo-label"),(hp(f),"Fluo detail 1-4px"),(hp(e),"ER detail 1-4px")]):
        a=ax[i,j]; a.imshow(im,cmap=None if im.ndim==3 else "gray",vmin=None if j<2 else (np.percentile(im,1) if j>2 else None),vmax=None if j<2 else (np.percentile(im,99) if j>2 else None)); a.set_xticks([]); a.set_yticks([])
        if i==0: a.set_title(t,fontsize=11)
    ax[i,0].set_ylabel(f"{s} ({BATCH[s]}) z{z}",fontsize=10)
plt.tight_layout(); plt.savefig("out/fig_panels.png",dpi=110); print("ok")
