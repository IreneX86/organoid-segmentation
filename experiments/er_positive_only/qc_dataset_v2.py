import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "er_information_audit"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from common import load_plane, tissue_mask, robust_norm, BATCH
S=["23298","23300","23961","23963","23965","23966"]
fig,ax=plt.subplots(len(S),3,figsize=(13.5,4.5*len(S)))
for i,s in enumerate(S):
    d=np.load(f"dataset_v2/{s}.npz"); k=len(d["z"])//2; z=int(d["z"][k]); M=d["mask"][k]>0; R=d["region"][k]>0; F=d["fluo"][k].astype(np.float32)
    E=load_plane(s,z,"er"); T=tissue_mask(F); e=robust_norm(np.log1p(E),T); f=robust_norm(F,T)
    edge=R&binary_dilation(~R,disk(2))
    o=np.dstack([e,e,e]).clip(0,1); o[M]=[1,.2,.6]; o[edge]=[0,.8,1]
    of=np.dstack([f,f,f]).clip(0,1); of[edge]=[0,.8,1]
    for j,(im,t) in enumerate([(e,"ER.tif (log)"),(o,"ER mask v2 + positive region"),(of,"Fluo (model input) + positive region")]):
        ax[i,j].imshow(im,cmap="gray" if im.ndim==2 else None); ax[i,j].set_xticks([]); ax[i,j].set_yticks([]); ax[i,j].set_title(f"{s} ({BATCH[s]}) z{z} — {t}",fontsize=10)
plt.tight_layout(); plt.savefig("dataset_v2/qc_overview.png",dpi=55); print("ok")
