"""Example predictions: each specimen predicted by the model that never saw it.
Fixed choices: middle plane of that specimen's dataset; zoom = 320px window with the
largest ER-positive-region fraction (chosen from labels, not from prediction quality)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, binary_dilation
from skimage.morphology import disk
from sklearn.metrics import roc_auc_score, average_precision_score
from np_unet import load_state_dict, predict
from common import load_plane, BATCH
def norm(f):
    lo,hi=np.percentile(f,[1,99.8]); return np.clip((f.astype(np.float32)-lo)/max(hi-lo,1e-6),0,1.5).astype(np.float32)
def disp(x,lo=1,hi=99.5):
    a,b=np.percentile(x,[lo,hi]); return np.clip((x-a)/max(b-a,1e-6),0,1)
S=["23298","23300","23961","23963","23965","23966"]
fig,ax=plt.subplots(len(S),6,figsize=(27,4.6*len(S)))
for i,s in enumerate(S):
    d=np.load(f"dataset_v2/{s}.npz"); k=len(d["z"])//2; z=int(d["z"][k])
    F=d["fluo"][k].astype(np.float32); M=d["mask"][k]>0; R=d["region"][k]>0
    E=load_plane(s,z,"er"); sd=load_state_dict(f"runs_dl/runs_v2/unet_heldout_{s}.pt"); p=predict(sd,norm(F))
    auc=roc_auc_score(M[R],p[R]); ap=average_precision_score(M[R],p[R]); prev=M[R].mean()
    fr=uniform_filter(R.astype(float),320); fr[:160]=fr[-160:]=0; fr[:,:160]=fr[:,-160:]=0
    cy,cx=np.unravel_index(np.argmax(fr),fr.shape); sl=(slice(cy-160,cy+160),slice(cx-160,cx+160))
    edge=R&binary_dilation(~R,disk(2))
    def ov(base,mask,col):
        o=np.dstack([base]*3); o[mask]=col; return o
    eb=disp(np.log1p(E)); full=ov(disp(p,0,100),edge,[0,.8,1])
    panels=[(disp(F)[sl],f"{s} ({BATCH[s]}) z{z}\nFluo (only model input)"),
            (disp(E)[sl],"ER.tif (not given to model)"),
            (ov(disp(E)[sl]*.7,M[sl],[1,.2,.6]),"target: ER mask from ER.tif"),
            (p[sl],"U-Net probability from Fluo"),
            (ov(disp(F)[sl]*.7,(p>=.5)[sl],[0,1,.3]),"U-Net mask (p≥0.5) on Fluo"),
            (full,f"whole image: U-Net probability\n(cyan = ER-positive cells)  AUC {auc:.2f}  AP {ap:.2f} (prev {prev:.2f})")]
    for j,(im,t) in enumerate(panels):
        a=ax[i,j]; a.imshow(im,cmap="gray" if im.ndim==2 else None,vmin=0,vmax=1 if j==3 else None); a.set_xticks([]); a.set_yticks([]); a.set_title(t,fontsize=12)
    ax[i,5].add_patch(plt.Rectangle((cx-160,cy-160),320,320,fill=False,ec="yellow",lw=1.5))
    print(s,z,round(auc,3),round(ap,3),round(prev,3),flush=True)
plt.tight_layout(); plt.savefig("unet_examples_heldout.png",dpi=55)
