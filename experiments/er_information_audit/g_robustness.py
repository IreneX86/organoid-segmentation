"""G: robustness of C2 headline (Fluo -> ER fine detail, LOSO, inside reporter-positive
neighbourhoods) to neighbourhood radius (8, 30 px) and sampling seed. Fluo arm only."""
import json, sys, numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from common import SPECIMENS, load_plane, sample_planes, tissue_mask
from c_predictability import features, targets, fit_eval
res={}
for rad,seed in ((8,11),(30,12)):
    rng=np.random.default_rng(seed); data={}
    for s in SPECIMENS:
        X=[];Y=[]
        for z in sample_planes(s):
            E,F,P=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z,"pl")
            m=tissue_mask(F); R=binary_dilation(P>0,disk(rad))&m; R[:24]=R[-24:]=False; R[:,:24]=R[:,-24:]=False
            ys,xs=np.nonzero(R); k=rng.choice(len(ys),min(8000,len(ys)),replace=False)
            X.append(features(F,m)[ys[k],xs[k]]); Y.append(targets(E,P,m)[0][ys[k],xs[k]])
        data[s]=(np.concatenate(X),np.concatenate(Y))
    res[f"r{rad}"]={s:fit_eval(np.concatenate([data[t][0] for t in SPECIMENS if t!=s]),np.concatenate([data[t][1] for t in SPECIMENS if t!=s]),*data[s],"reg")["r"] for s in SPECIMENS}
    print(rad,{k:round(v,3) for k,v in res[f"r{rad}"].items()}); sys.stdout.flush()
json.dump(res,open("out/g_robustness.json","w"),indent=1)
