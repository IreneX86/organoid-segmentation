"""C2: same as C but train/test restricted to ER-reporter-positive neighbourhoods
(pseudo-label dilated by 15 px, inside tissue). This removes the 'which cells express
the reporter' question and asks: inside cells that DO show ER, can Fluo predict the
ER network detail / pseudo-label layout on an unseen specimen? LOSO only."""
import json, sys, numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from common import SPECIMENS, load_plane, sample_planes, tissue_mask
from c_predictability import features, targets, fit_eval
RNG=np.random.default_rng(7)
def build(s):
    D={k:[] for k in ("fluo","er_adj","fluo_flip","t_fine","t_pl")}
    for z in sample_planes(s):
        E,F,E1,P=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z+1,"er"),load_plane(s,z,"pl")
        m=tissue_mask(F); R=binary_dilation(P>0,disk(15))&m
        R[:24],R[-24:],R[:,:24],R[:,-24:]=False,False,False,False
        ys,xs=np.nonzero(R); k=RNG.choice(len(ys),min(12000,len(ys)),replace=False); ys,xs=ys[k],xs[k]
        tf,_,tp=targets(E,P,m)
        for key,arr in (("fluo",features(F,m)),("er_adj",features(E1,m)),("fluo_flip",features(np.ascontiguousarray(F[:,::-1]),m[:,::-1]))):
            D[key].append(arr[ys,xs])
        D["t_fine"].append(tf[ys,xs]); D["t_pl"].append(tp[ys,xs])
    return {k:np.concatenate(v) for k,v in D.items()}
data={s:build(s) for s in SPECIMENS}; res={}
for s in SPECIMENS:
    tr=[t for t in SPECIMENS if t!=s]; res[s]={}
    for arm in ("fluo","er_adj","fluo_flip"):
        X=np.concatenate([data[t][arm] for t in tr])
        for tg,kind in (("t_fine","reg"),("t_pl","bin")):
            res[s][f"{arm}|{tg}"]=fit_eval(X,np.concatenate([data[t][tg] for t in tr]),data[s][arm],data[s][tg],kind)
    print(s,{k:{m:round(v,3) for m,v in d.items() if v is not None} for k,d in res[s].items()}); sys.stdout.flush()
json.dump(res,open("out/c2_within_expressing.json","w"),indent=1)
