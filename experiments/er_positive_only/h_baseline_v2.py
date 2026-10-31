"""H: CPU reference baseline on dataset_v2. Fluo-only multiscale features + gradient
boosting, leave-one-specimen-out, train AND score only inside ER_positive_region.
Metric: AUC / average precision of ER_mask_v2 (pseudo-label) vs prevalence, plus
F1 within 2 px tolerance at the training-set-chosen threshold. Not GT accuracy."""
import json, numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from c_predictability import features
from common import tissue_mask
S=["23298","23300","23961","23963","23965","23966"]; rng=np.random.default_rng(31)
D={}
for s in S:
    d=np.load(f"dataset_v2/{s}.npz"); X=[];Y=[];full=[]
    for k in range(len(d["z"])):
        F=d["fluo"][k].astype(np.float32); R=d["region"][k]>0; M=d["mask"][k]>0
        Fm=features(F,tissue_mask(F)|R); ys,xs=np.nonzero(R)
        pick=rng.choice(len(ys),min(4000,len(ys)),replace=False); X.append(Fm[ys[pick],xs[pick]]); Y.append(M[ys[pick],xs[pick]])
        if k in (len(d["z"])//3, 2*len(d["z"])//3): full.append((Fm,R,M))
    D[s]=(np.concatenate(X),np.concatenate(Y).astype(np.uint8),full); print("built",s,flush=True)
res={}
for s in S:
    Xtr=np.concatenate([D[t][0] for t in S if t!=s]); ytr=np.concatenate([D[t][1] for t in S if t!=s])
    m=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,random_state=0).fit(Xtr,ytr)
    ptr=m.predict_proba(Xtr)[:,1]; ths=np.linspace(.05,.6,23)
    thr=float(ths[np.argmax([2*((ptr>=t)&(ytr==1)).sum()/((ptr>=t).sum()+ytr.sum()) for t in ths])])
    p=m.predict_proba(D[s][0])[:,1]; y=D[s][1]
    f1=[]
    for Fm,R,M in D[s][2]:
        P=np.zeros(R.shape,bool); P[R]=m.predict_proba(Fm[R])[:,1]>=thr
        tp_p=(P&binary_dilation(M,disk(2))).sum()/max(P.sum(),1); tp_r=(M&R&binary_dilation(P,disk(2))).sum()/max((M&R).sum(),1)
        f1.append(2*tp_p*tp_r/max(tp_p+tp_r,1e-9))
    res[s]=dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)),prev=float(y.mean()),thr=thr,f1_tol2=float(np.mean(f1)))
    print(s,{k:round(v,3) for k,v in res[s].items()},flush=True)
res["mean"]={k:float(np.mean([res[s][k] for s in S])) for k in ("auc","ap","prev","f1_tol2")}; print("mean",res["mean"])
json.dump(res,open("out/h_baseline_v2.json","w"),indent=1)
