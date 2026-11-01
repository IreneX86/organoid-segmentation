import sys, json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from np_unet import load_state_dict, predict
def norm(f):
    lo,hi=np.percentile(f,[1,99.8]); return np.clip((f.astype(np.float32)-lo)/max(hi-lo,1e-6),0,1.5).astype(np.float32)
s=sys.argv[1]; d=np.load(f"dataset_v2/{s}.npz"); sd=load_state_dict(f"runs_dl/runs_v2/unet_heldout_{s}.pt")
P=[];Y=[]
for k in range(len(d["z"])):
    p=predict(sd,norm(d["fluo"][k])); R=d["region"][k]>0; P.append(p[R]); Y.append(d["mask"][k][R]>0)
P,Y=np.concatenate(P),np.concatenate(Y)
ref=json.load(open("runs_dl/runs_v2/results.json"))["results"][s]["heldout"]
print(s,"numpy AUC %.4f AP %.4f | GPU AUC %.4f AP %.4f"%(roc_auc_score(Y,P),average_precision_score(Y,P),ref["auc"],ref["ap"]))
