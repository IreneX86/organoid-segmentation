import numpy as np
from scipy import ndimage as ndi
import os; B=os.environ.get("GOLGI_WORK", "."); D9=f"{B}/golgi_v9/results/output"; folds=[[f"cell_{16471+i}_NZ_150" for i in range(18)][k::3] for k in range(3)]
fold_of={c:k for k in range(3) for c in folds[k]}
def load(c,z,man): return (np.load(f"{D9}/V9_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{D9}/V9_prob_{c}.npz")[f"z{z:03d}"])/255.
R={s:{"near":[0,0],"full":[0,0],"fold":{k:[0,0] for k in range(3)},"plane":[]} for s in ("teacher","manual")}
for i in range(18):
    c=f"cell_{16471+i}_NZ_150"; d=np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H,W=[int(v) for v in d["shape"]]
    for pre,sec in (("","teacher"),("man_","manual")):
        if pre+"z" not in d.files: continue
        zs=[int(z) for z in d[pre+"z"]]; n=len(zs); up=lambda k: np.unpackbits(d[pre+k])[:n*H*W].reshape(n,H,W).astype(bool)
        Y,RM=up("y"),up("rem")
        for k,z in enumerate(zs):
            if sec=="teacher" and z not in (71,103): continue
            pr=load(c,z,sec=="manual")>=0.6; near=(ndi.distance_transform_edt(~Y[k])<=40)&~RM[k]; y=Y[k]&near; p=pr&near
            a,b=2*(p&y).sum(),p.sum()+y.sum(); r=R[sec]; r["near"][0]+=a; r["near"][1]+=b; r["fold"][fold_of[c]][0]+=a; r["fold"][fold_of[c]][1]+=b
            r["plane"].append(a/max(b,1))
            q=pr&~RM[k]; yy=Y[k]; r["full"][0]+=2*(q&yy).sum(); r["full"][1]+=q.sum()+yy.sum()
for s,r in R.items():
    P=np.array(r["plane"])
    print(s, "planes",len(P),"near %.3f"%(r["near"][0]/r["near"][1]), "full-image %.3f"%(r["full"][0]/r["full"][1]),
          "folds", [round(a/b,3) if b else None for a,b in r["fold"].values()], "per-plane mean %.3f median %.3f min %.3f max %.3f"%(P.mean(),np.median(P),P.min(),P.max()))
