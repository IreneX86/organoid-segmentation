import numpy as np
from scipy import ndimage as ndi
import os; B=os.environ.get("GOLGI_WORK", ".")
def load(d,p,c,z,man): return (np.load(f"{d}/{p}_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{d}/{p}_prob_{c}.npz")[f"z{z:03d}"])/255.
V7=(f"{B}/golgi_v7/results/output","V7"); V9=(f"{B}/golgi_v9/results/output","V9")
TH=(0.5,0.55,0.6,0.65,0.7); M={"v9":{},"v7+v9":{}}
acc={m:{s:{t:[0,0] for t in TH} for s in ("teacher","manual")} for m in M}
for i in range(18):
    c=f"cell_{16471+i}_NZ_150"; d=np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H,W=[int(v) for v in d["shape"]]
    for pre,sec in (("","teacher"),("man_","manual")):
        if pre+"z" not in d.files: continue
        zs=[int(z) for z in d[pre+"z"]]; n=len(zs); up=lambda k: np.unpackbits(d[pre+k])[:n*H*W].reshape(n,H,W).astype(bool)
        Y,RM=up("y"),up("rem")
        for k,z in enumerate(zs):
            if sec=="teacher" and z not in (71,103): continue
            near=(ndi.distance_transform_edt(~Y[k])<=40)&~RM[k]; yv=Y[k]&near
            p9=load(*V9,c,z,sec=="manual"); p7=load(*V7,c,z,sec=="manual")
            for m,pr in (("v9",p9),("v7+v9",(p7+p9)/2)):
                for t in TH:
                    b=(pr>=t)&near; acc[m][sec][t][0]+=2*(b&yv).sum(); acc[m][sec][t][1]+=b.sum()+yv.sum()
for m in acc:
    for s in acc[m]: print(m,s," ".join(f"{t}:{a/b:.3f}" for t,(a,b) in acc[m][s].items()))
