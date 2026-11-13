"""Are the label objects v9 misses visible in Fluo? Fluo contrast = mean inside / mean of 3-8 px ring."""
import numpy as np
from scipy import ndimage as ndi
import os; B=os.environ.get("GOLGI_WORK", "."); D9=f"{B}/golgi_v9/results/output"
def load(c,z,man): return (np.load(f"{D9}/V9_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{D9}/V9_prob_{c}.npz")[f"z{z:03d}"])/255.
R={s:{"hit":[],"miss":[],"tpobj":[],"fpobj":[],"pmax_miss":[],"cover":[]} for s in ("teacher","manual")}
for i in range(18):
    c=f"cell_{16471+i}_NZ_150"; d=np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H,W=[int(v) for v in d["shape"]]
    for pre,sec in (("","teacher"),("man_","manual")):
        if pre+"z" not in d.files: continue
        zs=[int(z) for z in d[pre+"z"]]; n=len(zs); up=lambda k: np.unpackbits(d[pre+k])[:n*H*W].reshape(n,H,W).astype(bool)
        Y,RM=up("y"),up("rem")
        for k,z in enumerate(zs):
            if sec=="teacher" and z not in (71,103): continue
            f=d[pre+"fluo8"][k,3].astype(np.float32); near=(ndi.distance_transform_edt(~Y[k])<=40)&~RM[k]; y=Y[k]&near
            pr=load(c,z,sec=="manual"); p=(pr>=0.6)&near
            def con(m):
                ring=(ndi.binary_dilation(m,iterations=8)&~ndi.binary_dilation(m,iterations=3))&~Y[k]
                return f[m].mean()/max(f[ring].mean(),1) if ring.any() else np.nan
            Ly,ny=ndi.label(y,np.ones((3,3)))
            for j,sl in enumerate(ndi.find_objects(Ly),1):
                sl2=tuple(slice(max(s.start-12,0),s.stop+12) for s in sl); m=np.zeros_like(y); m[sl2]=Ly[sl2]==j
                if m.sum()<10: continue
                key="hit" if (m&p).any() else "miss"; R[sec][key].append((con(m),m.sum()))
                if key=="miss": R[sec]["pmax_miss"].append(pr[m].max())
                else: R[sec]["cover"].append((m&p).sum()/m.sum())
            Lp,npd=ndi.label(p,np.ones((3,3)))
            for j,sl in enumerate(ndi.find_objects(Lp),1):
                sl2=tuple(slice(max(s.start-12,0),s.stop+12) for s in sl); m=np.zeros_like(p); m[sl2]=Lp[sl2]==j
                if m.sum()<10: continue
                R[sec]["tpobj" if (m&y).any() else "fpobj"].append((con(m),m.sum()))
for sec,r in R.items():
    print("==",sec)
    for key in ("hit","miss","tpobj","fpobj"):
        a=np.array(r[key]); v=a[~np.isnan(a[:,0])]
        print(f"  {key:6s} n={len(a):4d}  median area {np.median(a[:,1]):5.0f}  Fluo contrast median {np.median(v[:,0]):.2f}  frac contrast<1.1 {np.mean(v[:,0]<1.1):.0%}")
    pm=np.array(r["pmax_miss"]); print(f"  missed objects: max prob median {np.median(pm):.2f}; >0.4: {np.mean(pm>0.4):.0%}; >0.5: {np.mean(pm>0.5):.0%}")
    print(f"  hit objects: median fraction covered {np.median(r['cover']):.2f}")
