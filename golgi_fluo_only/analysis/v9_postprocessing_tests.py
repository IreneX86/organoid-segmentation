"""Free post-processing tests on v9 probabilities (no retraining)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
import os; B=os.environ.get("GOLGI_WORK", "."); D9=f"{B}/golgi_v9/results/output"
def load(c,z,man): return (np.load(f"{D9}/V9_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{D9}/V9_prob_{c}.npz")[f"z{z:03d}"])/255.
def relmax(pr,lo,frac):
    seed=pr>=lo; L,n=ndi.label(seed,np.ones((3,3)))
    if n==0: return seed
    mx=ndi.maximum(pr,L,np.arange(1,n+1)); thr=np.r_[1,mx*frac][L]; return seed&(pr>=thr)
M={"thr0.6":lambda p:p>=0.6}
for lo in (0.35,0.4,0.45,0.5):
    for hi in (0.6,0.7):
        M[f"hyst{lo}-{hi}"]=lambda p,lo=lo,hi=hi: apply_hysteresis_threshold(p,lo,hi)
for lo in (0.35,0.4,0.45):
    for fr in (0.7,0.8,0.85):
        M[f"rel{lo}x{fr}"]=lambda p,lo=lo,fr=fr: relmax(p,lo,fr)
for s in (1.0,):
    M["smooth1+0.6"]=lambda p: ndi.gaussian_filter(p,1)>=0.6
acc={m:{s:[0,0] for s in ("teacher","manual")} for m in M}
for i in range(18):
    c=f"cell_{16471+i}_NZ_150"; d=np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H,W=[int(v) for v in d["shape"]]
    for pre,sec in (("","teacher"),("man_","manual")):
        if pre+"z" not in d.files: continue
        zs=[int(z) for z in d[pre+"z"]]; n=len(zs); up=lambda k: np.unpackbits(d[pre+k])[:n*H*W].reshape(n,H,W).astype(bool)
        Y,RM=up("y"),up("rem")
        for k,z in enumerate(zs):
            if sec=="teacher" and z not in (71,103): continue
            near=(ndi.distance_transform_edt(~Y[k])<=40)&~RM[k]; y=Y[k]&near; pr=load(c,z,sec=="manual")
            for m,f in M.items():
                q=f(pr)&near; acc[m][sec][0]+=2*(q&y).sum(); acc[m][sec][1]+=q.sum()+y.sum()
res=sorted(((a["teacher"][0]/a["teacher"][1],a["manual"][0]/a["manual"][1],m) for m,a in acc.items()),reverse=True)
for t,mn,m in res[:10]: print(f"{m:18s} teacher {t:.3f}  manual {mn:.3f}")
print("baseline", [r for r in res if r[2]=="thr0.6"])
