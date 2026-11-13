"""Where does v9 lose Dice? (held-out planes, within 40 px of labels, threshold 0.6)"""
import numpy as np
from scipy import ndimage as ndi
import os; B=os.environ.get("GOLGI_WORK", "."); D9=f"{B}/golgi_v9/results/output"
def load(c,z,man): return (np.load(f"{D9}/V9_manual_prob_{c}_z{z:03d}.npy") if man else np.load(f"{D9}/V9_prob_{c}.npz")[f"z{z:03d}"])/255.
S={s:dict(tp=0,fp=0,fn=0,fp_obj=0,fn_obj=0,nlab=0,nhit=0,npred=0,nfp=0,dil={k:[0,0] for k in (-2,-1,1,2)},orac=[0,0],plane=[]) for s in ("teacher","manual")}
TH=np.arange(0.3,0.86,0.05)
for i in range(18):
    c=f"cell_{16471+i}_NZ_150"; d=np.load(f"{B}/golgi_v5/payload/data/{c}.npz"); H,W=[int(v) for v in d["shape"]]
    for pre,sec in (("","teacher"),("man_","manual")):
        if pre+"z" not in d.files: continue
        zs=[int(z) for z in d[pre+"z"]]; n=len(zs); up=lambda k: np.unpackbits(d[pre+k])[:n*H*W].reshape(n,H,W).astype(bool)
        Y,RM=up("y"),up("rem")
        for k,z in enumerate(zs):
            if sec=="teacher" and z not in (71,103): continue
            near=(ndi.distance_transform_edt(~Y[k])<=40)&~RM[k]; y=Y[k]&near; pr=load(c,z,sec=="manual"); p=(pr>=0.6)&near; s=S[sec]
            s["tp"]+=(p&y).sum(); s["fp"]+=(p&~y).sum(); s["fn"]+=(y&~p).sum()
            Ly,ny=ndi.label(y,np.ones((3,3))); Lp,npd=ndi.label(p,np.ones((3,3)))
            hit=np.zeros(ny+1,bool); hit[np.unique(Ly[p&y])]=True; hit[0]=False
            s["nlab"]+=ny; s["nhit"]+=hit[1:].sum(); s["fn_obj"]+=(y&~hit[Ly]).sum()
            ph=np.zeros(npd+1,bool); ph[np.unique(Lp[p&y])]=True; ph[0]=False
            s["npred"]+=npd; s["nfp"]+=(~ph[1:]).sum(); s["fp_obj"]+=(p&~ph[Lp]).sum()
            for kk in s["dil"]:
                q=(ndi.binary_dilation(p,iterations=kk) if kk>0 else ndi.binary_erosion(p,iterations=-kk))&near
                s["dil"][kk][0]+=2*(q&y).sum(); s["dil"][kk][1]+=q.sum()+y.sum()
            q=p&ph[Lp]; s["orac"][0]+=2*(q&y).sum(); s["orac"][1]+=q.sum()+y.sum()   # oracle: drop pred objects that touch no label
            s["plane"].append([(2*(((pr>=t)&near)&y).sum(), ((pr>=t)&near).sum()+y.sum()) for t in TH])
for sec,s in S.items():
    tp,fp,fn=s["tp"],s["fp"],s["fn"]; dice=2*tp/(2*tp+fp+fn)
    P=np.array(s["plane"]); best=P[np.arange(len(P)),np.argmax(P[:,:,0]/np.maximum(P[:,:,1],1),1)].sum(0)
    print(f"== {sec}: Dice {dice:.3f}  precision {tp/(tp+fp):.3f}  recall {tp/(tp+fn):.3f}")
    print(f"   FP px {fp}, of which in objects touching no label {s['fp_obj']/fp:.0%}; FN px {fn}, of which in completely missed objects {s['fn_obj']/fn:.0%}")
    print(f"   label objects hit {s['nhit']}/{s['nlab']} ({s['nhit']/s['nlab']:.0%}); pred objects touching no label {s['nfp']}/{s['npred']}")
    print("   erode/dilate pred:", {k:round(a/b,3) for k,(a,b) in s["dil"].items()})
    print(f"   oracle remove FP objects -> Dice {s['orac'][0]/s['orac'][1]:.3f};  oracle per-plane threshold -> Dice {best[0]/best[1]:.3f}")
    # if all FN boundary pixels fixed but missed objects stay missed
    print(f"   oracle add back all completely missed objects -> Dice {2*(tp+s['fn_obj'])/(2*(tp+s['fn_obj'])+fp+fn-s['fn_obj']):.3f}")
