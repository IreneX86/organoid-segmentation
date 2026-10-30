"""A2: magnitude-squared coherence per frequency bin (invariant to any fixed linear
filter incl. blur and constant sub-pixel/pixel shift), averaged within bands.
Bias floor estimated by the rotated-ER null. Label-free."""
import json, numpy as np
from common import SPECIMENS, BATCH, load_plane, sample_planes, tissue_mask
from a_spectral_coherence import W, BANDS, period, radius, spec, windows
out={}
for s in SPECIMENS:
    Sxy={k:0 for k in ("FE","EE1","FEnull")}; Sxx={k:0 for k in Sxy}; Syy={k:0 for k in Sxy}; n=0
    for z in sample_planes(s):
        E,F,E1=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z+1,"er")
        m=tissue_mask(F)
        for y,x in windows(m):
            sl=(slice(y,y+W),slice(x,x+W)); a=spec(F[sl]); e=spec(E[sl]); e1=spec(E1[sl]); er=spec(np.rot90(E[sl]).copy())
            for k,(p,q) in {"FE":(a,e),"EE1":(e,e1),"FEnull":(a,er)}.items():
                Sxy[k]=Sxy[k]+p*np.conj(q); Sxx[k]=Sxx[k]+np.abs(p)**2; Syy[k]=Syy[k]+np.abs(q)**2
            n+=1
    res={}
    for k in Sxy:
        C=np.abs(Sxy[k])**2/(Sxx[k]*Syy[k]+1e-12)
        res[k]=[float(C[(period>=lo)&(period<hi)&(radius>0)].mean()) for lo,hi in BANDS]
    out[s]={"batch":BATCH[s],"windows":n,"msc":res}
    print(s,BATCH[s],n); [print("  %-6s"%k," ".join("%.3f"%v for v in res[k])) for k in res]
json.dump({"bands":BANDS,"results":out},open("out/a2_msc.json","w"),indent=1)
