"""Build the reporter-positive-only training set (CPU). For each specimen take every
5th plane in the central 10-90% of the stack (fixed rule). Save Fluo (input),
ER_mask_v2 (pseudo-label) and ER_positive_region (loss/eval region) per plane.
ER.tif is used ONLY to make labels/regions; it is never a model input."""
import sys, json, hashlib, numpy as np, tifffile
from common import n_planes, load_plane, tissue_mask, paths, BATCH
from er_mask_v2 import er_mask, P
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
def label_support(E,F):
    """Label-generation support = Fluo tissue OR ER-signal tissue (log, Otsu). ER may be used
    to define labels; it is never a model input."""
    l=gaussian_filter(np.log1p(E),8)
    return tissue_mask(F) | (l>threshold_otsu(l))
s=sys.argv[1]; n=n_planes(s); zs=list(range(int(round(.1*(n-1))),int(round(.9*(n-1)))+1,5))
F=[];M=[];R=[];st=[]
for z in zs:
    E,Fl=load_plane(s,z,"er"),load_plane(s,z,"fluo"); T=label_support(E,Fl); m,r=er_mask(E,T)
    F.append(Fl.astype(np.uint16)); M.append(m.astype(np.uint8)); R.append(r.astype(np.uint8))
    st.append(dict(z=z,region_frac_tissue=float(r.sum()/T.sum()),mask_frac_region=float(m.sum()/max(r.sum(),1))))
np.savez_compressed(f"dataset_v2/{s}.npz",fluo=np.stack(F),mask=np.stack(M),region=np.stack(R),z=np.array(zs))
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump(dict(specimen=s,batch=BATCH[s],n_planes=n,planes=st,params={k:list(v) if isinstance(v,tuple) else v for k,v in P.items()},
  er_sha256=sha(paths(s)[0]),fluo_sha256=sha(paths(s)[1])),open(f"dataset_v2/{s}.json","w"),indent=1)
print(s,len(zs),'planes; mean region frac %.3f'%np.mean([x['region_frac_tissue'] for x in st]))
