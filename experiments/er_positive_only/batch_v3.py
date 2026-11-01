"""Batch ER mask v3 for every Z plane of one specimen. ER.tif -> labels; Fluo only for
broad tissue support. Writes ZYX uint8 0/255 stacks with the same Z indexing as the
source plus per-plane statistics. Algorithmic pseudo-labels, NOT manual GT."""
import sys, json, csv, hashlib, time, numpy as np, tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from common import load_plane, n_planes, tissue_mask, paths, SPECIMENS_ALL, BATCH
from er_mask_v3 import er_mask, P3
from er_mask_v2 import P as P2
s=sys.argv[1]; n=n_planes(s); M=np.zeros((n,1200,1200),np.uint8); R=np.zeros_like(M); rows=[]; t0=time.time()
for z in range(n):
    E=load_plane(s,z,"er"); F=load_plane(s,z,"fluo")
    l=gaussian_filter(np.log1p(E),8); T=tissue_mask(F)|(l>threshold_otsu(l))
    m,r,info=er_mask(E,T); M[z]=m*255; R[z]=r*255
    rows.append(dict(z=z,region_px=int(r.sum()),**info))
    if z%10==0: print(s,z,n,round(time.time()-t0),flush=True)
batch,cell=SPECIMENS_ALL[s]; out=f"full_v3/{cell}"; import os; os.makedirs(out,exist_ok=True)
tifffile.imwrite(f"{out}/ER_mask_v3.tif",M,compression="zlib",metadata={"axes":"ZYX"})
tifffile.imwrite(f"{out}/ER_positive_region.tif",R,compression="zlib",metadata={"axes":"ZYX"})
with open(f"{out}/per_plane.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump(dict(specimen=s,cell=cell,batch_folder=batch.strip(),n_planes=n,params_v2=P2,params_v3=P3,
  source_er_sha256=sha(paths(s)[0]),source_fluo_sha256=sha(paths(s)[1]),
  mask_sha256=sha(f"{out}/ER_mask_v3.tif"),region_sha256=sha(f"{out}/ER_positive_region.tif"),
  seconds=round(time.time()-t0)),open(f"{out}/manifest.json","w"),indent=1,default=list)
print("DONE",s,round(time.time()-t0),flush=True)
