import json, numpy as np, tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter, center_of_mass, distance_transform_edt, uniform_filter
from skimage.filters import threshold_otsu
from skimage.morphology import disk, skeletonize
from common import load_plane, n_planes, tissue_mask, robust_norm, BATCH
from er_mask_v2 import er_mask as v2
from er_mask_v3 import er_mask as v3, P3
from d_pseudolabel_audit import ridge_ref
OUT="five_masks_v3"; import os; os.makedirs(OUT,exist_ok=True)
S=["23298","23300","23961","23965","23966"]; rows=[]
fig,ax=plt.subplots(5,5,figsize=(25,25))
def wstat(M):
    sk=skeletonize(M); w=2*distance_transform_edt(M)[sk]-1; return float(np.median(w)),float(np.percentile(w,90))
for i,s in enumerate(S):
    z=n_planes(s)//2; E=load_plane(s,z,"er"); F=load_plane(s,z,"fluo")
    l=gaussian_filter(np.log1p(E),8); T=tissue_mask(F)|(l>threshold_otsu(l))
    M2,R=v2(E,T); M3,R3,info=v3(E,T)
    tifffile.imwrite(f"{OUT}/{s}_z{z:03d}_ER_mask_v3.tif",(M3*255).astype(np.uint8)); tifffile.imwrite(f"{OUT}/{s}_z{z:03d}_ER_positive_region.tif",(R*255).astype(np.uint8))
    ref,_=ridge_ref(E,T); refR=ref&R; d2=disk(2)
    on=lambda M:float((M&binary_dilation(refR,d2)).sum()/max(M.sum(),1)); cov=lambda M:float((refR&binary_dilation(M,d2)).sum()/max(refR.sum(),1))
    r=dict(specimen=s,batch=BATCH[s],z=z,**info,width_v2=wstat(M2),width_v3=wstat(M3),on_ridge_v2=on(M2),on_ridge_v3=on(M3),ridge_cov_v2=cov(M2),ridge_cov_v3=cov(M3))
    rows.append(r); print({k:(tuple(round(x,2) for x in v) if isinstance(v,tuple) else (round(v,3) if isinstance(v,float) else v)) for k,v in r.items()})
    en=robust_norm(E,T); e=robust_norm(np.log1p(E),T)
    # zoom A: mask centroid (same as before); zoom B: densest area of removed pixels (shows what changed)
    cy,cx=map(int,center_of_mass(M3)); rem=(M2&~M3).astype(float); dens=uniform_filter(rem,200)
    by,bx=np.unravel_index(np.argmax(dens),dens.shape)
    clamp=lambda v:min(max(int(v),120),1080)
    def crop(cy,cx): cy,cx=clamp(cy),clamp(cx); return (slice(cy-120,cy+120),slice(cx-120,cx+120))
    sa,sb=crop(cy,cx),crop(by,bx)
    def ovl(M,sl):
        o=np.dstack([en,en,en]).clip(0,1)[sl].copy(); o[M[sl]]=[1,.2,.6]; return o
    def diff(sl):
        o=np.dstack([en,en,en]).clip(0,1)[sl].copy(); o[(M2&~M3)[sl]]=[1,.6,0]; o[M3[sl]]=[1,.2,.6]; return o
    ov=np.dstack([e,e,e]).clip(0,1); ov[M3]=[1,.2,.6]
    panels=[(ov,f"{s} ({BATCH[s]}) z{z} — v3 mask"),(en[sb],"zoom B (most changed) — ER"),(diff(sb),"zoom B: kept (magenta) / removed (orange)"),(en[sa],"zoom A (mask centre) — ER"),(diff(sa),"zoom A: kept / removed")]
    for j,(im,t) in enumerate(panels):
        ax[i,j].imshow(im,cmap="gray" if im.ndim==2 else None); ax[i,j].set_xticks([]); ax[i,j].set_yticks([]); ax[i,j].set_title(t,fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/five_masks_v2_vs_v3.png",dpi=55)
json.dump(dict(params_v3=P3,items=rows),open(f"{OUT}/manifest.json","w"),indent=1)
