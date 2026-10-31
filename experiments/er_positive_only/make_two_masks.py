import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "er_information_audit"))
import json, hashlib, numpy as np, tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize, disk
from common import load_plane, tissue_mask, robust_norm, paths
from er_mask_v2 import er_mask, P
from d_pseudolabel_audit import ridge_ref
OUT="/home/claude/er_audit/masks_v2"; import os; os.makedirs(OUT,exist_ok=True)
sel=[("23300",74),("23963",20)]; man={"params":{k:list(v) if isinstance(v,tuple) else v for k,v in P.items()},"items":[]}
fig,ax=plt.subplots(2,4,figsize=(20,10.4))
for i,(s,z) in enumerate(sel):
    E,F,OLD=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z,"pl")>0
    T=tissue_mask(F); M,R=er_mask(E,T)
    name=f"{s}_z{z:03d}"
    tifffile.imwrite(f"{OUT}/{name}_ER_mask_v2.tif",(M*255).astype(np.uint8)); tifffile.imwrite(f"{OUT}/{name}_ER_positive_region.tif",(R*255).astype(np.uint8))
    ref,_=ridge_ref(E,T); sk=skeletonize(M); w=2*distance_transform_edt(M)[sk]-1
    st=dict(specimen=s,z=z,mask_px=int(M.sum()),region_frac_tissue=float(R.sum()/T.sum()),mask_frac_region=float(M.sum()/max(R.sum(),1)),
            width_median=float(np.median(w)),old_px=int((OLD&T).sum()),old_covered_by_new=float((OLD&binary_dilation(M,disk(2))).sum()/max(OLD.sum(),1)),
            new_px_outside_old=float((M&~binary_dilation(OLD,disk(2))).sum()/max(M.sum(),1)),
            ridge_in_region_covered_new=float((ref&R&binary_dilation(M,disk(2))).sum()/max((ref&R).sum(),1)),
            ridge_in_region_covered_old=float((ref&R&binary_dilation(OLD,disk(2))).sum()/max((ref&R).sum(),1)),
            source_er_sha256=hashlib.sha256(open(paths(s)[0],'rb').read()).hexdigest())
    man["items"].append(st); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in st.items() if k!='source_er_sha256'})
    e=robust_norm(np.log1p(E),T); show=lambda a,im,t:(a.imshow(im,cmap="gray" if im.ndim==2 else None),a.set_title(t,fontsize=11),a.set_xticks([]),a.set_yticks([]))
    def ov(mask,col):
        o=np.dstack([e,e,e]).clip(0,1); o[mask]=col; return o
    o3=np.dstack([e,e,e]).clip(0,1); o3[R&~binary_dilation(R,disk(1))^R]=0; 
    edge=R&~(binary_dilation(~R,disk(2))==0); o3[edge]=[0,0.8,1]
    show(ax[i,0],e,f"{s} z{z}  ER.tif (log display)"); show(ax[i,1],ov(OLD&T,[1,.85,0]),"old pseudo-label (yellow)")
    o4=ov(M,[1,0.2,0.6]); o4[edge]=[0,0.8,1]; show(ax[i,2],o4,"NEW mask (magenta) + ER-positive region (cyan edge)")
    show(ax[i,3],(M*255).astype(np.uint8),"NEW mask 0/255")
plt.tight_layout(); plt.savefig(f"{OUT}/two_masks_overview.png",dpi=90)
json.dump(man,open(f"{OUT}/manifest.json","w"),indent=1)
