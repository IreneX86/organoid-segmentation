"""E: quantify ER-reporter mosaicism. 'Expressing region' = tissue pixels whose
smoothed ER (sigma 6) exceeds Otsu threshold of the in-tissue ER distribution.
Measured from ER.tif only; it is an algorithmic region, not ground truth."""
import json, numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from common import SPECIMENS, BATCH, load_plane, sample_planes, tissue_mask
out={}
for s in SPECIMENS:
    rows=[]
    for z in sample_planes(s):
        E,F,P=load_plane(s,z,"er"),load_plane(s,z,"fluo"),load_plane(s,z,"pl")>0
        m=tissue_mask(F); es=gaussian_filter(E,6); t=threshold_otsu(es[m]); X=(es>t)&m
        bg=np.percentile(es[m],10); 
        rows.append(dict(z=z,expr_frac=float(X.sum()/m.sum()),
            pl_inside_expr=float((P&X).sum()/max((P&m).sum(),1)),
            er_contrast=float((np.median(es[X])-bg)/max(np.median(es[m&~X])-bg,1e-6)),
            fluo_ratio=float(np.median(gaussian_filter(F,6)[X])/np.median(gaussian_filter(F,6)[m&~X]))))
    agg={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!="z"}
    out[s]=dict(batch=BATCH[s],planes=rows,mean=agg); print(s,BATCH[s],{k:round(v,3) for k,v in agg.items()})
json.dump(out,open("out/e_mosaic.json","w"),indent=1)
