import json, csv, numpy as np, tifffile, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from common import load_plane, SPECIMENS_ALL, BATCH
S=["23298","23300","23961","23963","23965","23966"]
fig,ax=plt.subplots(len(S),4,figsize=(18,4.6*len(S))); summ=[]
for i,s in enumerate(S):
    cell=SPECIMENS_ALL[s][1]; d=f"full_v3/{cell}"
    M=tifffile.imread(f"{d}/ER_mask_v3.tif"); R=tifffile.imread(f"{d}/ER_positive_region.tif"); n=len(M)
    rows=list(csv.DictReader(open(f"{d}/per_plane.csv")))
    summ.append(dict(specimen=s,batch=BATCH[s],planes=n,empty_planes=sum(int(r["v3_px"])==0 for r in rows) if "v3_px" in rows[0] else sum(int(r.get("v2_px",0))==0 for r in rows),
                     mean_mask_px=float(np.mean([int(r.get("v3_px",0)) for r in rows])),mean_region_frac=float((R>0).mean()),
                     puncta=sum(int(r.get("puncta",0)) for r in rows),dots=sum(int(r.get("dots",0)) for r in rows)))
    for j,f in enumerate((.2,.4,.6,.8)):
        z=int(round(f*(n-1))); E=load_plane(s,z,"er").astype(np.float32); e=np.log1p(E); a,b=np.percentile(e,[1,99.7]); e=np.clip((e-a)/(b-a),0,1)
        m=M[z]>0; r=R[z]>0; o=np.dstack([e]*3); o[r&binary_dilation(~r,disk(2))]=[0,.8,1]; o[m]=[1,.2,.6]
        ax[i,j].imshow(o); ax[i,j].set_xticks([]); ax[i,j].set_yticks([]); ax[i,j].set_title(f"{s} ({BATCH[s]}) z{z}/{n-1}",fontsize=12)
plt.tight_layout(); plt.savefig("full_v3/qc_overview.png",dpi=50)
json.dump(summ,open("full_v3/summary.json","w"),indent=1); [print(x) for x in summ]
