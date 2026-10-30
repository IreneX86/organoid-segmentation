"""Read-only diagnostic review: one memorized training crop, no accuracy claim."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];OUT=R/'outputs/er_single_roi_run_20261029';SRC=OUT/'extracted/results/fit'
def metrics(y,p):
 m=p>=.5;valid=y!=255;t=y==1
 tp=int((m&t).sum());fp=int((m&(y==0)).sum());fn=int((~m&t).sum())
 return dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn,all_positive_fraction=float(m.mean()),ignored_positive_fraction=float(m[~valid].mean()))
def main():
 assert (OUT/'extracted/exit_status.txt').read_text().strip()=='0'
 ck=SRC/'diagnostic_only.keras';assert ck.stat().st_size>0
 with zipfile.ZipFile(ck) as z:assert z.testzip() is None
 with np.load(SRC/'diagnostic.npz') as a:
  x=a['fluo'][0,...,0];y=a['label'][0,...,0];before=a['before'][0,...,0];after=a['after'][0,...,0]
 assert np.isfinite(before).all() and np.isfinite(after).all()
 stats=dict(before=metrics(y,before),after=metrics(y,after),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),checkpoint_zip_valid=True,scope='One fixed training crop; diagnostic memorization, not validation',production_approved=False)
 (OUT/'review_metrics.json').write_text(json.dumps(stats,indent=2))
 gray=np.repeat((x[...,None]*255).astype('uint8'),3,2);label=np.full(gray.shape,65,np.uint8);label[y==1]=[255,220,0];label[y==0]=[0,130,255]
 overlays=[]
 for p in [before,after]:
  o=gray.copy();o[p>=.5]=(o[p>=.5]*.35+np.array([255,220,0])*.65).astype('uint8');overlays.append(o)
 imgs=[gray,label,np.repeat((before[...,None]>=.5).astype('uint8')*255,3,2),np.repeat((after[...,None]>=.5).astype('uint8')*255,3,2),*overlays]
 names=['Fluo input','Positive / weak negative / ignore','Before mask 0.5','After 500 steps mask 0.5','Before overlay','After overlay']
 panel=Image.new('RGB',(960,700));d=ImageDraw.Draw(panel)
 for i,(im,name) in enumerate(zip(imgs,names)):
  px=(i%3)*320;py=(i//3)*350;panel.paste(Image.fromarray(im),(px,py+30));d.text((px+3,py+5),name,fill='white')
 panel.save(OUT/'single_roi_comparison.png');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
