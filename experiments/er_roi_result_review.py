"""Verify downloaded ROI student; present all validation crops, without reselection."""
from pathlib import Path
import json,hashlib,zipfile,csv
import numpy as np
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
D=R/'outputs/er_roi_student_run_20261029/extracted/results/student'
def main():
 result=json.loads((D/'result.json').read_text());ck=D/'er_roi_student.keras'
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==result['checkpoint_sha256']
 with zipfile.ZipFile(ck) as z:assert z.testzip() is None and 'config.json' in z.namelist()
 metrics=json.loads((D/'metrics.json').read_text());summary={}
 for s in ['train','validation']:
  rows=[r for r in metrics['rows'] if r['split']==s];tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
  summary[s]=dict(n=len(rows),dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),mean_patch_dice=float(np.mean([r['dice'] for r in rows])),unknown_positive_fraction_mean=float(np.mean([r['unknown_positive_fraction'] for r in rows])))
 sheets=[]
 for file in sorted((D/'predictions/validation').glob('*.npz')):
  with np.load(file) as a:x=a['fluo_normalized'];y=a['label'];p=a['probability']
  assert x.shape==y.shape==p.shape==(320,320) and np.isfinite(p).all() and np.isin(y,[0,1,255]).all()
  mask=(p>=.5).astype('uint8')*255
  tif.imwrite(file.with_suffix('.mask.tif'),mask);tif.imwrite(file.with_suffix('.probability.tif'),p)
  gray=np.repeat((x*255).astype('uint8')[...,None],3,2);overlay=gray.copy();overlay[p>=.5]=[255,220,0]
  labels=np.full((320,320,3),65,np.uint8);labels[y==1]=[255,220,0];labels[y==0]=[0,130,255]
  view=Image.new('RGB',(1280,350))
  for i,(img,title) in enumerate(zip([gray,labels,np.repeat(mask[...,None],3,2),overlay],['Fluo','Pseudo labels (+ / weak - / ignore)','Prediction threshold 0.5','Prediction overlay'])):
   view.paste(Image.fromarray(img),(i*320,30));ImageDraw.Draw(view).text((i*320+3,5),title,fill='white')
  view.save(file.with_suffix('.review.png'));sheets.append(view)
 assert len(sheets)==8
 canvas=Image.new('RGB',(1280,350*8))
 for i,v in enumerate(sheets):canvas.paste(v,(0,350*i))
 canvas.save(D/'validation_all8.png')
 (D/'verified_summary.json').write_text(json.dumps(dict(status='PASS',checkpoint_verified=True,result=result,metrics=summary,note='Known-pixel pseudo-label agreement, not independent manual ER accuracy'),indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
