"""Verify and visualize all binary-ROI validation results, no threshold selection."""
from pathlib import Path
import argparse,json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();r=a.root;s=r/'extracted/results/comparison'
 assert (r/'extracted/exit_status.txt').read_text().strip()=='0';complete=json.loads((s/'complete.json').read_text());assert complete['status']=='COMPLETE'
 ck=s/'candidate.keras';assert hashlib.sha256(ck.read_bytes()).hexdigest()==complete['checkpoint_sha256']
 with zipfile.ZipFile(ck) as z:assert z.testzip() is None
 means={};rows={}
 for arm,n in [('before',8),('after',8),('training',32)]:
  rows[arm]=json.loads((s/arm/'metrics.json').read_text());assert len(rows[arm])==n
  means[arm]={k:float(np.mean([v[k] for v in rows[arm]])) for k in ['dice','iou','precision','recall','positive_fraction']}
  for row in rows[arm]:
   with np.load(s/arm/row['file']) as f:
    assert f['fluo'].shape==f['label'].shape==f['probability'].shape==(320,320) and np.isfinite(f['probability']).all() and np.isin(f['label'],[0,1]).all()
 panel=Image.new('RGB',(1280,2800));d=ImageDraw.Draw(panel)
 for i,row in enumerate(rows['after']):
  with np.load(s/'after'/row['file']) as f:x=f['fluo'];y=f['label'];p=f['probability']
  with np.load(s/'before'/row['file']) as f:b=f['probability'];assert np.array_equal(y,f['label']) and np.array_equal(x,f['fluo'])
  imgs=[(255*np.clip(x,0,1)).astype('uint8'),y*255,((b>=.5)*255).astype('uint8'),((p>=.5)*255).astype('uint8')]
  for j,(im,title) in enumerate(zip(imgs,['Fluo '+row['file'].split('_z')[-1][:3],'New binary target','Source model on new target',f'After500 Dice={row["dice"]:.3f}'])):
   panel.paste(Image.fromarray(im).convert('RGB'),(j*320,i*350+30));d.text((j*320+3,i*350+5),title,fill='white')
 panel.save(r/'all8_binary_before_after.png')
 summary=dict(mean=means,rows=rows,complete=complete,verification='PASS',production_approved=False,scope='Userconfirmedbinarysemantics,pseudoERpositivecontours;onepreviouslyseenvalidation specimen,noindependenttest')
 (r/'review_metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
