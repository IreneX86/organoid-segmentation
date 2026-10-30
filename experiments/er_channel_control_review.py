"""Verify paired channel-control archive and produce a static scientific comparison."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'outputs/er_channel_control_20261030';S=O/'extracted/results'
def main():
 assert (O/'extracted/exit_status.txt').read_text().strip()=='0'
 complete=json.loads((S/'complete.json').read_text());assert complete['status']=='COMPLETE'
 rows={};means={}
 for arm in ['fluo','er']:
  ck=S/arm/'candidate.keras';assert hashlib.sha256(ck.read_bytes()).hexdigest()==(S/arm/'checkpoint.sha256').read_text().strip()
  with zipfile.ZipFile(ck) as z:assert z.testzip() is None
  rows[arm]=json.loads((S/arm/'metrics.json').read_text());assert len(rows[arm])==40
  means[arm]={split:{k:float(np.mean([r[k] for r in rows[arm] if r['split']==split])) for k in ['dice','iou','precision','recall','positive_fraction']} for split in ['train','validation']}
  for row in rows[arm]:
   with np.load(S/arm/row['file']) as a:assert a['input'].shape==a['label'].shape==a['probability'].shape==(320,320) and np.isfinite(a['probability']).all() and np.isin(a['label'],[0,1]).all()
 va=[r for r in rows['fluo'] if r['split']=='validation'];panel=Image.new('RGB',(1600,2800));d=ImageDraw.Draw(panel)
 for i,r in enumerate(va):
  with np.load(S/'fluo'/r['file']) as a:f=a['input'];y=a['label'];p=a['probability']
  with np.load(S/'er'/r['file']) as a:e=a['input'];q=a['probability'];assert np.array_equal(y,a['label'])
  for j,(im,title) in enumerate(zip([f,e,y,p>=.5,q>=.5],['Fluo input','ER input (control)','Same binary target','Fluo prediction','ER prediction'])):
   panel.paste(Image.fromarray(np.uint8(np.clip(im,0,1)*255)).convert('RGB'),(j*320,i*350+30));d.text((j*320+5,i*350+5),title,fill='white')
 panel.save(O/'all8_channel_control.png')
 summary=dict(means=means,rows=rows,verification='PASS',production_approved=False,limitations='Single reused validation specimen; labels derived from ER, control partly circular. No independent manual accuracy.')
 (O/'review_metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
