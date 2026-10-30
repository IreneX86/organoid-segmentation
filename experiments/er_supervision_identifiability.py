"""Train-only counterexample: identical masked supervision, different geometry.
These are constructed masks, NOT model predictions or ground-truth labels.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
DATA=R/'outputs/er_interline_student_ready_20261029'
OUT=R/'outputs/er_supervision_diagnostic_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loss(y,p):
 v=y!=255;t=y==1
 return float(np.mean((-(t*np.log(p))-(~t)*np.log1p(-p))[v])+1-(2*np.sum(p[v]*t[v])+1e-6)/(np.sum(t[v])+np.sum(p[v])+1e-6))
def main():
 OUT.mkdir(exist_ok=False)
 m=json.loads((DATA/'manifest.json').read_text())
 rows=sorted([r for r in m['records'] if r['split']=='train'],key=lambda r:r['file'])
 result=[];panels=[]
 for i,r in enumerate(rows):
  path=DATA/r['file'];assert sha(path)==r['sha256']
  with np.load(path) as a:
   y=a['label'];x=np.clip((a['fluo'].astype(float)-a['low'])/(a['high']-a['low']),0,1)
  pos=y==1;unknown=y==255
  # Fixed demonstration, not a proposed labelling rule: fill holes and widen
  # lines by 8px ONLY in ignored pixels, retaining every known label.
  expanded=pos|((ndi.binary_fill_holes(pos)|ndi.binary_dilation(pos,iterations=8))&unknown)
  thin=np.where(pos,.99,.01).astype('float32');wide=np.where(expanded,.99,.01).astype('float32')
  assert np.array_equal(thin[~unknown],wide[~unknown])
  l1=loss(y,thin);l2=loss(y,wide);assert l1==l2
  result.append(dict(file=r['file'],positive=int(pos.sum()),negative=int((y==0).sum()),ignored=int(unknown.sum()),added_unpenalized_pixels=int((expanded&~pos).sum()),line_loss=l1,expanded_loss=l2))
  if i in [0,8,16,24]:
   gt=np.full((*y.shape,3),65,dtype=np.uint8);gt[pos]=[255,220,0];gt[y==0]=[0,130,255]
   images=[np.repeat((x[...,None]*255).astype('uint8'),3,axis=2),gt,np.repeat((pos[...,None]*255).astype('uint8'),3,axis=2),np.repeat((expanded[...,None]*255).astype('uint8'),3,axis=2)]
   panel=Image.new('RGB',(1280,370));d=ImageDraw.Draw(panel);d.text((4,2),r['file'],fill='white')
   for j,(im,title) in enumerate(zip(images,['Fluo','Yellow positive / blue negative / gray ignore','Constructed line mask','Constructed wider mask: SAME LOSS'])):
    panel.paste(Image.fromarray(im),(j*320,50));d.text((j*320+4,28),title,fill='white')
   panels.append(panel)
 canvas=Image.new('RGB',(1280,len(panels)*370))
 for i,p in enumerate(panels):canvas.paste(p,(0,i*370))
 canvas.save(OUT/'same_loss_different_structure.png')
 summary=dict(scope='32 training crops only; no validation reads; no model training',rows=result,total_ignored=sum(r['ignored'] for r in result),total_pixels=32*320*320,total_added_unpenalized_pixels=sum(r['added_unpenalized_pixels'] for r in result),all_losses_exactly_equal=True,interpretation='Counterexample demonstrates supervision does not uniquely constrain line geometry; does NOT prove sole cause or that Fluo lacks ER information.')
 (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
