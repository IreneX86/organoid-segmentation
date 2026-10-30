"""All-eight fixed-validation paired review, pseudo-label agreement only."""
from pathlib import Path
import argparse,json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageDraw

def metrics(y,p):
 m=p>=.5;t=y==1;n=y==0;tp=int((m&t).sum());fp=int((m&n).sum());fn=int((~m&t).sum())
 return dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),positive_fraction=float(m.mean()),ignored_positive_fraction=float(m[y==255].mean()),tp=tp,fp=fp,fn=fn)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();r=a.root;s=r/'extracted/results/comparison'
 assert (r/'extracted/exit_status.txt').read_text().strip()=='0'
 assert json.loads((s/'complete.json').read_text())['status']=='COMPLETE'
 rows=[];checks={}
 for arm in ['control','mesh']:
  p=s/arm/'candidate.keras';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==json.loads((s/arm/'result.json').read_text())['checkpoint_sha256']
  with zipfile.ZipFile(p) as z:assert z.testzip() is None
  checks[arm]=h
 names=sorted(p.name for p in (s/'control').glob('*.npz'));assert len(names)==8
 panel=Image.new('RGB',(1280,8*350));d=ImageDraw.Draw(panel)
 for i,name in enumerate(names):
  with np.load(s/'control'/name) as q:x=q['fluo'];y=q['label'];c=q['probability']
  with np.load(s/'mesh'/name) as q:assert np.array_equal(x,q['fluo']) and np.array_equal(y,q['label']);m=q['probability']
  assert x.shape==y.shape==c.shape==m.shape==(320,320) and np.isfinite(c).all() and np.isfinite(m).all()
  row=dict(file=name,control=metrics(y,c),mesh=metrics(y,m));rows.append(row)
  g=np.repeat((np.clip(x,0,1)[...,None]*255).astype('uint8'),3,axis=2)
  l=np.full(g.shape,65,np.uint8);l[y==1]=[255,220,0];l[y==0]=[0,130,255]
  imgs=[g,l,np.repeat(((c>=.5)*255).astype('uint8')[...,None],3,2),np.repeat(((m>=.5)*255).astype('uint8')[...,None],3,2)]
  for j,(im,title) in enumerate(zip(imgs,['Fluo '+name.split('_z')[-1][:3],'Pseudo label / weak negative / ignore',f'Mesh only D={row["control"]["dice"]:.3f}',f'Gap D={row["mesh"]["dice"]:.3f}'])):
   panel.paste(Image.fromarray(im),(j*320,i*350+30));d.text((j*320+4,i*350+5),title,fill='white')
 means={arm:{k:float(np.mean([v[arm][k] for v in rows])) for k in ['dice','iou','precision','recall','positive_fraction','ignored_positive_fraction']} for arm in ['control','mesh']}
 result=dict(mean=means,paired_dice_change=[v['mesh']['dice']-v['control']['dice'] for v in rows],improved_crops=sum(v['mesh']['dice']>v['control']['dice'] for v in rows),rows=rows,checkpoints=checks,verification='PASS',scope='Eight previously seen validation crops from ONE specimen; partial pseudo-label agreement, not biological accuracy or independent generalization',production_approved=False)
 (r/'review_metrics.json').write_text(json.dumps(result,indent=2));panel.save(r/'all8_mesh_vs_gap.png');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','checkpoints']},indent=2))
if __name__=='__main__':main()
