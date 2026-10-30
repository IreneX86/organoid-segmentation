"""Select dense known-label windows, preserving all source data and label semantics.
No new positive/negative labels. Context320, loss support64; train-only selection.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];DATA=R/'outputs/er_interline_student_ready_20261029';REF=R/'outputs/er_expanded_roi_candidates_20261029';OUT=R/'outputs/er_dense_windows_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(a):
 lo,hi=np.percentile(a,[1,99.8]);g=(np.clip((a.astype(float)-lo)/max(hi-lo,1),0,1)*255).astype('uint8');return np.repeat(g[...,None],3,2)
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((DATA/'manifest.json').read_text());ref=json.loads((REF/'manifest.json').read_text());lookup={r['directory']:r for r in ref['records']};candidates=[];cache={}
 protocol=dict(context_size=320,loss_window_size=64,stride=16,min_known_fraction=.75,min_positive_fraction=.1,max_positive_fraction=.7,min_negative_fraction=.1,max_per_specimen=6,max_per_z=2,order='known coverage descending then filename/y/x; non-overlapping in global coordinates on same specimen/z',labels='unchanged 0 weak negative /1 pseudo-positive /255 ignore; outside selected window excluded only for derived training loss',validation='not opened, changed or selected; original fixed validation remains required',purpose='supervision-density diagnostic, not curated manual ground truth')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for r in m['records']:
  if r['split']!='train':continue
  f=DATA/r['file'];assert sha(f)==r['sha256']
  with np.load(f) as a:arrays={k:a[k].copy() for k in a.files}
  cache[r['file']]=arrays;y=arrays['label']
  for i in range(0,257,16):
   for j in range(0,257,16):
    w=y[i:i+64,j:j+64];pos=float((w==1).mean());neg=float((w==0).mean());known=pos+neg
    if known>=.75 and .1<=pos<=.7 and neg>=.1:candidates.append(dict(record=r,wy=i,wx=j,known=known,positive=pos,negative=neg))
 selected=[];nc=collections.Counter();nz=collections.Counter()
 for c in sorted(candidates,key=lambda c:(-c['known'],c['record']['file'],c['wy'],c['wx'])):
  r=c['record'];key=(r['cell'],r['z']);gy=r['y']+c['wy'];gx=r['x']+c['wx']
  if nc[r['cell']]>=6 or nz[key]>=2:continue
  if any(s['cell']==key[0] and s['z']==key[1] and abs(s['global_y']-gy)<64 and abs(s['global_x']-gx)<64 for s in selected):continue
  s=dict(cell=r['cell'],z=r['z'],file=r['file'],wy=c['wy'],wx=c['wx'],global_y=gy,global_x=gx,known_fraction=c['known'],positive_fraction=c['positive'],negative_fraction=c['negative'],source_sha256=r['sha256']);selected.append(s);nc[r['cell']]+=1;nz[key]+=1
 views=[];vc=collections.Counter();known=pos=neg=0
 for n,s in enumerate(selected):
  a=cache[s['file']];y=a['label'];i,j=s['wy'],s['wx'];support=np.zeros(y.shape,bool);support[i:i+64,j:j+64]=True;derived=np.where(support,y,255).astype('uint8');assert np.array_equal(derived[support],y[support]);assert np.all(derived[~support]==255)
  out=OUT/f'window_{n:02d}.npz';np.savez_compressed(out,fluo=a['fluo'],label=derived,low=a['low'],high=a['high'],loss_window=support)
  with np.load(out) as b:assert np.array_equal(b['fluo'],a['fluo']) and np.array_equal(b['label'],derived)
  s['output']=out.name;s['sha256']=sha(out);known+=int((derived!=255).sum());pos+=int((derived==1).sum());neg+=int((derived==0).sum())
  if vc[s['cell']]<2:
   erp=REF/s['file'][:-4]/'ER_reference.tif';assert sha(erp)==lookup[s['file'][:-4]]['hashes']['ER_reference'];er=tif.imread(erp)
   sem=np.full((*y.shape,3),65,np.uint8);sem[y==1]=[255,220,0];sem[y==0]=[0,130,255]
   context=Image.fromarray(rgb(a['fluo']));ImageDraw.Draw(context).rectangle((j,i,j+63,i+63),outline='cyan',width=2)
   ims=[context,Image.fromarray(rgb(er)[i:i+64,j:j+64]).resize((320,320)),Image.fromarray(rgb(a['fluo'])[i:i+64,j:j+64]).resize((320,320)),Image.fromarray(sem[i:i+64,j:j+64]).resize((320,320),Image.Resampling.NEAREST)]
   panel=Image.new('RGB',(1280,365));d=ImageDraw.Draw(panel);d.text((3,3),f"{s['cell']} z{s['z']} known {s['known_fraction']:.1%}",fill='white')
   for k,(im,title) in enumerate(zip(ims,['Fluo context / cyan loss window','ER reference crop','Fluo crop','Yellow + / blue weak - / gray ignore'])):panel.paste(im,(k*320,45));d.text((k*320+3,24),title,fill='white')
   views.append(panel);vc[s['cell']]+=1
 for r in m['records']:
  if r['split']=='train':assert sha(DATA/r['file'])==r['sha256']
 canvas=Image.new('RGB',(1280,len(views)*365))
 for i,p in enumerate(views):canvas.paste(p,(0,i*365))
 canvas.save(OUT/'dense_windows_review.png')
 result=dict(status='PASS',records=selected,per_cell=dict(nc),candidate_count=len(candidates),selected_count=len(selected),known_pixels=known,positive_pixels=pos,negative_pixels=neg,within_windows_known_fraction=known/max(len(selected)*4096,1),new_positive_or_negative_labels=0,validation_opened=False,training_started=False,source_manifest_sha256=sha(DATA/'manifest.json'),warning='High coverage is not proof of biological correctness; source labels remain pseudo/weak.')
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
