"""Train-only ER mesh-window proposals; no new labels or model fitting.
Topology is a sampling heuristic, never biological ground truth.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];DATA=R/'outputs/er_interline_student_ready_20261029';REF=R/'outputs/er_expanded_roi_candidates_20261029';OUT=R/'outputs/er_mesh_windows_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb(a):
 lo,hi=np.percentile(a,[1,99.8]);g=(np.clip((a.astype(float)-lo)/max(hi-lo,1),0,1)*255).astype('uint8');return np.repeat(g[...,None],3,2)
def geometry(pos):
 sk=skeletonize(pos);degree=ndi.convolve(sk.astype('uint8'),np.ones((3,3),np.uint8))-sk
 branches=(degree>=3)&sk;branches[:3]=False;branches[-3:]=False;branches[:,:3]=False;branches[:,-3:]=False
 _,nb=ndi.label(ndi.binary_dilation(branches),structure=np.ones((3,3)))
 holes=ndi.binary_fill_holes(pos)&~pos;labs,n=ndi.label(holes);sizes=np.bincount(labs.ravel())[1:]
 return int(nb),int(((sizes>=4)&(sizes<=256)).sum())
def main():
 OUT.mkdir(exist_ok=False);manifest=json.loads((DATA/'manifest.json').read_text());refs=json.loads((REF/'manifest.json').read_text());lookup={r['directory']:r for r in refs['records']}
 rules=dict(window=64,stride=16,context=320,positive_fraction=[.1,.6],min_negative_fraction=.05,min_small_enclosed_gaps=3,gap_area=[4,256],min_branch_clusters=3,negative_near_line_distance=8,min_near_negative_fraction=.5,min_ER_positive_negative_contrast_noise=1,max_per_cell=4,max_per_z=2,selection='small gap count descending capped at10, contrast descending, known coverage descending, file/y/x ties',labels='preserved original partial labels; no new negatives; 255 outside loss window',validation_used=False,production_approved=False)
 (OUT/'protocol.json').write_text(json.dumps(rules,indent=2))
 cache={};candidates=[];reasons=collections.Counter()
 for r in manifest['records']:
  if r['split']!='train':continue
  f=DATA/r['file'];assert sha(f)==r['sha256']
  with np.load(f) as a:arr={k:a[k].copy() for k in a.files}
  erp=REF/r['file'][:-4]/'ER_reference.tif';assert sha(erp)==lookup[r['file'][:-4]]['hashes']['ER_reference']
  er=tif.imread(erp).astype('float32');fine=ndi.gaussian_filter(er,1);res=er-fine;noise=max(float(np.median(abs(res-np.median(res)))/.67448975),1)
  y=arr['label'];dist=ndi.distance_transform_edt(y!=1);cache[r['file']]=(arr,er)
  for i in range(0,257,16):
   for j in range(0,257,16):
    w=y[i:i+64,j:j+64];pos=w==1;neg=w==0
    if not(.1<=pos.mean()<=.6 and neg.mean()>=.05):reasons['class_balance']+=1;continue
    b,h=geometry(pos)
    if b<3 or h<3:reasons['not_mesh_topology']+=1;continue
    near=float(np.mean(dist[i:i+64,j:j+64][neg]<=8))
    if near<.5:reasons['mostly_distant_negative']+=1;continue
    e=fine[i:i+64,j:j+64];contrast=float((np.median(e[pos])-np.median(e[neg]))/noise)
    if contrast<1:reasons['weak_ER_contrast']+=1;continue
    candidates.append(dict(record=r,wy=i,wx=j,branch_clusters=b,small_gaps=h,near_negative_fraction=near,ER_contrast_noise=contrast,known_fraction=float((w!=255).mean())))
 selected=[];nc=collections.Counter();nz=collections.Counter()
 for c in sorted(candidates,key=lambda c:(-min(c['small_gaps'],10),-c['ER_contrast_noise'],-c['known_fraction'],c['record']['file'],c['wy'],c['wx'])):
  r=c['record'];key=(r['cell'],r['z']);gy=r['y']+c['wy'];gx=r['x']+c['wx']
  if nc[r['cell']]>=4 or nz[key]>=2:continue
  if any(s['cell']==key[0] and s['z']==key[1] and abs(s['global_y']-gy)<64 and abs(s['global_x']-gx)<64 for s in selected):continue
  s={k:v for k,v in c.items() if k!='record'};s.update(cell=r['cell'],z=r['z'],file=r['file'],global_y=gy,global_x=gx,source_sha256=r['sha256']);selected.append(s);nc[key[0]]+=1;nz[key]+=1
 panels=[];overview=[];vc=collections.Counter();totals=collections.Counter()
 for idx,s in enumerate(selected):
  arr,er=cache[s['file']];y=arr['label'];i,j=s['wy'],s['wx'];support=np.zeros(y.shape,bool);support[i:i+64,j:j+64]=True;label=np.where(support,y,255).astype('uint8')
  assert np.array_equal(label[support],y[support]) and np.all(label[~support]==255)
  file=OUT/f'mesh_{idx:02d}.npz';np.savez_compressed(file,fluo=arr['fluo'],label=label,low=arr['low'],high=arr['high'],loss_window=support)
  with np.load(file) as a:assert np.array_equal(a['fluo'],arr['fluo']) and np.array_equal(a['label'],label)
  s.update(output=file.name,sha256=sha(file));totals['positive']+=int((label==1).sum());totals['negative']+=int((label==0).sum());totals['window_pixels']+=4096
  sem=np.full((*y.shape,3),65,np.uint8);sem[y==1]=[255,220,0];sem[y==0]=[0,130,255]
  context=Image.fromarray(rgb(arr['fluo']));ImageDraw.Draw(context).rectangle((j,i,j+63,i+63),outline='cyan',width=2)
  ims=[context,Image.fromarray(rgb(er)[i:i+64,j:j+64]).resize((320,320)),Image.fromarray(rgb(arr['fluo'])[i:i+64,j:j+64]).resize((320,320)),Image.fromarray(sem[i:i+64,j:j+64]).resize((320,320),Image.Resampling.NEAREST)]
  panel=Image.new('RGB',(1280,365));d=ImageDraw.Draw(panel);d.text((3,3),f"mesh_{idx:02d} {s['cell']} z{s['z']} gaps{s['small_gaps']} known{s['known_fraction']:.1%}",fill='white')
  for k,(im,title) in enumerate(zip(ims,['Fluo context / cyan loss region','ER reference crop','Fluo crop','Yellow + / blue weak - / gray ignore'])):panel.paste(im,(k*320,45));d.text((k*320+3,24),title,fill='white')
  panel.save(OUT/f'mesh_{idx:02d}_review.png');panels.append(panel)
  if vc[s['cell']]<2:overview.append(panel);vc[s['cell']]+=1
 if overview:
  sheet=Image.new('RGB',(1280,len(overview)*365))
  for i,p in enumerate(overview):sheet.paste(p,(0,i*365))
  sheet.save(OUT/'mesh_windows_review.png')
 for r in manifest['records']:
  if r['split']=='train':assert sha(DATA/r['file'])==r['sha256']
 result=dict(status='PASS',records=selected,candidate_count=len(candidates),selected_count=len(selected),per_cell=dict(nc),exclusions=dict(reasons),totals=dict(totals),new_labels=0,validation_opened=False,training_started=False,training_approved=False,note='Exploratory sampling only; pseudo labels not manual ground truth; do not change validation labels')
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
