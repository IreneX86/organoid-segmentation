"""Train-only conservative enclosed-gap weak-negative proposals; never ground truth."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import ndimage as ndi
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];SRC=R/'outputs/er_mesh_windows_20261029';REF=R/'outputs/er_expanded_roi_candidates_20261029';OUT=R/'outputs/er_gap_negative_candidates_20261030'
RULE=dict(gap_area=[4,256],boundary_distance_gt=1.5,rim_quantile=25,contrast_noise_multiplier=2,minimum_component=2,ER_smoothing_sigma=1,local_trough_sigma=3,raw_bright_guard=True,validation_used=False,labels_promoted=False,description='Only ignored enclosed-gap cores supported by a dark ER trough relative to the local labelled rim; not all holes become negative.')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def propose(er,label,window):
 fine=ndi.gaussian_filter(er.astype(float),1);broad=ndi.gaussian_filter(er.astype(float),3);res=er-fine;noise=max(float(np.median(abs(res-np.median(res)))/.67448975),1)
 i,j=window;y=label[i:i+64,j:j+64];e=fine[i:i+64,j:j+64];raw=er[i:i+64,j:j+64];b=broad[i:i+64,j:j+64];pos=y==1
 holes=ndi.binary_fill_holes(pos)&~pos;lab,n=ndi.label(holes);sizes=np.bincount(lab.ravel());add=np.zeros(y.shape,bool);detail=[]
 for k in range(1,n+1):
  if not 4<=sizes[k]<=256:continue
  gap=lab==k;rim=ndi.binary_dilation(gap)&pos
  if not rim.any():continue
  rim_low=float(np.percentile(e[rim],25));core=ndi.distance_transform_edt(gap)>1.5
  # Raw-intensity guard preserves bright structures missed by the source mask.
  candidate=gap&core&(y==255)&(e<=b)&(e<rim_low-2*noise)&(raw<rim_low-noise)
  cc,_=ndi.label(candidate);counts=np.bincount(cc.ravel());candidate&=(cc>0)&(counts[cc]>=2)
  add|=candidate;detail.append(dict(area=int(gap.sum()),added=int(candidate.sum()),rim_low=rim_low))
 full=np.zeros(label.shape,bool);full[i:i+64,j:j+64]=add
 return full,dict(noise_proxy=noise,gaps=detail)
def rgb(a):
 lo,hi=np.percentile(a,[1,99.8]);v=(255*np.clip((a-lo)/max(hi-lo,1),0,1)).astype('uint8');return np.repeat(v[...,None],3,2)
def tests():
 e=np.full((320,320),10.);y=np.full((320,320),255,np.uint8);y[22:42,22:42]=1;y[25:39,25:39]=255;e[y==1]=100
 a,_=propose(e,y,(0,0));assert a[29:35,29:35].any() and not a[y==1].any()
 e[25:39,31:33]=120;a,_=propose(e,y,(0,0));assert not a[25:39,31:33].any()
 a,_=propose(np.full(e.shape,10.),y,(0,0));assert not a.any()
 return dict(dark_core_selected=True,bright_unlabelled_line_protected=True,uniform_ER_rejected=True,positive_untouched=True)
def main():
 check=tests();OUT.mkdir(exist_ok=False);(OUT/'protocol.json').write_text(json.dumps(RULE,indent=2))
 m=json.loads((SRC/'manifest.json').read_text());refs={x['directory']:x for x in json.loads((REF/'manifest.json').read_text())['records']};rows=[];views=[];nc={}
 for r in m['records']:
  f=SRC/r['output'];assert sha(f)==r['sha256']
  with np.load(f) as q:arr={k:q[k].copy() for k in q.files}
  ref=refs[r['file'][:-4]];assert ref['split']=='train';ep=REF/r['file'][:-4]/'ER_reference.tif';assert sha(ep)==ref['hashes']['ER_reference'];er=tif.imread(ep).astype(float)
  old=arr['label'];add,diag=propose(er,old,(r['wy'],r['wx']));new=old.copy();new[add]=0
  assert np.array_equal(new==1,old==1) and np.all(new[old==0]==0) and np.all(old[add]==255) and not np.any(add&~arr['loss_window'])
  p=OUT/r['output'];np.savez_compressed(p,**{**arr,'label':new},proposed_negative=add)
  with np.load(p) as q:assert np.array_equal(q['label'],new) and np.array_equal(q['fluo'],arr['fluo'])
  i,j=r['wy'],r['wx'];crop=np.s_[i:i+64,j:j+64];sem=np.full((*old.shape,3),65,np.uint8);sem[old==1]=[255,220,0];sem[old==0]=[0,100,230];revised=sem.copy();revised[add]=[0,240,220];on_er=rgb(er);on_er[add]=[0,240,220]
  panel=Image.new('RGB',(1280,290));d=ImageDraw.Draw(panel);d.text((3,2),f'{r["output"]} {r["cell"]} z{r["z"]} | added {add.sum()} weak-negative pixels',fill='white')
  for k,(im,title) in enumerate(zip([rgb(er),rgb(arr['fluo']),sem,revised,on_er],['ER reference','Fluo input','Original partial labels','Cyan: candidate negatives','Candidates on ER'])):
   panel.paste(Image.fromarray(im[crop]).resize((256,256),Image.Resampling.NEAREST),(k*256,34));d.text((k*256+3,19),title,fill='white')
  panel.save(OUT/(p.stem+'_review.png'))
  if nc.get(r['cell'],0)<2:views.append(panel);nc[r['cell']]=nc.get(r['cell'],0)+1
  rows.append(dict(**{k:v for k,v in r.items() if k!='sha256'},original_mesh_sha256=r['sha256'],sha256=sha(p),new_negative=int(add.sum()),positive=int((new==1).sum()),negative_before=int((old==0).sum()),negative_after=int((new==0).sum()),ignored_before=int((old[arr['loss_window']]==255).sum()),ignored_after=int((new[arr['loss_window']]==255).sum()),diagnostic=diag))
  assert sha(f)==r['sha256'] and sha(ep)==ref['hashes']['ER_reference']
 totals={k:sum(v[k] for v in rows) for k in ['new_negative','positive','negative_before','negative_after','ignored_before','ignored_after']}
 result=dict(records=rows,totals=totals,synthetic_tests=check,source_checks='PASS',validation_opened=False,training_started=False,labels_promoted=False,production_approved=False)
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2));sheet=Image.new('RGB',(1280,len(views)*290))
 for i,p in enumerate(views):sheet.paste(p,(0,i*290))
 sheet.save(OUT/'eight_gap_candidates.png');print(json.dumps(totals,indent=2))
if __name__=='__main__':main()
