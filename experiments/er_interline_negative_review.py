"""Train-only inter-line weak-negative proposals; never automatic ground truth.
Geometry and ER trough evidence, not Fluo predictions, drive proposals.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import ndimage as ndi
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
SRC=R/'outputs/er_expanded_roi_candidates_20261029'
OUT=R/'outputs/er_interline_negative_review_20261029'
RULES=dict(distance_from_label_gt=3,max_distance=16,opposite_directions=[[0,1],[1,0],[1,1],[1,-1]],support_radius=1,
           contrast_noise_multiplier=2,local_positive_fraction_min=.04,
           trough='Gaussian ER sigma1 <= sigma3 and sigma6',
           fit_scope='32 train crops only; validation labels untouched',
           semantics='additional provisional negatives only; positives and old negatives unchanged')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def shifted(a,dy,dx):
 out=np.zeros_like(a);h,w=a.shape
 ya=max(0,-dy);yb=min(h,h-dy);xa=max(0,-dx);xb=min(w,w-dx)
 out[ya:yb,xa:xb]=a[ya+dy:yb+dy,xa+dx:xb+dx];return out

def propose(er,old,roi):
 pos=old==1;dist=ndi.distance_transform_edt(~pos);supported=ndi.binary_dilation(pos,iterations=1)
 bracket=np.zeros(pos.shape,bool)
 for dy,dx in RULES['opposite_directions']:
  left=np.zeros_like(pos);right=np.zeros_like(pos)
  for d in range(4,17):
   left|=shifted(supported,d*dy,d*dx);right|=shifted(supported,-d*dy,-d*dx)
  bracket|=left&right
 e=er.astype('float32');fine=ndi.gaussian_filter(e,1);mid=ndi.gaussian_filter(e,3);broad=ndi.gaussian_filter(e,6)
 residual=e-fine;noise=float(np.median(np.abs(residual-np.median(residual)))/.67448975)
 # Strong labelled lines must be visible locally; low ER alone does not qualify.
 peak=ndi.maximum_filter(np.where(pos,fine,-np.inf),size=33)
 density=ndi.uniform_filter(pos.astype('float32'),size=25)
 candidate=(old==255)&roi&bracket&(dist>3)&(dist<=16)&(fine<=mid)&(fine<=broad)&((peak-fine)>max(2*noise,1))&(density>=.04)
 candidate=ndi.binary_opening(candidate,iterations=1)
 components,n=ndi.label(candidate);sizes=np.bincount(components.ravel());candidate&=sizes[components]>=4
 return candidate,dict(noise_proxy=noise,bracket_pixels=int(bracket.sum()))
def rgb(a):
 lo,hi=np.percentile(a,[1,99.8]);g=(np.clip((a.astype('float32')-lo)/max(hi-lo,1),0,1)*255).astype('uint8');return np.repeat(g[...,None],3,2)
def synthetic_tests():
 er=np.ones((96,96),np.float32)*10;old=np.full((96,96),255,np.uint8);roi=np.ones_like(old,bool)
 er[:,30:33]=100;er[:,54:57]=100;old[:,30:33]=1;old[:,54:57]=1
 plain,_=propose(er,old,roi);assert plain[20:70,38:49].any()
 er[:,42:45]=120;protected,_=propose(er,old,roi);assert not protected[:,42:45].any()
 assert not protected[old==1].any()
 return dict(trough_selected=True,unlabelled_bright_line_protected=True,positives_unchanged=True)
def main():
 tests=synthetic_tests();OUT.mkdir(exist_ok=False)
 (OUT/'protocol.json').write_text(json.dumps(RULES,indent=2))
 m=json.loads((SRC/'manifest.json').read_text());rows=[];views=[];percell={}
 # Never open validation images or labels in this preparation experiment.
 for r in m['records']:
  if r['split']!='train':continue
  d=SRC/r['directory'];out=OUT/r['directory'];out.mkdir()
  sources={name:d/(name+'.tif') for name in ['ER_reference','Fluo_input','roi','partial_label_0neg_1pos_255ignore']}
  assert all(sha(p)==r['hashes'][k] for k,p in sources.items())
  er=tif.imread(sources['ER_reference']);fluo=tif.imread(sources['Fluo_input']);roi=tif.imread(sources['roi'])>0;old=tif.imread(sources['partial_label_0neg_1pos_255ignore'])
  add,diag=propose(er,old,roi);new=old.copy();new[add]=0
  assert np.array_equal(new==1,old==1) and np.all(new[old==0]==0) and not np.any(add&~roi)
  for name,a in [('partial_label_candidate',new),('new_interline_negative',add.astype('uint8')*255)]:tif.imwrite(out/(name+'.tif'),a);assert np.array_equal(tif.imread(out/(name+'.tif')),a)
  semantic=np.full((320,320,3),65,np.uint8);semantic[old==1]=[255,220,0];semantic[old==0]=[0,100,230]
  revised=semantic.copy();revised[add]=[0,240,220]
  on_er=rgb(er);on_er[add]=[0,240,220]
  view=Image.new('RGB',(1600,355));draw=ImageDraw.Draw(view)
  for j,(a,title) in enumerate(zip([rgb(er),rgb(fluo),semantic,revised,on_er],['ER reference','Fluo input','Old supervision','Cyan: proposed inter-line negatives','New proposals on ER'])):
   view.paste(Image.fromarray(a),(320*j,35));draw.text((320*j+3,5),title,fill='white')
  view.save(out/'review.png')
  index=percell.get(r['cell'],0);percell[r['cell']]=index+1
  # Fixed 1st and 5th entries per specimen, not selected by outcome.
  if index in [0,4]:views.append((r['directory'],view))
  rows.append(dict(directory=r['directory'],cell=r['cell'],split='train',positive=int((old==1).sum()),old_negative=int((old==0).sum()),new_negative=int(add.sum()),old_ignored=int((old==255).sum()),remaining_ignored=int((new==255).sum()),candidate_sha256=sha(out/'partial_label_candidate.tif'),source_label_sha256=sha(sources['partial_label_0neg_1pos_255ignore']),diagnostic=diag))
  assert all(sha(p)==r['hashes'][k] for k,p in sources.items())
 sheet=Image.new('RGB',(1600,len(views)*380))
 for i,(name,v) in enumerate(views):ImageDraw.Draw(sheet).text((4,i*380+2),name,fill='white');sheet.paste(v,(0,i*380+25))
 sheet.save(OUT/'eight_training_proposals.png')
 totals={key:sum(r[key] for r in rows) for key in ['positive','old_negative','new_negative','old_ignored','remaining_ignored']}
 result=dict(status='PASS',synthetic_tests=tests,records=rows,totals=totals,validation_opened=False,training_started=False,labels_promoted=False,source_manifest_sha256=sha(SRC/'manifest.json'),caveat='Weak proposals may still suppress faint ER. No biological ground-truth claim. Fixed old validation semantics must be preserved in any future comparison.')
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(totals,indent=2))
if __name__=='__main__':main()
