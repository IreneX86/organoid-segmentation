"""Conservative nERdy Fluo candidate cleanup; no retraining or GT claims."""
from pathlib import Path
import json,hashlib
import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, disk
from skimage.draw import line
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];SRC=R/'outputs/er_nerdy_fluo_five_20261029';OUT=R/'outputs/er_nerdy_connect_first_20261029'
CFG=dict(min_component_area=5,min_skeleton_length=3,max_gap_pixels=18,tangent_steps=6,min_facing_cosine=0.45,min_supported_gap_fraction=0.6,weak_probability_relative_to_endpoints=0.6)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(a):
 lo,hi=np.percentile(a,[1,99.8]);return np.clip((a-lo)/max(hi-lo,1),0,1)
def tangent(sk,pt):
 # Follow a short skeleton neighborhood; diagonal corners can have two graph
 # neighbors, so stopping at the first such corner would reject valid tangents.
 start=tuple(pt);visited={start};front={start}
 for k in range(CFG['tangent_steps']):
  new=set()
  for cy,cx in front:
   for y in range(cy-1,cy+2):
    for x in range(cx-1,cx+2):
     if 0<=y<sk.shape[0] and 0<=x<sk.shape[1] and sk[y,x] and (y,x) not in visited:new.add((y,x))
  if not new:break
  visited.update(new);front=new
 q=np.array(list(visited));dist=np.linalg.norm(q-np.array(pt),axis=1)
 if dist.max()<3:return None
 back=q[dist>=dist.max()-1].mean(axis=0);v=np.array(pt)-back
 return v/np.linalg.norm(v)

def refine(raw,old,prob):
 sm=ndi.gaussian_filter(raw.astype('float32'),0.8); contrast=sm-ndi.gaussian_filter(sm,6)
 residual=raw-sm;sigma=float(1.4826*np.median(np.abs(residual-np.median(residual))))
 lab,n=ndi.label(old,np.ones((3,3)));sk=skeletonize(old);areas=np.bincount(lab.ravel());lens=np.bincount(lab[sk],minlength=n+1)
 keep=np.zeros(n+1,bool)
 for idx,sl in enumerate(ndi.find_objects(lab),1):
  if sl is None or areas[idx]<CFG['min_component_area'] or lens[idx]<CFG['min_skeleton_length']:continue
  # Require real local contrast, not merely a long threshold texture fragment.
  vals=contrast[sl][lab[sl]==idx]
  keep[idx]=True  # Preserve short true-line candidates until after bridging.
 clean=keep[lab];sk=skeletonize(clean);nb=ndi.convolve(sk.astype('uint8'),np.ones((3,3),dtype='uint8'))-sk
 ends=np.argwhere(sk&(nb==1)); tree=cKDTree(ends);add=np.zeros_like(old);used=set();bridges=[]
 pcut=float(np.percentile(prob[old],15)) if old.any() else 1
 support=ndi.maximum_filter(contrast,size=3)>0  # Permit a one-pixel uncertainty in ridge position.
 pairs=sorted(tree.query_pairs(CFG['max_gap_pixels']),key=lambda ij:np.linalg.norm(ends[ij[0]]-ends[ij[1]]))
 for i,j in pairs:
  if i in used or j in used:continue
  a,b=ends[i],ends[j];delta=b-a;dist=np.linalg.norm(delta)
  if dist<2:continue
  va,vb=tangent(sk,a),tangent(sk,b)
  if va is None or vb is None or va@(delta/dist)<CFG['min_facing_cosine'] or vb@(-delta/dist)<CFG['min_facing_cosine']:continue
  yy,xx=line(*a,*b);gap=~clean[yy,xx]
  weak=CFG['weak_probability_relative_to_endpoints']*min(float(prob[tuple(a)]),float(prob[tuple(b)]))
  supported=support[yy[gap],xx[gap]]&(prob[yy[gap],xx[gap]]>=weak)
  if not gap.any() or np.mean(supported)<CFG['min_supported_gap_fraction']:continue
  # Never bridge through a third pre-existing component.
  touched=set(lab[yy,xx])-{0,int(lab[tuple(a)]),int(lab[tuple(b)])}
  if touched:continue
  add[yy,xx]=True;used.update((i,j));bridges.append(dict(a=a.tolist(),b=b.tolist(),distance=float(dist)))
 # Added center lines get at most a one-pixel side support; no blanket closing/fill.
 added=ndi.binary_dilation(add,structure=disk(1))&support
 final=clean|add|added
 # Only isolated tiny leftovers are removed, after connectivity has been considered.
 ll,nn=ndi.label(final,np.ones((3,3)));aa=np.bincount(ll.ravel());ss=skeletonize(final);length=np.bincount(ll[ss],minlength=nn+1)
 retain=(aa>=12)|(length>=8);retain[0]=False;final=retain[ll]
 return final,dict(old_components=n,kept_components=int(keep.sum()),bridges=bridges,removed_pixels=int((old&~final).sum()),added_pixels=int((final&~old).sum()),noise_proxy=sigma,probability_support_cutoff=pcut,final_components=int(ndi.label(final,np.ones((3,3)))[1]))

def main():
 OUT.mkdir(exist_ok=False);manifest=json.loads((SRC/'manifest.json').read_text());records=[];overview=Image.new('RGB',(1600,425*5),'black')
 for i,row in enumerate(manifest['records']):
  src=SRC/row['cell'];out=OUT/row['cell'];out.mkdir()
  assert sha(src/'Fluo_input.tif')==row['input_sha256'] and sha(src/'candidate_mask_NOT_GT.tif')==row['mask_sha256']
  raw=tf.imread(src/'Fluo_input.tif');old=tf.imread(src/'candidate_mask_NOT_GT.tif')>0;prob=tf.imread(src/'model_sigmoid_NOT_calibrated.tif')
  new,stats=refine(raw,old,prob);tf.imwrite(out/'connected_candidate_mask.tif',new.astype('uint8')*255)
  tf.imwrite(out/'added_pixels.tif',(new&~old).astype('uint8')*255);tf.imwrite(out/'removed_pixels.tif',(old&~new).astype('uint8')*255)
  rgb=np.repeat((norm(raw.astype('float32'))*255).astype('uint8')[...,None],3,2)
  def overlay(m):
   a=rgb.copy();a[m]=(a[m]*.35+np.array([255,220,0])*.65).astype('uint8');return a
  panels=[rgb,overlay(old),overlay(new),np.repeat((new.astype('uint8')*255)[...,None],3,2)]
  full=Image.new('RGB',(4800,1230),'black')
  for j,(panel,title) in enumerate(zip(panels,['Fluo','Original nERdy','Connect first, retain short lines','Candidate mask, NOT GT'])):
   full.paste(Image.fromarray(panel),(j*1200,30));ImageDraw.Draw(full).text((j*1200+5,5),title,fill='white');overview.paste(Image.fromarray(panel).resize((400,400)),(j*400,i*425+25))
  full.save(out/'before_after.png');ImageDraw.Draw(overview).text((5,i*425+5),f'{row["cell"]} Z{row["z_one_based"]}: Fluo / old / revised / mask',fill='white')
  diff=rgb.copy();diff[old&~new]=[255,65,65];diff[new&~old]=[0,240,255];Image.fromarray(diff).save(out/'changes_red_removed_cyan_added.png')
  record=dict(cell=row['cell'],z=row['z_one_based'],source_mask_sha256=row['mask_sha256'],output_sha256=sha(out/'connected_candidate_mask.tif'),**stats);records.append(record);print(json.dumps(record),flush=True)
 overview.save(OUT/'five_before_after.png');(OUT/'manifest.json').write_text(json.dumps(dict(config=CFG,records=records,warning='Exploratory morphology changes. Tubular appearance does not establish ER identity; no biological accuracy claim.',source=str(SRC)),indent=2))
if __name__=='__main__':main()
