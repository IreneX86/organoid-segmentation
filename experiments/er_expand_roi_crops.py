"""Reproducible positive-rich ROI candidates, not complete manual labels."""
from pathlib import Path
import json, hashlib
import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/er_expanded_roi_candidates_20261029'
OLD=ROOT/'outputs/er_user_roi_dataset_20261029/manifest.json'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def rgb(a):
 lo,hi=np.percentile(a,[1,99.8])
 g=(255*np.clip((a.astype('float32')-lo)/max(hi-lo,1),0,1)).astype('uint8')
 return np.repeat(g[...,None],3,2)

def main():
 OUT.mkdir(exist_ok=False)
 old=json.loads(OLD.read_text()); splits={r['cell']:r['split'] for r in old['records']}
 records=[]; hashes_seen=set(); pages=[]
 for mp in sorted((ROOT/'outputs/er_coherent_fiber_full_20261028').rglob('manifest.json')):
  cell=mp.parent.name
  if cell not in splits:continue
  m=json.loads(mp.read_text()); paths=[Path(m['source_er']),Path(m['source_fluo']),mp.parent/'ER_mask_line.tif']
  expected=[m['source_er_sha256'],m['source_fluo_sha256'],m['outputs']['ER_mask_line']['sha256']]
  assert all(sha(p)==s for p,s in zip(paths,expected)),cell
  rows=[]
  with tf.TiffFile(paths[0]) as ef,tf.TiffFile(paths[1]) as ff,tf.TiffFile(paths[2]) as mf:
   depth,h,w=m['source_shape']; assert all(f.series[0].shape==(depth,h,w) for f in [ef,ff,mf])
   # Four separated Z positions, two non-overlapping XY windows per plane.
   # They are correlated biological samples, never counted as independent cells.
   for z in sorted(set(int((depth-1)*q) for q in [.15,.35,.65,.85])):
    er=ef.pages[z].asarray(); fl=ff.pages[z].asarray(); mask=mf.pages[z].asarray()>0
    choices=[]
    for y in sorted(set(range(0,h-320+1,80))|{h-320}):
     for x in sorted(set(range(0,w-320+1,80))|{w-320}):
      n=int(mask[y:y+320,x:x+320].sum())
      if n>=2048:choices.append((n,y,x))
    chosen=[]
    for n,y,x in sorted(choices,key=lambda v:(-v[0],v[1],v[2])):
     if any(abs(y-a)<320 and abs(x-b)<320 for a,b in chosen):continue
     chosen.append((y,x))
     sl=np.s_[y:y+320,x:x+320]; e=er[sl]; f=fl[sl]; pos=mask[sl]
     # Local support around existing lines; do not fill cell interiors as ER.
     roi=ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_dilation(pos,iterations=8),iterations=4))
     roi[:5]=False;roi[-5:]=False;roi[:,:5]=False;roi[:,-5:]=False
     p=pos&roi; sm=ndi.gaussian_filter(e.astype('float32'),.8)
     neg=roi&~pos&(ndi.distance_transform_edt(~pos)>3)&(sm<np.percentile(sm[roi],35))
     lab=np.full(e.shape,255,np.uint8);lab[p]=1;lab[neg]=0
     name=f'{cell}_z{z:03d}_y{y}_x{x}'; d=OUT/name;d.mkdir()
     arrays={'Fluo_input':f,'ER_reference':e,'positive_mask':p.astype('uint8')*255,'roi':roi.astype('uint8')*255,'partial_label_0neg_1pos_255ignore':lab,'valid_loss_mask':(lab!=255).astype('uint8')*255}
     hashes={}
     for k,a in arrays.items():
      tf.imwrite(d/(k+'.tif'),a);hashes[k]=sha(d/(k+'.tif'));assert np.array_equal(tf.imread(d/(k+'.tif')),a)
     assert hashes['Fluo_input'] not in hashes_seen;hashes_seen.add(hashes['Fluo_input'])
     over=rgb(f);over[p]=[255,220,0]
     sem=np.full((320,320,3),65,np.uint8);sem[lab==1]=[255,220,0];sem[lab==0]=[0,130,255]
     view=Image.new('RGB',(1280,350));draw=ImageDraw.Draw(view)
     for j,(a,title) in enumerate(zip([rgb(e),rgb(f),over,sem],['ER reference','Fluo input','Old ER mask within ROI','Yellow + / blue provisional - / gray ignore'])):
      view.paste(Image.fromarray(a),(j*320,30));draw.text((j*320+3,5),title,fill='white')
     view.save(d/'review.png');rows.append((name,view))
     records.append(dict(cell=cell,split=splits[cell],directory=name,z_zero_based=z,y0=y,x0=x,shape=[320,320],positive_pixels=int(p.sum()),positive_fraction=float(p.mean()),provisional_negative_pixels=int(neg.sum()),ignored_pixels=int((lab==255).sum()),hashes=hashes,source_manifest=str(mp),source_hashes=expected))
     if len(chosen)==2:break
  sheet=Image.new('RGB',(1280,len(rows)*375),'black');draw=ImageDraw.Draw(sheet)
  for i,(name,v) in enumerate(rows):draw.text((4,i*375),name,fill='white');sheet.paste(v,(0,i*375+25))
  sheet.save(OUT/(cell+'_review.png'));pages.append((cell,rows))
  print(cell,len(rows),flush=True)
 # Compact contact sheet: paired Fluo and overlay for every selected crop.
 overview=Image.new('RGB',(1280,5*400),'black');draw=ImageDraw.Draw(overview)
 for i,(cell,rows) in enumerate(pages):
  draw.text((4,i*400+2),cell+' | '+splits[cell],fill='white')
  for j,(name,v) in enumerate(rows):
   thumb=v.crop((320,30,960,350));thumb.thumbnail((320,160)); xx=(j%4)*320;yy=i*400+25+(j//4)*185
   overview.paste(thumb,(xx,yy));draw.text((xx,yy+162),name.split('_NZ_')[1],fill='white')
 overview.save(OUT/'all_40_paired_overview.png')
 (OUT/'manifest.json').write_text(json.dumps(dict(records=records,source_five_roi_manifest_sha256=sha(OLD),selection='Four Z fractions .15/.35/.65/.85; top two disjoint 320px windows by existing positive count, >=2% foreground',label_semantics='0 provisional negative,1 old-mask positive,255 ignored; automatic candidates not manual truth',training_started=False),indent=2))
 assert {r['cell'] for r in records if r['split']=='train'}.isdisjoint({r['cell'] for r in records if r['split']=='validation'})
 (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',pairs=len(records),train=sum(r['split']=='train' for r in records),validation=sum(r['split']=='validation' for r in records),source_hashes_checked=True,output_roundtrip_checked=True,exact_duplicate_inputs=0,within_plane_windows_nonoverlapping=True,original_five_preserved=True,training_started=False),indent=2))
if __name__=='__main__':main()
