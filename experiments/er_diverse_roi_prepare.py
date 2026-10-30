"""Expand fixed ER/Fluo pseudo-label crops without consulting model predictions.

Old 40 crops remain immutable evaluation anchors. New candidates are sampled by
fixed Z locations and foreground-density strata, not validation performance.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import tifffile as tif
from scipy import ndimage as ndi
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
OLD=R/'outputs/er_binary_roi_20261030/payload/data'
O=R/'outputs/er_diverse_roi_20261030'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gray(a):
 l,h=np.percentile(a,[1,99.8]);return np.uint8(np.clip((a.astype('float32')-l)/max(h-l,1),0,1)*255)
def pick(mask):
 h,w=mask.shape;candidates=[]
 for y in sorted(set(range(0,h-319,80))|{h-320}):
  for x in sorted(set(range(0,w-319,80))|{w-320}):
   n=int(mask[y:y+320,x:x+320].sum())
   if n>=2048:candidates.append((n,y,x))
 if not candidates:return []
 candidates.sort();chosen=[]
 # Distinct density strata provide more than the old densest-patch sampler.
 targets=np.quantile([c[0] for c in candidates],[.2,.5,.8])
 for target in targets:
  for n,y,x in sorted(candidates,key=lambda q:(abs(q[0]-target),q[1],q[2])):
   if any(abs(y-v[1])<320 and abs(x-v[2])<320 for v in chosen):continue
   chosen.append((n,y,x));break
 return chosen

def main():
 O.mkdir(exist_ok=False);(O/'data').mkdir()
 old=json.loads((OLD/'manifest.json').read_text());orig=json.loads((R/'outputs/er_expanded_roi_candidates_20261029/manifest.json').read_text());lookup={r['directory']+'.npz':r for r in orig['records']}
 cells=sorted({r['cell'] for r in old['records']});records=[];sources=[];skipped=[];seen=set();sheets=[]
 for cell in cells:
  anchors=[r for r in old['records'] if r['cell']==cell];source=Path(lookup[anchors[0]['file']]['source_manifest']);sm=json.loads(source.read_text());paths=[Path(sm['source_fluo']),Path(sm['source_er']),source.parent/'ER_mask_line.tif'];hashes=[sm['source_fluo_sha256'],sm['source_er_sha256'],sm['outputs']['ER_mask_line']['sha256']]
  assert all(sha(p)==h for p,h in zip(paths,hashes)),cell
  sources.append(dict(cell=cell,manifest=str(source),sha256=hashes,paths=[str(p) for p in paths]))
  for a in anchors:
   assert sha(OLD/a['file'])==a['sha256'];records.append(dict(a,role='old_anchor',path=str((OLD/a['file']).resolve())))
  rows=[];depth,h,w=sm['source_shape'];az={r['z'] for r in anchors};zs=[]
  # Eight evenly spaced reference positions. Snap deterministically away from
  # old anchor planes by >=4 slices; this does not make biological cells independent.
  for q in np.linspace(.08,.92,8):
   target=int(round((depth-1)*q));allowed=[z for z in range(depth) if all(abs(z-v)>=4 for v in az) and all(abs(z-v)>=4 for v in zs)]
   if allowed:zs.append(min(allowed,key=lambda z:(abs(z-target),z)))
  with tif.TiffFile(paths[0]) as ff,tif.TiffFile(paths[1]) as ef,tif.TiffFile(paths[2]) as mf:
   assert all(tuple(f.series[0].shape)==(depth,h,w) for f in [ff,ef,mf])
   for z in sorted(zs):
    fl=ff.pages[z].asarray();er=ef.pages[z].asarray();mask=mf.pages[z].asarray();assert np.isin(mask,[0,255]).all();mask=mask>0;selected=pick(mask)
    if len(selected)<3:skipped.append(dict(cell=cell,z=z,reason='Fewer than three nonoverlapping positive windows',count=len(selected)))
    for density,(n,y,x) in enumerate(selected):
     sl=np.s_[y:y+320,x:x+320];f=fl[sl];e=er[sl];pos=mask[sl]
     # Retain the prior ROI positive derivation; label conversion follows the
     # approved crop-level yellow-positive/gray-blue-background convention.
     roi=ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_dilation(pos,iterations=8),iterations=4));roi[:5]=False;roi[-5:]=False;roi[:,:5]=False;roi[:,-5:]=False
     lab=(pos&roi).astype('uint8');digest=hashlib.sha256(f.tobytes()).hexdigest()
     assert digest not in seen;seen.add(digest)
     name=f'{cell}_z{z:03d}_y{y}_x{x}.npz';p=O/'data'/name;lo,hi=np.percentile(f,[1,99.8]);np.savez_compressed(p,fluo=f,er=e,label=lab,low=lo,high=hi)
     with np.load(p) as check:assert np.array_equal(check['fluo'],f) and np.array_equal(check['label'],lab) and np.array_equal(check['er'],e)
     records.append(dict(file=name,path=str(p.resolve()),sha256=sha(p),cell=cell,role='expanded_candidate',z=z,y=y,x=x,shape=[320,320],density_stratum=density,foreground_fraction=float(lab.mean()),fluo_content_sha256=digest))
     rows.append((name,f,lab))
  sheet=Image.new('RGB',(960,220*((len(rows)+2)//3)));d=ImageDraw.Draw(sheet)
  for i,(name,f,lab) in enumerate(rows):
   over=np.repeat(gray(f)[...,None],3,2);over[lab>0]=[255,220,0];pair=Image.new('RGB',(640,320));pair.paste(Image.fromarray(gray(f)).convert('RGB'),(0,0));pair.paste(Image.fromarray(over),(320,0));pair=pair.resize((320,160))
   xx=(i%3)*320;yy=(i//3)*220;sheet.paste(pair,(xx,yy+25));d.text((xx+3,yy+4),name.split('_NZ_')[-1],fill='white')
  sheet.save(O/(cell+'_review.png'));sheets.append(cell);print(cell,'new',len(rows),flush=True)
 folds=[]
 for i,held in enumerate(cells):
  traincells=[c for c in cells if c!=held];folds.append(dict(fold=i,held_out_specimen=held,train_specimens=traincells,test_files=[r['file'] for r in records if r['cell']==held and r['role']=='old_anchor'],control_train_files=[r['file'] for r in records if r['cell'] in traincells and r['role']=='old_anchor'],expanded_train_files=[r['file'] for r in records if r['cell'] in traincells],unused_heldout_new_files=[r['file'] for r in records if r['cell']==held and r['role']=='expanded_candidate']))
 for fold in folds:
  assert not set(fold['expanded_train_files'])&set(fold['test_files']);assert len(fold['test_files'])==8
 assert len({f for fold in folds for f in fold['test_files']})==40
 m=dict(status='CANDIDATE_DATA_PREPARED_NO_TRAINING',selection='8 fixed Z positions, >=4 slices from old anchors and other selected Z; up to3 nonoverlapping320pxwindows at20/50/80th percentile foreground counts among>=2%windows',label_semantics='1 existing ER-derived positive within historical ROI; 0 remaining crop pixels per user convention; new crops not individually reviewed, pseudo-labels not manual truth',sources=sources,records=records,shortfalls=skipped,folds=folds,limitations='Same five previously inspected specimens; grouped exploratory CV, not independent biological validation; density selection uses teacher mask and does not represent entire image distribution')
 (O/'manifest.json').write_text(json.dumps(m,indent=2));(O/'verification.json').write_text(json.dumps(dict(status='PASS',old_anchors=40,new_candidates=len(records)-40,source_hashes_verified=True,roundtrip_verified=True,folds=5,test_anchor_coverage_once=True,training_started=False),indent=2))
 print('TOTAL',len(records)-40,'new;',len(records),'including anchors')
if __name__=='__main__':main()
