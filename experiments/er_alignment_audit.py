"""Label-free XY/Z correspondence diagnostic; never resample training data or select models."""
from pathlib import Path
import json,hashlib
import numpy as np
import tifffile as tif
from scipy import ndimage as ndi,signal
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];SRC=R/'outputs/er_expanded_roi_candidates_20261029';OUT=R/'outputs/er_alignment_audit_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):
 l,h=np.percentile(x,[1,99.8]);return np.clip((x.astype('float32')-l)/max(h-l,1),0,1)
def feature(x,kind):
 x=norm(x)[::2,::2]
 return ndi.gaussian_filter(x,1)-ndi.gaussian_filter(x,8) if kind=='detail' else ndi.gaussian_filter(x,3)
def nccmap(a,b):
 # Positive dy/dx means matching ER at (y+dy,x+dx) to the fixed Fluo core.
 m=8;c=a[m:-m,m:-m];c=c-c.mean();n=c.size
 sums=signal.correlate(b,np.ones(c.shape),mode='valid',method='fft')
 sq=signal.correlate(b*b,np.ones(c.shape),mode='valid',method='fft')
 num=signal.correlate(b,c,mode='valid',method='fft')
 return num/np.sqrt(np.maximum(sq-sums*sums/n,1e-12)*np.sum(c*c)+1e-12)
def main():
 OUT.mkdir(exist_ok=False);records=json.loads((SRC/'manifest.json').read_text())['records'];allrows=[];meta=[];images=[]
 for cell in sorted({r['cell'] for r in records}):
  rr=[r for r in records if r['cell']==cell];sm=json.loads(Path(rr[0]['source_manifest']).read_text());paths=[Path(sm[k]) for k in ['source_fluo','source_er']]
  handles=[tif.TiffFile(p) for p in paths];cache=[{},{}]
  def plane(ch,z):
   if z not in cache[ch]:cache[ch][z]=handles[ch].pages[z].asarray()
   return cache[ch][z]
  assert all(tuple(t.series[0].shape)==tuple(sm['source_shape']) for t in handles)
  meta.append(dict(cell=cell,shapes=[list(t.series[0].shape) for t in handles],axes=[t.series[0].axes for t in handles],dtype=[str(t.series[0].dtype) for t in handles],pages=[len(t.pages) for t in handles]))
  for i,r in enumerate(rr):
   z,y,x=r['z_zero_based'],r['y0'],r['x0'];f=plane(0,z)[y:y+320,x:x+320];e=plane(1,z)[y:y+320,x:x+320]
   assert np.array_equal(f,tif.imread(SRC/r['directory']/'Fluo_input.tif')) and np.array_equal(e,tif.imread(SRC/r['directory']/'ER_reference.tif'))
   row=dict(cell=cell,split=r['split'],roi=r['directory'],z=z,source_crop_exact_match=True,features={})
   for kind in ['coarse','detail']:
    a=feature(f,kind);curves=[]
    for dz in [-2,-1,0,1,2]:
     zz=z+dz
     if not 0<=zz<len(handles[1].pages):continue
     b=feature(plane(1,zz)[y:y+320,x:x+320],kind);mat=nccmap(a,b);iy,ix=np.unravel_index(np.argmax(mat),mat.shape)
     curves.append(dict(dz=dz,zero_xy_ncc=float(mat[8,8]),best_ncc=float(mat[iy,ix]),dy=int((iy-8)*2),dx=int((ix-8)*2),search_boundary=bool(iy in [0,16] or ix in [0,16])))
    same=next(v for v in curves if v['dz']==0);best=max(curves,key=lambda v:v['best_ncc']);row['features'][kind]=dict(same_z=same,best_z=best,z_curve=curves)
   allrows.append(row)
   if i==0:
    fn,en=norm(f),norm(e);rgb=np.stack([en,fn,np.zeros_like(fn)],axis=-1);images.append((cell,fn,en,rgb))
  for t in handles:t.close()
 summary=[]
 for cell in sorted({r['cell'] for r in allrows}):
  rr=[r for r in allrows if r['cell']==cell];item=dict(cell=cell,split=rr[0]['split'],n=len(rr))
  for kind in ['coarse','detail']:
   same=[r['features'][kind]['same_z'] for r in rr];best=[r['features'][kind]['best_z'] for r in rr]
   item[kind]=dict(median_zero_ncc=float(np.median([s['zero_xy_ncc'] for s in same])),median_xy_gain=float(np.median([s['best_ncc']-s['zero_xy_ncc'] for s in same])),median_dy=float(np.median([s['dy'] for s in same])),median_dx=float(np.median([s['dx'] for s in same])),dy_range=[min(s['dy'] for s in same),max(s['dy'] for s in same)],dx_range=[min(s['dx'] for s in same),max(s['dx'] for s in same)],boundary_count=sum(s['search_boundary'] for s in same),best_dz=[s['dz'] for s in best],median_z_gain=float(np.median([b['best_ncc']-s['best_ncc'] for s,b in zip(same,best)])))
  summary.append(item)
 panel=Image.new('RGB',(960,350*len(images)));d=ImageDraw.Draw(panel)
 for i,(cell,f,e,rgb) in enumerate(images):
  for j,(im,title) in enumerate([(f,'Fluo'),(e,'ER'),(rgb,'ER red / Fluo green (unshifted)')]):
   panel.paste(Image.fromarray(np.uint8(im*255)).convert('RGB'),(j*320,i*350+30));d.text((j*320+3,i*350+4),title+' '+cell.split('_')[2],fill='white')
 panel.save(OUT/'five_specimen_alignment.png')
 result=dict(scope='Diagnostic only; NCC maxima are not validated registration transforms; labels never used to estimate shifts',search='XY +/-16 original pixels, stride2; ER z offset -2..2; coarse sigma3 and detail DoG1,8 at half resolution',metadata=meta,summary=summary,rows=allrows)
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
