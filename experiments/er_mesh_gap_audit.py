"""Read-only train-window audit of supervised gaps and cross-channel contrast."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata,spearmanr
from PIL import Image,ImageDraw
import tifffile as tif
R=Path(__file__).resolve().parents[1];M=R/'outputs/er_mesh_windows_20261029';D=R/'outputs/er_interline_student_ready_20261029';REF=R/'outputs/er_expanded_roi_candidates_20261029';O=R/'outputs/er_mesh_gap_audit_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(x,a,b):
 u=x[a];v=x[b]
 if not len(u) or not len(v):return None
 ranks=rankdata(np.r_[u,v]);return float((ranks[:len(u)].sum()-len(u)*(len(u)+1)/2)/(len(u)*len(v)))
def gray(x):
 lo,hi=np.percentile(x,[1,99.8]);a=np.clip((x-lo)/max(hi-lo,1),0,1);return np.repeat((255*a[...,None]).astype('uint8'),3,2)
def main():
 O.mkdir(exist_ok=False);records=json.loads((M/'manifest.json').read_text())['records'];refs={v['directory']:v for v in json.loads((REF/'manifest.json').read_text())['records']};rows=[];panels=[];percell={}
 for n,r in enumerate(records):
  p=D/r['file'];assert sha(p)==r['source_sha256']
  with np.load(p) as f:x=f['fluo'].astype(float);label=f['label']
  ep=REF/r['file'][:-4]/'ER_reference.tif';assert sha(ep)==refs[r['file'][:-4]]['hashes']['ER_reference'];er=tif.imread(ep).astype(float)
  i,j=r['wy'],r['wx'];x=x[i:i+64,j:j+64];er=er[i:i+64,j:j+64];y=label[i:i+64,j:j+64];pos=y==1
  gaps=ndi.binary_fill_holes(pos)&~pos;l,num=ndi.label(gaps);sz=np.bincount(l.ravel());ids=np.flatnonzero((sz>=4)&(sz<=256));ids=ids[ids!=0];g=np.isin(l,ids);rim=ndi.binary_dilation(g)&pos
  gaplist=[]
  for k in ids:
   z=l==k;gaplist.append(dict(area=int(z.sum()),negative=int(((y==0)&z).sum()),ignored=int(((y==255)&z).sum())))
  row=dict(window=r['output'],cell=r['cell'],z=r['z'],gap_pixels=int(g.sum()),negative_gap_pixels=int((g&(y==0)).sum()),ignored_gap_pixels=int((g&(y==255)).sum()),gaps=len(ids),fully_ignored_gaps=sum(v['negative']==0 for v in gaplist),ER_rim_vs_gap_auc=auc(er,rim,g),Fluo_rim_vs_gap_auc=auc(x,rim,g),ER_rim_minus_gap=float(np.median(er[rim])-np.median(er[g])),Fluo_rim_minus_gap=float(np.median(x[rim])-np.median(x[g])),ER_Fluo_spearman=float(spearmanr(er.ravel(),x.ravel()).statistic),gap_details=gaplist)
  rows.append(row)
  sem=np.full((64,64,3),55,np.uint8);sem[pos]=[255,220,0];sem[y==0]=[0,130,255]
  gapview=gray(x);gapview[g&(y==255)]=[255,70,190];gapview[g&(y==0)]=[0,220,255]
  sheet=Image.new('RGB',(1024,290));dr=ImageDraw.Draw(sheet);dr.text((4,3),f'{r["output"]} {r["cell"]} z{r["z"]} | gap unknown {row["ignored_gap_pixels"]}/{row["gap_pixels"]}',fill='white')
  for k,(a,title) in enumerate(zip([gray(er),gray(x),sem,gapview],['ER reference','Fluo input','yellow + / blue weak - / gray ignore','gaps: pink ignored / cyan weak -'])):
   sheet.paste(Image.fromarray(a).resize((256,256),Image.Resampling.NEAREST),(k*256,34));dr.text((k*256+2,19),title,fill='white')
  sheet.save(O/f'{n:02d}_gap_review.png')
  if percell.get(r['cell'],0)<2:panels.append(sheet);percell[r['cell']]=percell.get(r['cell'],0)+1
  assert sha(p)==r['source_sha256'] and sha(ep)==refs[r['file'][:-4]]['hashes']['ER_reference']
 totals={k:sum(v[k] for v in rows) for k in ['gap_pixels','negative_gap_pixels','ignored_gap_pixels','gaps','fully_ignored_gaps']};totals['ignored_gap_fraction']=totals['ignored_gap_pixels']/totals['gap_pixels']
 totals['median_ER_auc']=float(np.median([v['ER_rim_vs_gap_auc'] for v in rows]));totals['median_Fluo_auc']=float(np.median([v['Fluo_rim_vs_gap_auc'] for v in rows]));totals['Fluo_auc_below_0_6']=sum(v['Fluo_rim_vs_gap_auc']<.6 for v in rows)
 result=dict(totals=totals,records=rows,source_verification='PASS',validation_opened=False,new_labels=0,scope='16 previously selected training windows; geometric gaps are not certified biological negatives; raw-intensity AUC is not model accuracy')
 (O/'audit.json').write_text(json.dumps(result,indent=2));out=Image.new('RGB',(1024,len(panels)*290))
 for k,p in enumerate(panels):out.paste(p,(0,k*290))
 out.save(O/'eight_gap_reviews.png');print(json.dumps(totals,indent=2))
if __name__=='__main__':main()
