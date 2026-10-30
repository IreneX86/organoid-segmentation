"""Read-only replay of frozen crop sampling and label-visibility diagnostics.

No model fitting, threshold selection, relabelling, or outer-test selection.
Visual examples are fixed median-density crops, never chosen by model score.
"""
from pathlib import Path
import json,csv
import numpy as np
import tifffile
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
O=R/'outputs/er_diverse_exposure_audit_20261030'
D=R/'outputs/er_diverse_cv_20261030'
def norm(x):
 lo,hi=np.percentile(x,[1,99.8]);return np.clip((x.astype('float32')-lo)/max(hi-lo,1),0,1)
def main():
 O.mkdir(exist_ok=False)
 m=json.loads((D/'payload/data/manifest.json').read_text())
 src=json.loads((R/'outputs/er_diverse_roi_20261030/manifest.json').read_text())
 sources={s['cell']:s for s in src['sources']};data={};records={r['file']:r for r in m['records']};stats=[]
 for r in m['records']:
  with np.load(D/'payload/data'/r['file']) as a:x=norm(a['fluo']);y=a['label'].astype(bool)
  data[r['file']]=(x,y)
  stats.append(dict(file=r['file'],cell=r['cell'],role=r['role'],positive_fraction=float(y.mean()),fluo_positive_minus_background=float(x[y].mean()-x[~y].mean())))
 replay=[]
 for f in m['folds']:
  for arm,key in [('control','control_train_files'),('expanded','expanded_train_files')]:
   names=f[key];assert all(records[n]['cell']!=f['held_out_specimen'] for n in names)
   cells=sorted(f['train_specimens']);groups={c:[n for n in names if records[n]['cell']==c] for c in cells}
   rng=np.random.default_rng(20261030+f['fold']);counts={n:0 for n in names}
   for _ in range(1000):
    for c in cells:counts[groups[c][rng.integers(len(groups[c]))]]+=1
   h=json.loads((D/f"extracted/results/fold{f['fold']}"/arm/'history.json').read_text())
   anchors=[counts[n] for n in f['control_train_files']]
   replay.append(dict(fold=f['fold'],arm=arm,crops=len(names),draws=sum(counts.values()),mean_anchor_visits=float(np.mean(anchors)),min_anchor_visits=min(anchors),max_anchor_visits=max(anchors),sampled_positive_fraction=float(sum(counts[n]*data[n][1].mean() for n in names)/4000),loss_at_500=h[4]['loss'],loss_at_1000=h[9]['loss'],mean_loss_100_300=float(np.mean([v['loss'] for v in h[:3]])),mean_loss_800_1000=float(np.mean([v['loss'] for v in h[-3:]]))))
 cells=sorted(sources);sheet=Image.new('RGB',(1920,5*350));draw=ImageDraw.Draw(sheet);examples=[]
 for i,c in enumerate(cells):
  for j,role in enumerate(['old_anchor','expanded_candidate']):
   candidates=sorted([r for r in stats if r['cell']==c and r['role']==role],key=lambda r:(r['positive_fraction'],r['file']))
   chosen=candidates[len(candidates)//2];rec=records[chosen['file']];x,y=data[chosen['file']]
   # Recover the corresponding ER crop for visibility review only, not model input.
   with tifffile.TiffFile(sources[c]['paths'][1]) as t:er=t.pages[rec['z']].asarray()[rec['y']:rec['y']+320,rec['x']:rec['x']+320]
   en=norm(er);over=np.repeat(np.uint8(en*255)[...,None],3,2);over[y]=[255,220,0]
   for k,(a,title) in enumerate([(np.uint8(x*255),'Fluo'),(np.uint8(en*255),'ER reference'),(over,'ER + fixed label')]):
    xx=j*960+k*320;yy=i*350;sheet.paste(Image.fromarray(a).convert('RGB'),(xx,yy+30));draw.text((xx+4,yy+5),f'{role} {title} fg={chosen["positive_fraction"]:.1%}',fill='white')
   examples.append(chosen)
 sheet.save(O/'five_specimen_label_review.png')
 summary={role:{'n':sum(r['role']==role for r in stats),'mean_positive_fraction':float(np.mean([r['positive_fraction'] for r in stats if r['role']==role])),'mean_fluo_contrast':float(np.mean([r['fluo_positive_minus_background'] for r in stats if r['role']==role]))} for role in ['old_anchor','expanded_candidate']}
 result=dict(status='READ_ONLY_COMPLETE',summary=summary,replay=replay,examples=examples,limitations='Pseudo-label visibility review cannot certify completeness; sampled loss is noisy and not proof of convergence. No outer-test predictions/metrics consulted.')
 (O/'audit.json').write_text(json.dumps(result,indent=2))
 for name,rows in [('crop_stats.csv',stats),('sampling_replay.csv',replay)]:
  with (O/name).open('w') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
