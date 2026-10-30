"""Read-only verification of nested decisions, outer metrics, and saved models."""
from pathlib import Path
import json,hashlib,zipfile,csv
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'outputs/er_inner_convergence_20261030';S=O/'extracted/results/study'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (O/'extracted/exit_status.txt').read_text().strip()=='0'
 m=json.loads((O/'protocol.json').read_text());settings=m['settings'];done=json.loads((S/'complete.json').read_text());assert done['status']=='COMPLETE' and done['models']==10
 assert json.loads((S/'run_lock.json').read_text())['protocol_sha256']==sha(O/'protocol.json')
 lock=json.loads((S/'outer_evaluation_lock.json').read_text());assert lock['protocol_sha256']==sha(O/'protocol.json')
 by={r['file']:r for r in m['records']};folds=[];rows=[];seen=set();overview=Image.new('RGB',(1280,1750));od=ImageDraw.Draw(overview);checkpoints=[]
 for f in m['folds']:
  fid=f['fold'];fold=dict(fold=fid,outer=f['outer_specimen'],inner=f['inner_specimen'],arms={});pan=Image.new('RGB',(1280,2800));pd=ImageDraw.Draw(pan);values={};init=[]
  for arm in ['control','expanded']:
   d=S/f'fold{fid}'/arm;decision=json.loads((d/'decision.json').read_text());assert sha(d/'decision.json')==lock['selection_decision_hashes'][f'{fid}/{arm}']
   assert decision['outer_evaluated'] is False and decision['checkpoint_sha256']==sha(d/'candidate.keras');init.append(decision['initial_sha256'])
   for fname in ['candidate.keras','step1000.keras']:
    with zipfile.ZipFile(d/fname) as z:assert z.testzip() is None
    checkpoints.append(dict(file=str((d/fname).relative_to(O)),sha256=sha(d/fname),bytes=(d/fname).stat().st_size))
   h=json.loads((d/'history.json').read_text());assert [v['step'] for v in h]==list(range(250,decision['completed_step']+1,250))
   # Replay the frozen rule independently, including patience and eligible steps.
   best=None;bs=None;stale=0
   for v in h:
    assert np.isfinite(v['loss']) and np.isfinite(v['inner_dice'])
    if v['step']<1000:continue
    assert stale<4
    if best is None or v['inner_dice']>best+settings['min_delta']:best=v['inner_dice'];bs=v['step'];stale=0
    else:stale+=1
   assert bs==decision['selected_step'] and best==decision['inner_dice']
   assert (stale>=4)==decision['stopped_by_patience'];assert decision['completed_step']==4000 or stale>=4
   assert all(by[n]['cell'] in f['training_specimens'] for n in f[arm+'_train'])
   metrics=json.loads((d/'outer_metrics.json').read_text());assert len(metrics)==8 and {v['file'] for v in metrics}==set(f['outer_reserved']);values[arm]={}
   for name in f['outer_reserved']:
    with np.load(d/name) as z:v={k:z[k] for k in z.files}
    assert all(v[k].shape==(320,320) for k in ['input','label','probability']);assert np.isfinite(v['probability']).all() and ((v['probability']>=0)&(v['probability']<=1)).all()
    with np.load(O/'payload/data'/name) as z:
     assert np.array_equal(v['label'],z['label']);x=z['fluo'].astype('float32');lo,hi=np.percentile(x,[1,99.8]);assert np.allclose(v['input'],np.clip((x-lo)/(hi-lo),0,1),atol=1e-6)
    p=v['probability']>=.5;t=v['label']>0;tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum());rr=next(q for q in metrics if q['file']==name);assert (tp,fp,fn)==(rr['tp'],rr['fp'],rr['fn']);values[arm][name]=v
    assert abs(rr['dice']-2*tp/max(2*tp+fp+fn,1))<1e-12
   fold['arms'][arm]=dict(selected_step=bs,completed_step=decision['completed_step'],inner_dice=best,inner_dice_step1000=next(v['inner_dice'] for v in h if v['step']==1000),stopped_by_patience=decision['stopped_by_patience'],seconds=h[-1]['seconds'],**{k:float(np.mean([v[k] for v in metrics])) for k in ['dice','iou','precision','recall']})
  assert init[0]==init[1]
  for i,name in enumerate(f['outer_reserved']):
   assert name not in seen;seen.add(name);c=values['control'][name];e=values['expanded'][name];assert np.array_equal(c['input'],e['input']) and np.array_equal(c['label'],e['label'])
   row=dict(fold=fid,file=name,cell=f['outer_specimen'],foreground_fraction=float(c['label'].mean()))
   for arm,v in [('control',c),('expanded',e)]:
    p=v['probability']>=.5;t=v['label']>0;tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
    for k,val in dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),fp=fp,fn=fn).items():row[arm+'_'+k]=val
   row['dice_delta']=row['expanded_dice']-row['control_dice'];rows.append(row)
   for j,(v,title) in enumerate([(c['input'],'Fluo'),(c['label'],'ER-derived target'),(c['probability']>=.5,'Original pool'),(e['probability']>=.5,'Expanded pool')]):
    img=Image.fromarray(np.uint8(np.clip(v,0,1)*255)).convert('RGB');pan.paste(img,(j*320,i*350+30));pd.text((j*320+4,i*350+4),title,fill='white')
    if i==0:overview.paste(img,(j*320,fid*350+30));od.text((j*320+4,fid*350+4),f'{title} fold{fid}',fill='white')
  fold['delta']=fold['arms']['expanded']['dice']-fold['arms']['control']['dice'];folds.append(fold);pan.save(O/f'fold{fid}_all8.png')
 assert len(seen)==40 and len(checkpoints)==20
 summary={}
 for arm in ['control','expanded']:
  a=np.array([r[arm+'_dice'] for r in rows]);fm=[f['arms'][arm]['dice'] for f in folds]
  summary[arm]=dict(dice_mean=float(a.mean()),crop_std=float(a.std()),specimen_std=float(np.std(fm)),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),**{k:float(np.mean([r[arm+'_'+k] for r in rows])) for k in ['iou','precision','recall']})
 result=dict(status='VERIFIED',summary=summary,folds=folds,paired=dict(mean_dice_delta=float(np.mean([r['dice_delta'] for r in rows])),improved_crops=sum(r['dice_delta']>0 for r in rows),improved_folds=sum(f['delta']>0 for f in folds)),production_approved=False)
 (O/'review_metrics.json').write_text(json.dumps(result,indent=2));(O/'checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2));overview.save(O/'five_fold_overview.png')
 with (O/'paired_oof.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
