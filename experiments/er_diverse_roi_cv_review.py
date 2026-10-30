"""Verify all paired CV checkpoints/OOF outputs and summarize fixed comparisons."""
from pathlib import Path
import hashlib,json,zipfile,csv
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'outputs/er_diverse_cv_20261030';S=O/'extracted/results'
def main():
 assert (O/'extracted/exit_status.txt').read_text().strip()=='0';assert json.loads((S/'complete.json').read_text())['status']=='COMPLETE'
 manifest_path=O/'payload/data/manifest.json'
 protocol=json.loads((S/'protocol.json').read_text())
 assert protocol['source_manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
 assert protocol['updates']==1000 and protocol['threshold']==.5 and protocol['tf']=='2.21.0'
 manifest=json.loads(manifest_path.read_text());byfile={r['file']:r for r in manifest['records']};allrows=[];folds=[];seen=set();overview=Image.new('RGB',(1280,1750));od=ImageDraw.Draw(overview)
 for f in manifest['folds']:
  fid=f['fold'];fd=S/f'fold{fid}';comp=json.loads((fd/'complete.json').read_text());assert comp['held_out_specimen']==f['held_out_specimen'];rows={};locks={}
  for arm in ['control','expanded']:
   d=fd/arm;ck=d/'candidate.keras';assert hashlib.sha256(ck.read_bytes()).hexdigest()==(d/'checkpoint.sha256').read_text().strip()
   with zipfile.ZipFile(ck) as z:assert z.testzip() is None
   locks[arm]=json.loads((d/'lock.json').read_text());assert locks[arm]['steps']==1000 and locks[arm]['test_files']==f['test_files']
   expected=f['control_train_files'] if arm=='control' else f['expanded_train_files'];assert locks[arm]['training_files']==expected
   assert all(byfile[n]['cell']!=f['held_out_specimen'] for n in expected)
   rows[arm]=json.loads((d/'metrics.json').read_text());assert len(rows[arm])==40
   history=json.loads((d/'history.json').read_text())
   assert [h['step'] for h in history]==list(range(100,1001,100))
   assert all(np.isfinite(h['loss']) and h['seconds']>0 for h in history)
  assert locks['control']['initial_weights_sha256']==locks['expanded']['initial_weights_sha256']==comp['initial_weights_sha256']
  metrics={arm:{split:{k:float(np.mean([r[k] for r in rows[arm] if r['split']==split])) for k in ['dice','iou','precision','recall']} for split in ['test','train_anchor']} for arm in rows}
  folds.append(dict(fold=fid,cell=f['held_out_specimen'],metrics=metrics,dice_delta=metrics['expanded']['test']['dice']-metrics['control']['test']['dice']))
  panel=Image.new('RGB',(1280,2800));pd=ImageDraw.Draw(panel)
  for i,name in enumerate(f['test_files']):
   assert name not in seen;seen.add(name);vals={}
   for arm in ['control','expanded']:
    with np.load(fd/arm/name) as a:vals[arm]={k:a[k] for k in a.files}
    v=vals[arm];assert v['input'].shape==v['label'].shape==v['probability'].shape==(320,320) and np.isfinite(v['probability']).all() and np.isin(v['label'],[0,1]).all()
   c,e=vals['control'],vals['expanded'];assert np.array_equal(c['input'],e['input']) and np.array_equal(c['label'],e['label'])
   with np.load(O/'payload/data'/name) as a:assert np.array_equal(c['label'],a['label'])
   row=dict(fold=fid,cell=f['held_out_specimen'],file=name,foreground_fraction=float(c['label'].mean()))
   for arm in rows:
    metricsrow=next(r for r in rows[arm] if r['file']==name and r['split']=='test')
    # Independently recompute counts against downloaded probabilities.
    pr=vals[arm]['probability']>=.5;t=vals[arm]['label']>0;tp=int((pr&t).sum());fp=int((pr&~t).sum());fn=int((~pr&t).sum());assert (tp,fp,fn)==(metricsrow['tp'],metricsrow['fp'],metricsrow['fn'])
    for k in ['dice','iou','precision','recall','fp','fn']:row[arm+'_'+k]=metricsrow[k]
   row['dice_delta']=row['expanded_dice']-row['control_dice'];allrows.append(row)
   imgs=[c['input'],c['label'],c['probability']>=.5,e['probability']>=.5]
   for j,(im,title) in enumerate(zip(imgs,['Fluo','Fixed binary target','Original crop pool','Expanded crop pool'])):
    image=Image.fromarray(np.uint8(np.clip(im,0,1)*255)).convert('RGB');panel.paste(image,(j*320,i*350+30));pd.text((j*320+3,i*350+5),title,fill='white')
    if i==0:overview.paste(image,(j*320,fid*350+30));od.text((j*320+3,fid*350+5),title+' fold'+str(fid),fill='white')
  panel.save(O/f'fold{fid}_all8.png')
 assert len(seen)==40
 summary={}
 for arm in ['control','expanded']:
  a=np.array([r[arm+'_dice'] for r in allrows]);fm=np.array([f['metrics'][arm]['test']['dice'] for f in folds]);low=[r for r in allrows if r['foreground_fraction']<=.05]
  summary[arm]=dict(mean_dice=float(a.mean()),crop_std_ddof0=float(a.std()),fold_mean_std_ddof0=float(fm.std()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),**{k:float(np.mean([r[arm+'_'+k] for r in allrows])) for k in ['iou','precision','recall']},low_foreground_n=len(low),low_foreground_dice=float(np.mean([r[arm+'_dice'] for r in low])) if low else None,train_anchor_dice=float(np.mean([f['metrics'][arm]['train_anchor']['dice'] for f in folds])))
 delta=np.array([r['dice_delta'] for r in allrows]);result=dict(status='VERIFIED',summary=summary,folds=folds,paired=dict(mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),improved_crops=int((delta>0).sum()),total_crops=40,improved_folds=sum(f['dice_delta']>0 for f in folds)),rows=allrows,production_approved=False)
 (O/'review_metrics.json').write_text(json.dumps(result,indent=2));overview.save(O/'five_fold_overview.png')
 with (O/'paired_oof.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 print(json.dumps({k:result[k] for k in ['summary','paired','folds']},indent=2))
if __name__=='__main__':main()
