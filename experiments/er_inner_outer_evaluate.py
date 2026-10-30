"""Outer evaluation only after all ten inner-selection decisions are frozen."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads(a.protocol.read_text());complete=json.loads((a.output/'selection_complete.json').read_text());assert complete['models']==10 and complete['protocol_sha256']==sha(a.protocol)
 decisions={}
 for f in m['folds']:
  for arm in ['control','expanded']:
   d=a.output/f"fold{f['fold']}"/arm;decision=json.loads((d/'decision.json').read_text());assert sha(d/'candidate.keras')==decision['checkpoint_sha256'];decisions[f"{f['fold']}/{arm}"]=sha(d/'decision.json')
 (a.output/'outer_evaluation_lock.json').write_text(json.dumps({'selection_decision_hashes':decisions,'protocol_sha256':sha(a.protocol)},indent=2))
 import tensorflow as tf
 g=tf.config.list_physical_devices('GPU');assert len(g)==1;tf.config.experimental.set_memory_growth(g[0],True)
 records={r['file']:r for r in m['records']}
 for f in m['folds']:
  for arm in ['control','expanded']:
   d=a.output/f"fold{f['fold']}"/arm;tf.keras.backend.clear_session();model=tf.keras.models.load_model(d/'candidate.keras',compile=False);rows=[]
   for name in f['outer_reserved']:
    assert records[name]['cell']==f['outer_specimen'];path=a.data/name;assert sha(path)==records[name]['sha256']
    with np.load(path) as z:
     x=z['fluo'].astype('float32');y=z['label'];lo,hi=np.percentile(x,[1,99.8]);assert hi>lo;x=np.clip((x-lo)/(hi-lo),0,1)
    pr=model(x[None,...,None],training=False).numpy()[0,...,0];assert np.isfinite(pr).all();pred=pr>=m['settings']['threshold'];t=y>0;tp=int((pred&t).sum());fp=int((pred&~t).sum());fn=int((~pred&t).sum())
    rows.append(dict(file=name,cell=f['outer_specimen'],tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1)))
    np.savez_compressed(d/name,input=x,label=y,probability=pr)
   (d/'outer_metrics.json').write_text(json.dumps(rows,indent=2))
 for key,h in decisions.items():
  fid,arm=key.split('/');assert sha(a.output/f'fold{fid}'/arm/'decision.json')==h
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','folds':5,'models':10,'selection_unchanged_after_outer':True,'production_approved':False},indent=2));print('INNER_STUDY_COMPLETE',flush=True)
if __name__=='__main__':main()
