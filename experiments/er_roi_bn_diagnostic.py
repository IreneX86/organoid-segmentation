"""Diagnostic only: same weights, disable dropout, compare stored vs batch BN stats.

No optimizer, gradients, calibration pass, checkpoint write, or model selection.
BN batch mode uses the tested batch's image statistics and is NOT deployable evidence.
"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import tensorflow as tf
from er_roi_student import load
R=Path(__file__).resolve().parents[1];D=R/'outputs/er_roi_student_run_20261029/extracted/results/student';DATA=R/'outputs/er_roi_student_ready_20261029'
def measure(y,p):
 v=y!=255;t=y==1;pred=p>=.5
 tp=int((pred&t&v).sum());fp=int((pred&~t&v).sum());fn=int((~pred&t&v).sum())
 return dict(dice=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),positive_fraction=float(pred.mean()),positive_probability=float(p[t].mean()),negative_probability=float(p[y==0].mean()),tp=tp,fp=fp,fn=fn)
def main():
 ck=D/'er_roi_student.keras';before=hashlib.sha256(ck.read_bytes()).hexdigest();model=tf.keras.models.load_model(ck,compile=False)
 for layer in model.layers:
  if isinstance(layer,tf.keras.layers.Dropout):layer.rate=0.
 bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
 original=model.get_weights();rows=json.loads((DATA/'manifest.json').read_text())['records'];result={}
 for split in ['train','validation']:
  records=[r for r in rows if r['split']==split];ys=[];stored=[];batch=[]
  for offset in range(0,len(records),4):
   pairs=[load(DATA,r) for r in records[offset:offset+4]];x=np.stack([a for a,b in pairs]);y=np.stack([b for a,b in pairs]);ys.append(y)
   model.set_weights(original);stored.append(model(x,training=False).numpy())
   # Disposable in-memory stats changes only; restore before every diagnostic batch.
   batch.append(model(x,training=True).numpy())
  y=np.concatenate(ys);a=np.concatenate(stored);b=np.concatenate(batch)
  result[split]=dict(stored_BN=measure(y,a),batch_BN_dropout_off=measure(y,b),label_fraction=dict(positive=float((y==1).mean()),negative=float((y==0).mean()),ignored=float((y==255).mean())))
 result['bn_momentum']=[float(l.momentum) for l in bn];result['initial_stats_weight_at_best_epoch']=.99**72
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==before
 result['checkpoint_unchanged']=True;result['caveat']='Batch-stat diagnostic is batch-dependent and uses evaluated image stats. Not a production fix or independent accuracy.'
 (D/'bn_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
