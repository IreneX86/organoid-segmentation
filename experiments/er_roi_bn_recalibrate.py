"""One fixed train-input-only, layerwise population-moment BN recalibration.
No optimizer, label-dependent fitting, validation calibration or threshold scan.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import tensorflow as tf
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from er_roi_student import load
from er_roi_bn_diagnostic import measure
DATA=R/'outputs/er_roi_student_ready_20261029'
SRC=R/'outputs/er_roi_student_run_20261029/extracted/results/student'
OUT=R/'outputs/er_roi_bn_recalibration_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 protocol=dict(method='topological layerwise exact pooled activation moments; upstream recalibrated BN in inference mode; dropout off via training=False',fit='32 training Fluo crops only; no labels',evaluation='8 fixed validation crops; threshold0.5; no repeat or selection by metrics',production_approved=False)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 ck=SRC/'er_roi_student.keras';original_sha=sha(ck);model=tf.keras.models.load_model(ck,compile=False)
 weights=[v.numpy().copy() for v in model.trainable_weights]
 manifest=json.loads((DATA/'manifest.json').read_text());train=[r for r in manifest['records'] if r['split']=='train'];val=[r for r in manifest['records'] if r['split']=='validation']
 assert len(train)==32 and len(val)==8 and not {r['cell'] for r in train}&{r['cell'] for r in val}
 xx=[]
 for r in train:
  assert sha(DATA/r['file'])==r['sha256']
  # Read only input and its fixed normalization parameters during calibration.
  with np.load(DATA/r['file']) as a:xx.append(np.clip((a['fluo'].astype('float32')-float(a['low']))/(float(a['high'])-float(a['low'])),0,1)[...,None])
 stats=[]
 for layer in model.layers:
  if not isinstance(layer,tf.keras.layers.BatchNormalization):continue
  probe=tf.keras.Model(model.inputs,layer.input);total=0;mean=None;m2=None
  for x in xx:
   activation=probe(x[None],training=False)
   mu,var=tf.nn.moments(activation,axes=[0,1,2]);mu=mu.numpy().astype('float64');var=var.numpy().astype('float64');n=int(np.prod(activation.shape[:-1]))
   if mean is None:mean=mu;m2=var*n;total=n
   else:
    delta=mu-mean;nt=total+n;m2+=var*n+delta*delta*total*n/nt;mean+=delta*n/nt;total=nt
  oldmean=layer.moving_mean.numpy().copy();oldvar=layer.moving_variance.numpy().copy()
  layer.moving_mean.assign(mean);layer.moving_variance.assign(m2/total)
  stats.append(dict(layer=layer.name,n=total,mean_shift_l2=float(np.linalg.norm(mean-oldmean)),variance_ratio_median=float(np.median((m2/total)/np.maximum(oldvar,1e-8)))))
  print('CALIBRATED',layer.name,flush=True)
 assert all(np.array_equal(a,v.numpy()) for a,v in zip(weights,model.trainable_weights))
 calibrated=OUT/'er_roi_student_train_bn.keras';model.save(calibrated)
 # Verify saved candidate, not transient batch-stat inference.
 model=tf.keras.models.load_model(calibrated,compile=False);old=tf.keras.models.load_model(ck,compile=False)
 results={};views=[]
 for split,rows in [('train',train),('validation',val)]:
  labels=[];before=[];after=[]
  for r in rows:
   x,y=load(DATA,r);a=old(x[None],training=False).numpy()[0];b=model(x[None],training=False).numpy()[0]
   assert np.isfinite(b).all();labels.append(y);before.append(a);after.append(b)
   if split=='validation':
    d=OUT/r['file'].replace('.npz','');d.mkdir();tif.imwrite(d/'probability.tif',b[...,0]);tif.imwrite(d/'mask.tif',(b[...,0]>=.5).astype('uint8')*255)
    gray=np.repeat((x*255).astype('uint8'),3,2);overlay=gray.copy();overlay[b[...,0]>=.5]=[255,220,0]
    gt=np.full((320,320,3),65,np.uint8);gt[y[...,0]==1]=[255,220,0];gt[y[...,0]==0]=[0,130,255]
    v=Image.new('RGB',(1600,350));draw=ImageDraw.Draw(v)
    imgs=[gray,gt,np.repeat((a>=.5).astype('uint8')*255,3,2),np.repeat((b>=.5).astype('uint8')*255,3,2),overlay]
    for j,(img,title) in enumerate(zip(imgs,['Fluo','Pseudo positive / weak negative / ignore','Original mask 0.5','Recalibrated mask 0.5','Recalibrated overlay'])):
     v.paste(Image.fromarray(img),(j*320,30));draw.text((j*320+4,6),title,fill='white')
    v.save(d/'review.png');views.append(v)
  y=np.stack(labels);results[split]=dict(before=measure(y,np.stack(before)),after=measure(y,np.stack(after)))
 # Same frozen model must give same result independently of inference batch composition.
 x0,_=load(DATA,val[0]);x1,_=load(DATA,val[1]);one=model(x0[None],training=False).numpy()[0];two=model(np.stack([x0,x1]),training=False).numpy()[0]
 delta=float(np.max(np.abs(one-two)));assert np.allclose(one,two,atol=1e-5)
 assert sha(ck)==original_sha
 canvas=Image.new('RGB',(1600,350*len(views)))
 for i,v in enumerate(views):canvas.paste(v,(0,350*i))
 canvas.save(OUT/'validation_before_after.png')
 result=dict(metrics=results,statistics=stats,source_checkpoint_sha256=original_sha,candidate_sha256=sha(calibrated),trainable_weights_identical=True,original_unchanged=True,inference_batch_max_difference=delta,calibration_cells=sorted({r['cell'] for r in train}),no_validation_used_for_calibration=True,production_approved=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
