"""GPU-only training-capacity diagnostic; never a production model.
One training crop, fixed selection, existing partial labels unchanged.
Frozen calibrated BN and dropout-off isolate gradient fitting from BN drift.
No validation, augmentation, threshold selection or pseudo-label edits.
"""
from pathlib import Path
import argparse,json,sys,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 import tensorflow as tf
 from er_roi_student import load
 from er_partial_supervision import masked_bce_dice
 gpu=tf.config.list_physical_devices('GPU');assert len(gpu)==1,'GPU required; no CPU training'
 tf.config.experimental.set_memory_growth(gpu[0],True)
 tf.keras.utils.set_random_seed(20261029)
 with tf.device('/GPU:0'):
  t=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in t.device
 m=json.loads((a.data/'manifest.json').read_text());r=sorted([r for r in m['records'] if r['split']=='train'],key=lambda r:r['file'])[0]
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(a.data/r['file'])==r['sha256'];source_sha=sha(a.checkpoint)
 x,y=load(a.data,r);x=x[None];y=y[None]
 model=tf.keras.models.load_model(a.checkpoint,compile=False)
 for l in model.layers:
  if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
 opt=tf.keras.optimizers.Adam(.0003)
 a.output.mkdir(parents=True,exist_ok=False)
 @tf.function
 def step():
  with tf.GradientTape() as tape:
   p=model(x,training=False);loss=masked_bce_dice(y,p)
  grads=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(grads,model.trainable_variables));return loss
 before=model(x,training=False).numpy();history=[];start=time.time()
 for i in range(1,501):
  loss=float(step());assert np.isfinite(loss)
  if i%25==0:history.append(dict(step=i,loss=loss,seconds=time.time()-start));print(history[-1],flush=True)
  if time.time()-start>600:break
 after=model(x,training=False).numpy();assert np.isfinite(after).all()
 np.savez_compressed(a.output/'diagnostic.npz',fluo=x,label=y,before=before,after=after)
 model.save(a.output/'diagnostic_only.keras');assert sha(a.checkpoint)==source_sha
 (a.output/'result.json').write_text(json.dumps(dict(crop=r,source_checkpoint_sha256=source_sha,history=history,production_approved=False,interpretation='Training capacity on known pixels only; cannot demonstrate accuracy on ignored pixels or generalization'),indent=2))
if __name__=='__main__':main()
