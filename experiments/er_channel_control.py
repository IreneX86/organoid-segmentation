"""Matched ER versus Fluo input control, fixed binary crops and final checkpoints."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import tensorflow as tf
 from model.unet import build_unet_2d
 from er_partial_supervision import masked_bce_dice
 tf.config.experimental.enable_op_determinism()
 gpu=tf.config.list_physical_devices('GPU');assert len(gpu)==1
 tf.config.experimental.set_memory_growth(gpu[0],True)
 with tf.device('/GPU:0'):
  v=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in v.device
 m=json.loads((R/'data/manifest.json').read_text());data=[]
 for r in m['records']:
  p=R/'data'/r['file'];assert sha(p)==r['sha256']
  with np.load(p) as a:data.append((r,{k:a[k] for k in a.files}))
 tr=[a for a in data if a[0]['split']=='train'];va=[a for a in data if a[0]['split']=='validation'];cells=sorted({r['cell'] for r,a in tr});assert len(tr)==32 and len(va)==8 and not set(cells)&{r['cell'] for r,a in va}
 out=Path('/workspace/channel_control_run/results');out.mkdir(exist_ok=True,parents=True)
 tf.keras.utils.set_random_seed(20261030);initial=build_unet_2d(1,16);weights=initial.get_weights();initial.save(out/'initial.keras');del initial
 def norm(x):
  lo,hi=np.percentile(x,[1,99.8]);assert hi>lo
  return np.clip((x.astype('float32')-lo)/(hi-lo),0,1).astype('float32')[...,None]
 for arm in ['fluo','er']:
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(20261030)
  model=build_unet_2d(1,16);model.set_weights(weights)
  # Identical architecture/initial weights and sampling; only input channel changes.
  opt=tf.keras.optimizers.Adam(.0003);rng=np.random.default_rng(20261030)
  pairs={r['file']:(norm(a[arm]),a['label'][...,None].astype('float32')) for r,a in data}
  groups={c:[r for r,a in tr if r['cell']==c] for c in cells}
  @tf.function
  def step(x,y):
   with tf.GradientTape() as tape:
    p=model(x,training=True);loss=masked_bce_dice(y,p)
   opt.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return loss
  dest=out/arm;dest.mkdir(exist_ok=False);history=[];start=time.time()
  for n in range(1,1001):
   batch=[pairs[groups[c][rng.integers(len(groups[c]))]['file']] for c in cells]
   loss=float(step(np.stack([a[0] for a in batch]),np.stack([a[1] for a in batch])));assert np.isfinite(loss)
   if n%100==0:history.append(dict(step=n,loss=loss,seconds=time.time()-start));print(arm,history[-1],flush=True)
   if time.time()-start>600:raise TimeoutError('10min per-arm guard')
  model.save(dest/'candidate.keras');rows=[]
  for r,a in data:
   x,y=pairs[r['file']];p=model(x[None],training=False).numpy()[0,...,0];assert np.isfinite(p).all();t=y[...,0]==1;pr=p>=.5;tp=int((pr&t).sum());fp=int((pr&~t).sum());fn=int((~pr&t).sum())
   rows.append(dict(file=r['file'],split=r['split'],dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),positive_fraction=float(pr.mean())))
   np.savez_compressed(dest/r['file'],input=x[...,0],label=y[...,0].astype('uint8'),probability=p)
  (dest/'metrics.json').write_text(json.dumps(rows,indent=2));(dest/'history.json').write_text(json.dumps(history,indent=2))
  (dest/'checkpoint.sha256').write_text(sha(dest/'candidate.keras'))
 (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',tf=tf.__version__,build=tf.sysconfig.get_build_info(),steps_per_arm=1000,threshold=.5,production_approved=False),indent=2));print('CHANNEL_CONTROL_COMPLETE',flush=True)
if __name__=='__main__':main()
