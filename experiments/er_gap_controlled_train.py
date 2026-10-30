"""Fixed mesh-only versus mesh+gap supervision diagnostic; no validation selection."""
from pathlib import Path
import json,sys,time,hashlib,argparse
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import tensorflow as tf
 from er_roi_student import load
 from er_partial_supervision import masked_bce_dice
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 gpu=tf.config.list_physical_devices('GPU');assert len(gpu)==1
 tf.config.experimental.set_memory_growth(gpu[0],True)
 with tf.device('/GPU:0'):
  z=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in z.device
 data=R/'data';mesh=R/'mesh';checkpoint=R/'source.keras';ckhash=sha(checkpoint)
 m=json.loads((data/'manifest.json').read_text());w=json.loads((mesh/'manifest.json').read_text());train=[r for r in m['records'] if r['split']=='train'];val=[r for r in m['records'] if r['split']=='validation'];cells=sorted(set(r['cell'] for r in train));assert len(cells)==4 and len(val)==8 and not set(cells)&set(r['cell'] for r in val)
 def read(p,r):
  assert sha(p/r['file'])==r['sha256'];return load(p,r)
 control={c:[read(mesh,dict(file=r['output'],sha256=r['sha256'])) for r in w['records'] if r['cell']==c] for c in cells}
 gap=R/'gap';g=json.loads((gap/'manifest.json').read_text())
 windows={c:[read(gap,dict(file=r['output'],sha256=r['sha256'])) for r in g['records'] if r['cell']==c] for c in cells}
 # Same crop order/input, and ONLY ignored-to-negative target changes.
 for c in cells:
  assert len(control[c])==len(windows[c])==4
  for (x,y),(xx,yy) in zip(control[c],windows[c]):
   assert np.array_equal(x,xx) and np.array_equal(y==1,yy==1)
   assert np.all((y==yy)|((y==255)&(yy==0)))
 validation=[(r,read(data,r)) for r in val]
 conf=dict(arms=['control','mesh'],arm_semantics={'control':'original 16 mesh windows','mesh':'same 16 windows plus1832 provisional gap negatives'},updates=500,batch_size=4,sampling='one uniform crop per training specimen per update',lr=.0003,seed=20261029,BN='frozen including affine weights; train-input calibrated source',dropout='off in both arms',augmentation='none in both arms',threshold=.5,selection='fixed final update, no validation selection',source_sha256=ckhash,production_approved=False)
 (a.output/'protocol.json').write_text(json.dumps(conf,indent=2));(a.output/'environment.json').write_text(json.dumps(dict(tf=tf.__version__,build=tf.sysconfig.get_build_info(),gpu=str(gpu)),indent=2))
 for arm,groups in [('control',control),('mesh',windows)]:
  dest=a.output/arm;dest.mkdir();tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(conf['seed']);model=tf.keras.models.load_model(checkpoint,compile=False)
  for layer in model.layers:
   if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
  opt=tf.keras.optimizers.Adam(conf['lr']);rng=np.random.default_rng(conf['seed'])
  @tf.function
  def step(x,y):
   with tf.GradientTape() as tape:p=model(x,training=False);loss=masked_bce_dice(y,p)
   grads=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(grads,model.trainable_variables));return loss
  start=time.time();hist=[]
  for n in range(1,501):
   batch=[groups[c][rng.integers(len(groups[c]))] for c in cells];loss=float(step(np.stack([b[0] for b in batch]),np.stack([b[1] for b in batch])));assert np.isfinite(loss)
   if n%50==0:
    row=dict(step=n,loss=loss,seconds=time.time()-start);hist.append(row);print(arm,row,flush=True)
   if time.time()-start>600:raise TimeoutError('Arm exceeded 10min guard; no incomplete comparison promotion')
  model.save(dest/'candidate.keras');metrics=[]
  # Evaluate fixed validation once AFTER fixed updates; never select a checkpoint.
  for r,(x,y) in validation:
   p=model(x[None],training=False).numpy()[0,...,0];y=y[...,0];assert np.isfinite(p).all();pred=p>=.5;truth=y==1
   tp=int((pred&truth).sum());fp=int((pred&(y==0)).sum());fn=int((~pred&truth).sum())
   metrics.append(dict(file=r['file'],tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),positive_fraction=float(pred.mean())))
   np.savez_compressed(dest/r['file'],fluo=x[...,0],label=y.astype('uint8'),probability=p)
  (dest/'result.json').write_text(json.dumps(dict(history=hist,metrics=metrics,seconds=time.time()-start,checkpoint_sha256=sha(dest/'candidate.keras')),indent=2))
 assert sha(checkpoint)==ckhash
 (a.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',arms=2,production_approved=False),indent=2));print('ER_GAP_CONTROLLED_COMPLETE',flush=True)
if __name__=='__main__':main()
