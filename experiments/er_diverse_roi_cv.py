"""Frozen five-specimen paired CV: only training crop pool differs between arms."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wd(weights):
 h=hashlib.sha256()
 for w in weights:h.update(w.tobytes())
 return h.hexdigest()
def main():
 import tensorflow as tf
 from model.unet import build_unet_2d
 from er_partial_supervision import masked_bce_dice
 tf.config.experimental.enable_op_determinism();g=tf.config.list_physical_devices('GPU');assert len(g)==1
 tf.config.experimental.set_memory_growth(g[0],True)
 with tf.device('/GPU:0'):
  v=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in v.device
 m=json.loads((R/'data/manifest.json').read_text());records={r['file']:r for r in m['records']};pairs={}
 for r in m['records']:
  p=R/'data'/r['file'];assert sha(p)==r['sha256']
  with np.load(p) as a:
   x=a['fluo'];y=a['label'];assert x.shape==y.shape==(320,320) and np.isin(y,[0,1]).all()
   lo,hi=np.percentile(x,[1,99.8]);assert hi>lo;pairs[r['file']]=(np.clip((x.astype('float32')-lo)/(hi-lo),0,1).astype('float32')[...,None],y[...,None].astype('float32'))
 out=Path('/workspace/diverse_cv_run/results');out.mkdir(exist_ok=True,parents=True)
 (out/'protocol.json').write_text(json.dumps(dict(updates=1000,batch=4,lr=.0003,threshold=.5,selection='fixed final checkpoint; no held-out tuning',BN='normal training',dropout='existing normal training',augmentation='none',source_manifest_sha256=sha(R/'data/manifest.json'),tf=tf.__version__,build=tf.sysconfig.get_build_info()),indent=2))
 for fold in m['folds']:
  fid=fold['fold'];seed=20261030+fid;fd=out/f'fold{fid}';fd.mkdir(exist_ok=False)
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);initial=build_unet_2d(1,16);weights=initial.get_weights();initial.save(fd/'initial.keras');weight_hash=wd(weights);del initial
  for arm,key in [('control','control_train_files'),('expanded','expanded_train_files')]:
   train=fold[key];test=fold['test_files'];assert not set(train)&set(test)
   assert all(records[n]['cell']!=fold['held_out_specimen'] for n in train)
   assert all(records[n]['cell']==fold['held_out_specimen'] for n in test)
   cells=sorted(fold['train_specimens']);groups={c:[n for n in train if records[n]['cell']==c] for c in cells};assert len(groups)==4 and all(groups.values())
   tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);model=build_unet_2d(1,16);model.set_weights(weights);assert wd(model.get_weights())==weight_hash
   opt=tf.keras.optimizers.Adam(.0003);rng=np.random.default_rng(seed);dest=fd/arm;dest.mkdir();start=time.time();history=[]
   @tf.function
   def step(x,y):
    with tf.GradientTape() as tape:
     pr=model(x,training=True);loss=masked_bce_dice(y,pr)
    opt.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return loss
   for n in range(1,1001):
    b=[pairs[groups[c][rng.integers(len(groups[c]))]] for c in cells]
    loss=float(step(np.stack([a[0] for a in b]),np.stack([a[1] for a in b])));assert np.isfinite(loss)
    if n%100==0:history.append(dict(step=n,loss=loss,seconds=time.time()-start));print(fid,arm,history[-1],flush=True)
    if time.time()-start>600:raise TimeoutError('10minute per-arm guard')
   model.save(dest/'candidate.keras');rows=[]
   # Held-out outputs are evaluated only after the fixed training has finished.
   # Identical old training anchors provide comparable training-fit diagnostics.
   for split,names in [('train_anchor',fold['control_train_files']),('test',test)]:
    for name in names:
     x,y=pairs[name];pr=model(x[None],training=False).numpy()[0,...,0];assert np.isfinite(pr).all();t=y[...,0]>0;p=pr>=.5;tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
     rows.append(dict(file=name,cell=records[name]['cell'],split=split,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn,foreground_fraction=float(t.mean()),predicted_fraction=float(p.mean())))
     if split=='test':np.savez_compressed(dest/name,input=x[...,0],label=y[...,0].astype('uint8'),probability=pr)
   (dest/'metrics.json').write_text(json.dumps(rows,indent=2));(dest/'history.json').write_text(json.dumps(history,indent=2));(dest/'checkpoint.sha256').write_text(sha(dest/'candidate.keras'))
   (dest/'lock.json').write_text(json.dumps(dict(fold=fid,arm=arm,seed=seed,initial_weights_sha256=weight_hash,training_files=train,test_files=test,steps=1000,production_approved=False),indent=2))
  (fd/'complete.json').write_text(json.dumps(dict(status='COMPLETE',held_out_specimen=fold['held_out_specimen'],initial_weights_sha256=weight_hash),indent=2));print('FOLD_COMPLETE',fid,flush=True)
 (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',folds=5,arms=2,production_approved=False),indent=2));print('DIVERSE_CV_COMPLETE',flush=True)
if __name__=='__main__':main()
