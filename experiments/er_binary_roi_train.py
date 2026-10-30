"""User-defined binary ROI pilot: yellow ER; blue/gray background inside320px crops."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import tensorflow as tf
 from er_roi_student import load
 from er_partial_supervision import masked_bce_dice
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 g=tf.config.list_physical_devices('GPU');assert len(g)==1;tf.config.experimental.set_memory_growth(g[0],True)
 with tf.device('/GPU:0'):
  v=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in v.device
 data=R/'data';m=json.loads((data/'manifest.json').read_text());tr=[r for r in m['records'] if r['split']=='train'];va=[r for r in m['records'] if r['split']=='validation'];cells=sorted(set(r['cell'] for r in tr));assert len(tr)==32 and len(va)==8 and not set(cells)&set(r['cell'] for r in va)
 def read(r):
  assert sha(data/r['file'])==r['sha256'];x,y=load(data,r);assert set(np.unique(y))<={0,1};return x,y
 groups={c:[read(r) for r in tr if r['cell']==c] for c in cells};validation=[(r,read(r)) for r in va]
 checkpoint=R/'source.keras';h=sha(checkpoint);tf.keras.utils.set_random_seed(20261029);model=tf.keras.models.load_model(checkpoint,compile=False)
 for l in model.layers:
  if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
 def evaluate(name,records):
  out=a.output/name;out.mkdir();rows=[]
  for r,(x,y) in records:
   p=model(x[None],training=False).numpy()[0,...,0];yy=y[...,0];assert np.isfinite(p).all();pred=p>=.5;t=yy==1;tp=int((pred&t).sum());fp=int((pred&~t).sum());fn=int((~pred&t).sum())
   rows.append(dict(file=r['file'],tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),positive_fraction=float(pred.mean())))
   np.savez_compressed(out/r['file'],fluo=x[...,0],label=yy.astype('uint8'),probability=p)
  (out/'metrics.json').write_text(json.dumps(rows,indent=2));return rows
 before=evaluate('before',validation)
 conf=dict(seed=20261029,updates=500,batch_size=4,lr=.0003,threshold=.5,selection='fixedfinal500,noearlystopping/validationselection',BN='frozenstatsandaffine',dropout='off',augmentation='none',labels='yellow=1;grayandblue=0 inselected320pxcrops only',source_sha256=h,production_approved=False)
 (a.output/'protocol.json').write_text(json.dumps(conf,indent=2));(a.output/'environment.json').write_text(json.dumps(dict(tf=tf.__version__,build=tf.sysconfig.get_build_info()),indent=2))
 opt=tf.keras.optimizers.Adam(.0003);rng=np.random.default_rng(20261029)
 @tf.function
 def step(x,y):
  with tf.GradientTape() as tape:p=model(x,training=False);loss=masked_bce_dice(y,p)
  opt.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return loss
 start=time.time();hist=[]
 for n in range(1,501):
  b=[groups[c][rng.integers(len(groups[c]))] for c in cells];loss=float(step(np.stack([v[0] for v in b]),np.stack([v[1] for v in b])));assert np.isfinite(loss)
  if n%50==0:hist.append(dict(step=n,loss=loss,seconds=time.time()-start));print(hist[-1],flush=True)
  if time.time()-start>600:raise TimeoutError('10min training guard')
 model.save(a.output/'candidate.keras');after=evaluate('after',validation);trainmetrics=evaluate('training',[(r,read(r)) for r in tr]);assert sha(checkpoint)==h
 (a.output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',training_seconds=hist[-1]['seconds'],elapsed_seconds=time.time()-start,history=hist,checkpoint_sha256=sha(a.output/'candidate.keras'),production_approved=False),indent=2));print('ER_BINARY_ROI_COMPLETE',flush=True)
if __name__=='__main__':main()
