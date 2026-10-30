"""GPU-only paired inner selection; outer labels are never loaded by this runner."""
from pathlib import Path
import argparse,json,hashlib,time,sys
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def choose(best,score,delta):return best is None or score>best+delta
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import tensorflow as tf
 from model.unet import build_unet_2d
 from er_partial_supervision import masked_bce_dice
 m=json.loads(a.protocol.read_text());s=m['settings'];records={r['file']:r for r in m['records']}
 g=tf.config.list_physical_devices('GPU');assert len(g)==1;tf.config.experimental.set_memory_growth(g[0],True);tf.config.experimental.enable_op_determinism()
 with tf.device('/GPU:0'):
  v=tf.matmul(tf.ones((8,8)),tf.ones((8,8)));assert 'GPU' in v.device
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'run_lock.json').write_text(json.dumps({'protocol_sha256':sha(a.protocol),'tensorflow':tf.__version__,'build':tf.sysconfig.get_build_info(),'outer_evaluated':False},indent=2))
 for f in m['folds']:
  fid=f['fold'];seed=s['seed_base']+fid;fd=a.output/f'fold{fid}';fd.mkdir();data={}
  # Even though other folds need these files, this fold loads only train/inner data.
  for name in set(f['expanded_train']+f['inner_validation']):
   assert records[name]['cell']!=f['outer_specimen'];path=a.data/name;assert sha(path)==records[name]['sha256']
   with np.load(path) as z:
    x=z['fluo'].astype('float32');y=z['label'];assert x.shape==y.shape==(320,320) and np.isin(y,[0,1]).all()
    lo,hi=np.percentile(x,[1,99.8]);assert hi>lo;data[name]=(np.clip((x-lo)/(hi-lo),0,1)[...,None].astype('float32'),y[...,None].astype('float32'))
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);model=build_unet_2d(1,s['base_filters']);initial=model.get_weights();ih=hashlib.sha256(b''.join(x.tobytes() for x in initial)).hexdigest();del model
  for arm in ['control','expanded']:
   tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);model=build_unet_2d(1,s['base_filters']);model.set_weights(initial);opt=tf.keras.optimizers.Adam(s['learning_rate'])
   dest=fd/arm;dest.mkdir();rng=np.random.default_rng(seed);cells=sorted(f['training_specimens']);groups={c:[n for n in f[arm+'_train'] if records[n]['cell']==c] for c in cells};assert len(cells)==3 and all(groups.values())
   @tf.function
   def step(x,y):
    with tf.GradientTape() as tape:pr=model(x,training=True);loss=masked_bce_dice(y,pr)
    opt.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return loss
   best=None;best_step=None;stale=0;history=[];start=time.time()
   for n in range(1,s['maximum_steps']+1):
    selected=cells+[cells[(n-1)%3]];b=[data[groups[c][rng.integers(len(groups[c]))]] for c in selected]
    loss=float(step(np.stack([v[0] for v in b]),np.stack([v[1] for v in b])));assert np.isfinite(loss)
    if n%s['evaluate_every']:continue
    scores=[]
    for name in f['inner_validation']:
     x,y=data[name];pr=model(x[None],training=False).numpy()[0,...,0];assert np.isfinite(pr).all();pred=pr>=s['threshold'];t=y[...,0]>0
     scores.append(float(2*(pred&t).sum()/max(int(pred.sum()+t.sum()),1)))
    score=float(np.mean(scores));history.append(dict(step=n,loss=loss,inner_dice=score,seconds=time.time()-start));(dest/'history.json').write_text(json.dumps(history,indent=2));print(fid,arm,history[-1],flush=True)
    if n==s['minimum_steps']:model.save(dest/'step1000.keras')
    if n<s['minimum_steps']:continue
    # A single preregistered minimum improvement governs both selection and patience.
    if choose(best,score,s['min_delta']):
     best=score;best_step=n;stale=0;model.save(dest/'candidate.keras')
    else:stale+=1
    if stale>=s['patience_evaluations']:break
   decision=dict(fold=fid,arm=arm,initial_sha256=ih,selected_step=best_step,inner_dice=best,completed_step=n,stopped_by_patience=stale>=s['patience_evaluations'],checkpoint_sha256=sha(dest/'candidate.keras'),outer_evaluated=False,production_approved=False)
   (dest/'decision.json').write_text(json.dumps(decision,indent=2))
 (a.output/'selection_complete.json').write_text(json.dumps({'folds':5,'models':10,'outer_evaluated':False,'protocol_sha256':sha(a.protocol)},indent=2))
if __name__=='__main__':main()
