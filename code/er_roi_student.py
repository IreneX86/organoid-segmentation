"""Exploratory Fluo-only student for fixed, partially labelled 320px ROIs.

GPU training only. ER intensities are never packaged or passed to the model.
"""
from pathlib import Path
import argparse, json, hashlib, time, csv
import numpy as np
import tifffile as tif

CONFIG=dict(seed=20261029,patch_size=320,base_filters=16,batch_size=4,max_epochs=40,
            patience=8,learning_rate=0.0003,threshold=0.5,
            loss='masked_bce_dice',model='existing build_unet_2d; unchanged BN/dropout',
            normalization='per-input-patch Fluo p1-p99.8; identical at inference',
            augmentation='paired rot90 and horizontal flip',
            selection='minimum validation masked loss',
            labels='0 provisional negative, 1 pseudo-positive, 255 ignore',
            evaluation='known-pixel pseudo-label agreement only; no independent test')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare(source,out):
 out.mkdir(exist_ok=False,parents=True); records=[]
 m=json.loads((source/'manifest.json').read_text())
 for r in m['records']:
  d=source/r['directory']; ip=d/'Fluo_input.tif';lp=d/'partial_label_0neg_1pos_255ignore.tif'
  assert sha(ip)==r['hashes']['Fluo_input'] and sha(lp)==r['hashes']['partial_label_0neg_1pos_255ignore']
  x=tif.imread(ip);y=tif.imread(lp)
  assert x.shape==y.shape==(320,320) and np.isin(y,[0,1,255]).all()
  assert np.any(y==0) and np.any(y==1)
  lo,hi=np.percentile(x,[1,99.8]);assert hi>lo
  filename=r['directory']+'.npz'
  np.savez_compressed(out/filename,fluo=x,label=y,low=lo,high=hi)
  records.append(dict(file=filename,sha256=sha(out/filename),cell=r['cell'],split=r['split'],
                      z=r['z_zero_based'],y=r['y0'],x=r['x0'],source_hashes=[sha(ip),sha(lp)]))
 write(out/'manifest.json',dict(config=CONFIG,source_manifest_sha256=sha(source/'manifest.json'),records=records))
 return verify(out)
def load(root,r):
 with np.load(root/r['file'],allow_pickle=False) as a:
  x=np.clip((a['fluo'].astype('float32')-float(a['low']))/(float(a['high'])-float(a['low'])),0,1)
  return x[...,None],a['label'][...,None].astype('float32')
def verify(root):
 m=json.loads((root/'manifest.json').read_text());assert m['config']==CONFIG
 groups={s:{r['cell'] for r in m['records'] if r['split']==s} for s in ['train','validation']}
 assert len(groups['train'])==4 and len(groups['validation'])==1 and not groups['train']&groups['validation']
 for r in m['records']:
  assert sha(root/r['file'])==r['sha256'];x,y=load(root,r)
  assert x.shape==y.shape==(320,320,1) and np.isfinite(x).all() and np.isin(y,[0,1,255]).all()
  assert np.any(y==0) and np.any(y==1)
 return dict(status='PASS',counts={s:sum(r['split']==s for r in m['records']) for s in groups},er_channel_packaged=False)
def train(root,out):
 import tensorflow as tf
 from model.unet import build_unet_2d
 from er_partial_supervision import masked_bce_dice
 verify(root);gpu=tf.config.list_physical_devices('GPU');assert len(gpu)==1,'Exactly one GPU required; CPU training prohibited'
 tf.config.experimental.set_memory_growth(gpu[0],True)
 tf.config.set_soft_device_placement(False)
 with tf.device('/GPU:0'):
  a=tf.ones((32,32));b=tf.matmul(a,a);assert 'GPU' in b.device
 tf.config.set_soft_device_placement(True);tf.keras.utils.set_random_seed(CONFIG['seed'])
 out.mkdir(parents=True,exist_ok=False);start=time.time()
 m=json.loads((root/'manifest.json').read_text());write(out/'config.json',CONFIG);write(out/'data_manifest.json',m)
 write(out/'environment.json',dict(tensorflow=tf.__version__,numpy=np.__version__,build=tf.sysconfig.get_build_info(),gpu=str(gpu)))
 rows={s:[r for r in m['records'] if r['split']==s] for s in ['train','validation']}
 data={s:[load(root,r) for r in rows[s]] for s in rows}
 model=build_unet_2d(1,CONFIG['base_filters']);opt=tf.keras.optimizers.Adam(CONFIG['learning_rate'])
 # Explicit loop avoids asynchronous loader duplication and covers each crop once/epoch.
 @tf.function
 def step(x,y):
  with tf.GradientTape() as tape:
   p=model(x,training=True);loss=masked_bce_dice(y,p)
  grads=tape.gradient(loss,model.trainable_variables)
  opt.apply_gradients(zip(grads,model.trainable_variables));return loss
 rng=np.random.default_rng(CONFIG['seed']);best=float('inf');bad=0;best_epoch=0
 ck=out/'er_roi_student.keras'
 with open(out/'training_history.csv','w') as f:
  writer=csv.DictWriter(f,fieldnames=['epoch','loss','val_loss','elapsed_seconds']);writer.writeheader()
  for epoch in range(1,CONFIG['max_epochs']+1):
   losses=[];order=rng.permutation(len(data['train']))
   for offset in range(0,len(order),4):
    xx=[];yy=[]
    for idx in order[offset:offset+4]:
     x,y=data['train'][idx];k=int(rng.integers(4));x=np.rot90(x,k);y=np.rot90(y,k)
     if rng.random()<.5:x=np.flip(x,1);y=np.flip(y,1)
     xx.append(x);yy.append(y)
    losses.append(float(step(np.stack(xx),np.stack(yy))))
   vl=float(np.mean([float(masked_bce_dice(y[None],model(x[None],training=False))) for x,y in data['validation']]))
   row=dict(epoch=epoch,loss=float(np.mean(losses)),val_loss=vl,elapsed_seconds=time.time()-start)
   assert np.isfinite(list(row.values())).all(),'Nonfinite training';writer.writerow(row);f.flush();print(json.dumps(row),flush=True)
   if vl<best:best=vl;best_epoch=epoch;bad=0;model.save(ck)
   else:bad+=1
   if bad>=CONFIG['patience']:break
 model=tf.keras.models.load_model(ck,compile=False);metrics=[]
 for split in rows:
  for r,(x,y) in zip(rows[split],data[split]):
   p=model(x[None],training=False).numpy()[0,...,0];y=y[...,0];assert np.isfinite(p).all()
   pred=p>=.5;valid=y!=255;truth=y==1
   tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
   metrics.append(dict(file=r['file'],split=split,tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),unknown_positive_fraction=float(pred[~valid].mean()) if (~valid).any() else None))
   d=out/'predictions'/split;d.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(d/r['file'],fluo_normalized=x[...,0],label=y.astype('uint8'),probability=p)
 write(out/'metrics.json',dict(note=CONFIG['evaluation'],rows=metrics))
 write(out/'result.json',dict(epochs=epoch,best_epoch=best_epoch,early_stopping=bad>=CONFIG['patience'],elapsed_seconds=time.time()-start,checkpoint_sha256=sha(ck),production_approved=False))
 print('ER_ROI_STUDENT_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','verify','train']);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
 if a.mode=='prepare':print(prepare(a.data,a.output))
 elif a.mode=='verify':print(verify(a.data))
 else:train(a.data,a.output)
