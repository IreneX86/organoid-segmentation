"""CAREamics official 2D N2V, whole-specimen split, no ER labels used.
Outputs are Fluo structure candidates, not validated organelle identity.
"""
from pathlib import Path
import json, hashlib, time, sys, csv, os
import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from skimage.restoration import estimate_sigma, denoise_nl_means
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import remove_small_objects, skeletonize
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/er_n2v_20261029'
TARGET='cell_20260904_23966_NZ_100'; VAL='cell_20260904_23963_NZ_40'; UNUSED='cell_20260814_23300_NZ_100'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p,z):return tf.imread(p,key=int(z)).astype(np.float32)
def corr(x,y):
 return float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def audit():
 records=[]; train=[]; val=[]; samplehashes={}; rng=np.random.default_rng(20261029)
 for p in sorted((ROOT/'outputs/er_coherent_fiber_full_20261028').glob('*/*/manifest.json')):
  m=json.loads(p.read_text()); cell=p.parent.name; path=m['source_fluo']; shape=m['source_shape']; nz=shape[0]
  split='test_target' if cell==TARGET else 'validation' if cell==VAL else 'unused_old_test' if cell==UNUSED else 'train'
  with tf.TiffFile(path) as t: meta=dict(shape=list(t.series[0].shape),axes=t.series[0].axes,dtype=str(t.series[0].dtype),pages=len(t.pages),description=str(t.pages[0].description)[:1500],resolution=str(t.pages[0].tags.get('XResolution')))
  stats=[]
  for z in np.unique(np.round(np.linspace(0,nz-1,5)).astype(int)):
   a=read(path,z); sm=ndi.gaussian_filter(a,1); r=a-sm; grad=np.hypot(*np.gradient(sm)); quiet=grad<np.percentile(grad,20)
   sigma=float(estimate_sigma(a,channel_axis=None)); denom=max(sigma,1e-6)
   stats.append(dict(z_zero_based=int(z),percentiles=np.percentile(a,[0,1,10,50,90,99,99.8,100]).tolist(),wavelet_noise_sigma_proxy=sigma,contrast_to_noise_proxy=float((np.percentile(a,95)-np.percentile(a,10))/denom),saturated_fraction=float(np.mean(a>=65535)),quiet_residual_lag_x=corr(r[:,:-1][quiet[:,:-1]],r[:,1:][quiet[:,:-1]]),quiet_residual_lag_y=corr(r[:-1][quiet[:-1]],r[1:][quiet[:-1]]),adjacent_z_correlation=corr(a,read(path,min(int(z)+1,nz-1)))))
  actualsha=sha(path)
  if actualsha!=m['source_fluo_sha256']:raise ValueError('source changed '+cell)
  records.append(dict(cell=cell,source=path,sha256=actualsha,split=split,metadata=meta,stats=stats))
  if split in ('train','validation'):
   # Identical number of widely separated planes per specimen. Split is by whole specimen.
   for z in np.round(np.linspace(0,nz-1,8)).astype(int):
    a=read(path,z)
    # Six non-overlapping 128px fields, deterministic spatial coverage; no cross-split crops.
    for y,x in [(128,128),(128,536),(128,944),(800,128),(800,536),(800,944)]:
     c=a[y:y+128,x:x+128].copy(); key=hashlib.sha256(c.tobytes()).hexdigest()
     if key in samplehashes and samplehashes[key]!=split:raise ValueError('cross split duplicate')
     samplehashes[key]=split
     (train if split=='train' else val).append(c)
 save('audit.json',dict(records=records,noise_caveat='Single acquisition: sigma/contrast and high-pass residual correlations are proxies, not true SNR or a proven noise model. Correlation mixes noise and biological texture; no Poisson/Gaussian certainty.',no_manual_ER_ground_truth=True))
 np.save(OUT/'train.npy',np.stack(train));np.save(OUT/'validation.npy',np.stack(val))
 print('AUDIT_COMPLETE',len(train),len(val),flush=True)
 return records

def train():
 import torch
 from careamics import CAREamist
 from careamics.config.factories import create_advanced_n2v_config
 from lightning.pytorch.callbacks import EarlyStopping,Callback
 torch.set_num_threads(4)
 class History(Callback):
  def on_validation_epoch_end(self,trainer,module):
   if trainer.sanity_checking:return
   row={'epoch':int(trainer.current_epoch+1),'metrics':{k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if hasattr(v,'numel') and v.numel()==1}}
   with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print('EPOCH',json.dumps(row),flush=True)
 config=create_advanced_n2v_config(experiment_name='er_fluo_n2v_2d',data_type='array',axes='SYX',patch_size=[64,64],batch_size=16,num_epochs=30,num_steps=30,n_val_patches=8,num_workers=0,seed=20261029,model_params={'depth':2,'num_channels_init':16},trainer_params={'accelerator':'cpu','devices':1,'enable_progress_bar':False,'enable_model_summary':False,'num_sanity_val_steps':0,'limit_val_batches':6},checkpoint_params={'save_top_k':1,'save_last':True,'monitor':'val_loss','mode':'min'},optimizer_params={'lr':0.0004},normalization='mean_std',use_n2v2=False)
 config.training_config.early_stopping_params={'monitor':'val_loss','patience':7,'mode':'min'}
 save('config.json',config.model_dump(mode='json'))
 engine=CAREamist(config,work_dir=OUT/'training',callbacks=[History()],enable_progress_bar=False)
 start=time.monotonic();engine.train(train_data=np.load(OUT/'train.npy'),val_data=np.load(OUT/'validation.npy'))
 best=engine.trainer.checkpoint_callback.best_model_path
 save('training_result.json',dict(checkpoint=best,checkpoint_sha256=sha(best),seconds=time.monotonic()-start,epochs=int(engine.trainer.current_epoch),best_validation_loss=float(engine.trainer.checkpoint_callback.best_model_score),device='cpu',official_implementation='careamics==0.3.5'))
 print('TRAINING_COMPLETE',best,flush=True)
 return best

def segment(a,sigma):
 # Fixed methods for raw/NLM/N2V; no manual masks or test-label-dependent tuning.
 global_mask=a>threshold_otsu(a)
 local_mask=a>threshold_local(a,51,method='gaussian',offset=-1.5*sigma)
 return [remove_small_objects(m,min_size=16) for m in (global_mask,local_mask)]

def predict(best):
 import torch
 from careamics import CAREamist
 torch.set_num_threads(4)
 records=json.loads((OUT/'audit.json').read_text())['records']; target=next(r for r in records if r['cell']==TARGET)
 engine=CAREamist(checkpoint_path=best,work_dir=OUT/'inference',enable_progress_bar=False)
 nz=target['metadata']['shape'][0]; chosen=[0,24,49,74,99]; review=Image.new('RGB',(1500,270*5),'black'); metrics=[]
 writers={}
 try:
  for name in ['N2V_denoised','NLM_denoised']+[f'{m}_{t}_mask' for m in ('raw','NLM','N2V') for t in ('otsu','local')]:writers[name]=tf.TiffWriter(OUT/(name+'.tif'),bigtiff=False)
  for z in range(nz):
   a=read(target['source'],z); sigma=float(estimate_sigma(a,channel_axis=None))
   baseline=denoise_nl_means(a,h=0.8*sigma,sigma=sigma,fast_mode=True,patch_size=5,patch_distance=6,preserve_range=True,channel_axis=None).astype('float32')
   preds,_=engine.predict(a,axes='YX',data_type='array',tile_size=(512,512),tile_overlap=(64,64),batch_size=4,num_workers=0)
   den=np.asarray(preds[0]).squeeze().astype('float32');assert den.shape==a.shape and np.isfinite(den).all()
   writers['N2V_denoised'].write(den,compression='zlib',contiguous=False,metadata=None)
   writers['NLM_denoised'].write(baseline,compression='zlib',contiguous=False,metadata=None)
   maskmap={}
   for method,im in [('raw',a),('NLM',baseline),('N2V',den)]:
    masks=segment(im,sigma)
    for name,mask in zip(('otsu','local'),masks):
     writers[f'{method}_{name}_mask'].write(mask.astype('uint8')*255,compression='zlib',metadata=None)
     maskmap[f'{method}_{name}']=mask
   if z in chosen:
    i=chosen.index(z);folder=OUT/f'preview_z{z+1:03}';folder.mkdir(exist_ok=True);tf.imwrite(folder/'raw_fluo.tif',a.astype('uint16'))
    lo,hi=np.percentile(a,[1,99.8])
    def display(im):return np.repeat((np.clip((im-lo)/max(hi-lo,1),0,1)*255).astype('uint8')[...,None],3,2)
    overlay=display(a);mask=maskmap['N2V_local'];overlay[mask]=(0.4*overlay[mask]+0.6*np.array([255,220,0])).astype('uint8')
    panels=[display(a),display(baseline),display(den),np.repeat(maskmap['N2V_otsu'][...,None]*255,3,2).astype('uint8'),np.repeat(mask[...,None]*255,3,2).astype('uint8'),overlay]
    full=Image.new('RGB',(1200*6,1230),'black')
    for j,(panel,title) in enumerate(zip(panels,['Raw','NLM','N2V','N2V Otsu','N2V local','N2V local overlay'])):
     full.paste(Image.fromarray(panel),(j*1200,30));ImageDraw.Draw(full).text((j*1200+4,5),title,fill='white')
     review.paste(Image.fromarray(panel).resize((250,250)),(j*250,i*270+20))
    full.save(folder/'comparison.png');ImageDraw.Draw(review).text((4,i*270+3),f'Z {z+1} | Raw / NLM / N2V / Otsu / Local / Overlay',fill='white')
    review.save(OUT/'five_slice_comparison.png')
    r=a-den; row={'z_one_based':z+1,'noise_sigma_proxy_raw':sigma,'noise_sigma_proxy_NLM':float(estimate_sigma(baseline,channel_axis=None)),'noise_sigma_proxy_N2V':float(estimate_sigma(den,channel_axis=None)),'raw_N2V_pearson_not_accuracy':corr(a,den),'residual_adjacent_x':corr(r[:,:-1],r[:,1:]),'residual_rms':float(np.sqrt(np.mean(r*r)))}
    for name,mask in maskmap.items():
     sk=skeletonize(mask);row[name]={'foreground_fraction':float(mask.mean()),'connected_components':int(ndi.label(mask)[1]),'skeleton_pixels':int(sk.sum())}
    metrics.append(row);save('diagnostics.json',metrics)
   if (z+1)%10==0:print('PREDICTED',z+1,nz,flush=True)
 finally:
  for w in writers.values():w.close()
 verified=[]
 for name in writers:
  p=OUT/(name+'.tif')
  # Pack streamed pages into an explicit ZYX series for Fiji/napari interoperability.
  packed=p.with_name(p.stem+'_packed.tif')
  with tf.TiffFile(p) as t:
   tf.imwrite(packed,(page.asarray() for page in t.pages),shape=(nz,1200,1200),dtype=t.pages[0].dtype,compression='zlib',metadata={'axes':'ZYX'})
  packed.replace(p)
  with tf.TiffFile(p) as t:
   assert t.series[0].shape==(nz,1200,1200) and t.series[0].axes=='ZYX'
   assert len(t.pages)==nz
   for page in t.pages:
    a=page.asarray();assert a.shape==(1200,1200) and np.isfinite(a).all()
    if 'mask' in name:assert a.dtype==np.uint8 and set(np.unique(a))<={0,255}
  verified.append({'file':str(p),'pages':nz,'sha256':sha(p)})
 assert sha(target['source'])==target['sha256']
 save('verification.json',{'status':'PASS','outputs':verified,'source_unchanged':True})
 print('ALL_COMPLETE',flush=True)
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 stage=sys.argv[1] if len(sys.argv)>1 else 'all'
 if stage in ('all','audit'):audit()
 if stage in ('all','train'):best=train()
 if stage in ('all','predict'):predict(best if stage=='all' else json.loads((OUT/'training_result.json').read_text())['checkpoint'])
