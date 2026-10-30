"""Read-only review of completed N2V output; no ground-truth accuracy claims."""
import json
from pathlib import Path
import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];O=R/'outputs/er_n2v_20261029'
def corr(x,y):return float(np.corrcoef(x.ravel(),y.ravel())[0,1])
def main():
 assert json.loads((O/'verification.json').read_text())['status']=='PASS'
 ds=json.loads((O/'diagnostics.json').read_text()); out=[]
 zoom=Image.new('RGB',(1600,440*5),'black'); masksheet=Image.new('RGB',(1500,270*5),'black')
 for i,row in enumerate(ds):
  z=row['z_one_based']-1;raw=tf.imread(O/f'preview_z{z+1:03}'/'raw_fluo.tif').astype('float32');den=tf.imread(O/'N2V_denoised.tif',key=z);nlm=tf.imread(O/'NLM_denoised.tif',key=z)
  # Same fixed coordinates, no cherry-picking by model output.
  box=(slice(200,600),slice(100,500)); crop=raw[box];lo,hi=np.percentile(crop,[1,99.8]);res=raw-den
  residual_scale=float(np.percentile(np.abs(res[box]),99));res_rgb=np.zeros((400,400,3),dtype='uint8');res_rgb[...,0]=(np.clip(res[box]/max(residual_scale,1),0,1)*255).astype('uint8');res_rgb[...,2]=(np.clip(-res[box]/max(residual_scale,1),0,1)*255).astype('uint8')
  for j,(im,title) in enumerate([(raw,'Raw'),(nlm,'NLM'),(den,'N2V'),(None,'Residual: red removed / blue added')]):
   a=res_rgb if im is None else np.repeat((np.clip((im[box]-lo)/max(hi-lo,1),0,1)*255).astype('uint8')[...,None],3,2)
   zoom.paste(Image.fromarray(a),(400*j,440*i+40));ImageDraw.Draw(zoom).text((400*j+4,440*i+6),f'Z{z+1} {title}',fill='white')
  r=dict(z_one_based=z+1,residual_raw_correlation=corr(res,raw),mean_intensity_bias=float(np.mean(den-raw)),residual_rms=float(np.sqrt(np.mean(res**2))))
  j=0
  for method in ('raw','NLM','N2V'):
   for th in ('otsu','local'):
    m=tf.imread(O/f'{method}_{th}_mask.tif',key=z)>0
    masksheet.paste(Image.fromarray(m.astype('uint8')*255).convert('RGB').resize((250,250)),(250*j,270*i+20));ImageDraw.Draw(masksheet).text((250*j+3,270*i+3),f'Z{z+1} {method} {th}',fill='white');j+=1
  rawm=tf.imread(O/'raw_local_mask.tif',key=z)>0;denm=tf.imread(O/'N2V_local_mask.tif',key=z)>0
  rawsk=skeletonize(rawm);densk=skeletonize(denm)
  r['raw_skeleton_within_2px_of_N2V_fraction']=float(np.mean(ndi.distance_transform_edt(~densk)[rawsk]<=2))
  r['N2V_skeleton_farther_than_2px_from_raw_fraction']=float(np.mean(ndi.distance_transform_edt(~rawsk)[densk]>2))
  r['warning']='changes relative to noisy raw threshold, NOT ER false negative/positive rates'
  out.append(r)
 zoom.save(O/'five_detail_residuals.png');masksheet.save(O/'six_method_masks.png');(O/'structure_change_diagnostics.json').write_text(json.dumps(out,indent=2))
 h=[json.loads(l) for l in (O/'history.jsonl').read_text().splitlines()]
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['epoch'] for r in h],[r['metrics']['val_loss'] for r in h],marker='o',label='Held-out specimen blind-spot MSE');ax.axvline(12,color='orange',linestyle='--',label='Selected epoch 12');ax.set(xlabel='Epoch',ylabel='Normalized masked-pixel MSE',title='Noise2Void validation (not ER segmentation accuracy)');ax.legend();fig.tight_layout();fig.savefig(O/'validation_curve.png',dpi=150);plt.close(fig)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
