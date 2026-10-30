"""ER-channel nERdy + existing ER line seeds; reviewed-ROI label candidates only.
No Fluo-derived model masks are used to make targets. No student training.
"""
import sys,json,shutil
from pathlib import Path
import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from skimage.filters import sato
from PIL import Image,ImageDraw
from er_nerdy_reference_pilot import normalize,sha
R=Path(__file__).resolve().parents[1];SRC=R/'outputs/er_positive_paired_crops_20261029';OUT=R/'outputs/er_combined_label_preview_20261029';CLONE=Path('/private/tmp/er-nerdy-reference-20261029')
SELECT=['cell_20260814_23298_NZ_50_z025_y272_x272','cell_20260814_23299_NZ_200_z100_y272_x816','cell_20260904_23961_NZ_100_z050_y816_x544','cell_20260904_23962_NZ_170_z085_y272_x272','cell_20260904_23965_NZ_140_z070_y272_x272']
def main():
 OUT.mkdir(exist_ok=False);np.int=int;np.float=float;np.complex=complex
 sys.path.insert(0,str(CLONE/'nERdy+'));import torch
 from model import D4nERdy
 from postprocessing import postprocessing
 torch.set_num_threads(4);weights=CLONE/'nERdy+'/ 'NNet_groupy_p4m_v2_VecAdam.pth';model=D4nERdy(1,1);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
 index={r['directory']:r for r in json.load(open(SRC/'manifest.json'))['records']};rows=[];sheet=Image.new('RGB',(384*6,414*5),'black')
 for i,name in enumerate(SELECT):
  row=index[name];assert row['split']=='train';src=SRC/name;dst=OUT/name;dst.mkdir()
  for stem,expected in row['crop_sha256'].items():assert sha(src/(stem+'.tif'))==expected
  er=tf.imread(src/'ER_reference.tif');fluo=tf.imread(src/'Fluo_input.tif');old=tf.imread(src/'observed_ER_positive_seed_NOT_COMPLETE_LABEL.tif')>0
  # Run the fixed public network on ER, not on the student's Fluo input.
  with torch.inference_mode():prob=torch.sigmoid(model(torch.from_numpy(normalize(er)[None,None]))).squeeze().numpy()
  score=(prob-prob.min())/max(float(np.ptp(prob)),1e-8);external=postprocessing(score)>0
  a=normalize(er);sm=ndi.gaussian_filter(a,.7);contrast=sm-ndi.gaussian_filter(sm,6);res=a-sm;noise=max(float(1.4826*np.median(np.abs(res-np.median(res)))),.001)
  ridge=sato(sm,sigmas=[.8,1.5,2.5],black_ridges=False);rth=float(np.percentile(ridge[ridge>0],65))
  support=(contrast>0.5*noise)&(ridge>rth*.5)
  strong=(contrast>1.5*noise)&(ridge>rth)
  supported_old=old&ndi.maximum_filter(support,size=3)
  additions=external&strong&~old&(ndi.distance_transform_edt(~old)<=12)
  positive=supported_old|additions
  # All unmarked intracellular tissue stays unknown. Even these low-intensity
  # negatives are provisional and must not be called manually verified background.
  negative=(ndi.maximum_filter(sm,size=11)<np.percentile(sm,25))&(ndi.distance_transform_edt(old|external|support)>12)
  negative=ndi.binary_erosion(negative,iterations=2)&~positive
  partial=np.full(old.shape,255,dtype='uint8');partial[negative]=0;partial[positive]=1
  files={'ER_reference':er,'Fluo_input':fluo,'old_ER_mask_line':old.astype('uint8')*255,'nERdy_ER_input_candidate':external.astype('uint8')*255,'combined_positive_candidate':positive.astype('uint8')*255,'partial_label_0neg_1pos_255ignore':partial,'loss_valid_candidate':((partial!=255)*255).astype('uint8'),'added_positive':additions.astype('uint8')*255,'old_now_unknown':(old&~positive).astype('uint8')*255}
  hashes={}
  for stem,arr in files.items():tf.imwrite(dst/(stem+'.tif'),arr);hashes[stem]=sha(dst/(stem+'.tif'))
  def overlay(raw,m):
   im=np.repeat((normalize(raw)*255).astype('uint8')[...,None],3,2);im[m]=(im[m]*.35+np.array([255,220,0])*.65).astype('uint8');return im
  validity=np.full((*old.shape,3),65,dtype='uint8');validity[positive]=[255,220,0];validity[negative]=[0,130,255]
  panels=[overlay(er,np.zeros_like(old)),overlay(fluo,np.zeros_like(old)),overlay(er,old),overlay(er,external),overlay(er,positive),validity]
  titles=['ER reference','Fluo input','Old ER mask','nERdy on ER','Combined candidate','Yellow + / blue weak - / gray ignore']
  local=Image.new('RGB',(384*6,414),'black')
  for j,(panel,title) in enumerate(zip(panels,titles)):
   local.paste(Image.fromarray(panel),(384*j,30));ImageDraw.Draw(local).text((384*j+3,5),title,fill='white')
  local.save(dst/'review.png');sheet.paste(local,(0,i*414));Image.fromarray(overlay(fluo,positive)).save(dst/'Fluo_with_combined_positive.png')
  r=dict(**row,old_pixels=int(old.sum()),external_pixels=int(external.sum()),positive_pixels=int(positive.sum()),added_pixels=int(additions.sum()),old_to_unknown_pixels=int((old&~positive).sum()),weak_negative_pixels=int(negative.sum()),unknown_pixels=int((partial==255).sum()),output_hashes=hashes,noise_proxy=noise,ridge_threshold=rth);rows.append(r);print(name,'positive',r['positive_pixels'],'add',r['added_pixels'],'negative',r['weak_negative_pixels'],flush=True)
 sheet.save(OUT/'five_combined_labels_review.png');(OUT/'manifest.json').write_text(json.dumps(dict(training_ready=False,status='ROI review candidates, not complete manual truth',model_input='ER only',future_student_input='Fluo only',external_weights_sha256=sha(weights),rules='old seeds require nearby ER contrast/ridge support; nERdy additions require stronger ER contrast/ridge support and <=12px from old; conservative low intensity negatives provisional; remaining pixels ignored',records=rows),indent=2))
if __name__=='__main__':main()
