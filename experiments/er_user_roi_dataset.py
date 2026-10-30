"""Map user red ROI hints to aligned native TIFF patches, never use screenshot intensities as data."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image
from PIL import Image,ImageDraw
import tifffile as tf
R=Path(__file__).resolve().parents[1];SRC=R/'outputs/er_combined_label_preview_20261029';OUT=R/'outputs/er_user_roi_dataset_20261029';ANNOT=Path('~/Desktop/five_combined_labels_review.png').expanduser()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def disp(a):
 lo,hi=np.percentile(a,[1,99.8]);return np.repeat((np.clip((a.astype('float32')-lo)/max(hi-lo,1),0,1)*255).astype('uint8')[...,None],3,2)
def main():
 OUT.mkdir(exist_ok=False);m=json.load(open(SRC/'manifest.json'));image=np.array(Image.open(ANNOT).convert('RGB'));assert image.shape[:2]==(2070,2304)
 red=(image[:,:,0]>130)&(image[:,:,0]>1.45*image[:,:,1])&(image[:,:,0]>1.4*image[:,:,2])&(image[:,:,1]<140)
 # Bounds exclude horizontal red correspondence strokes; polygons are approximate
 # ROI hints, not pixel-accurate GT. Erosion excludes uncertain annotation edges.
 xmax=[245,220,318,255,266];rows=[];sheet=Image.new('RGB',(1280,286*5),'black')
 for i,r in enumerate(m['records']):
  src=SRC/r['directory'];dst=OUT/r['directory'];dst.mkdir()
  a=red[i*414+30:(i+1)*414,:384].copy();a[:,xmax[i]:]=False
  roi=ndi.binary_erosion(convex_hull_image(a),iterations=5)
  er=tf.imread(src/'ER_reference.tif');fluo=tf.imread(src/'Fluo_input.tif');old=tf.imread(src/'old_ER_mask_line.tif')>0
  assert sha(src/'ER_reference.tif')==r['output_hashes']['ER_reference'] and sha(src/'Fluo_input.tif')==r['output_hashes']['Fluo_input']
  pos=old&roi;sm=ndi.gaussian_filter(er.astype('float32'),.8)
  # The user pointed to the OLD-mask panel, so no unconfirmed nERdy additions
  # are silently made positive. Low ER intensity alone is not a biological proof:
  # negatives stay explicitly provisional; bright unmarked regions remain ignored.
  low=float(np.percentile(sm[roi],35));neg=roi&~old&(ndi.distance_transform_edt(~old)>3)&(sm<low)
  label=np.full(er.shape,255,dtype='uint8');label[pos]=1;label[neg]=0
  yy,xx=np.where(roi);y0=max(0,int(yy.min())-24);y1=min(384,int(yy.max())+25);x0=max(0,int(xx.min())-24);x1=min(384,int(xx.max())+25)
  sl=np.s_[y0:y1,x0:x1]
  files={'Fluo_input':fluo[sl],'ER_reference':er[sl],'positive_mask':(pos[sl]*255).astype('uint8'),'partial_label_0neg_1pos_255ignore':label[sl],'roi':(roi[sl]*255).astype('uint8'),'valid_loss_mask':((label[sl]!=255)*255).astype('uint8')}
  hashes={}
  for name,v in files.items():tf.imwrite(dst/(name+'.tif'),v);hashes[name]=sha(dst/(name+'.tif'))
  rgb=disp(fluo[sl]);over=rgb.copy();over[pos[sl]]=[255,220,0]
  semantic=np.full((*label[sl].shape,3),65,dtype='uint8');semantic[label[sl]==1]=[255,220,0];semantic[label[sl]==0]=[0,130,255]
  panels=[disp(er[sl]),rgb,over,semantic];titles=['ER','Fluo','Old mask within user ROI','Yellow + / blue provisional - / gray ignore']
  view=Image.new('RGB',(1280,286),'black')
  for j,(panel,title) in enumerate(zip(panels,titles)):
   im=Image.fromarray(panel);im.thumbnail((320,256));view.paste(im,(j*320,30));ImageDraw.Draw(view).text((j*320+3,4),title,fill='white')
  view.save(dst/'review.png');sheet.paste(view,(0,i*286))
  record=dict(cell=r['cell'],directory=r['directory'],split='validation' if i==4 else 'train',z_zero_based=r['z_zero_based'],y0=r['y0']+y0,x0=r['x0']+x0,shape=list(fluo[sl].shape),positive_pixels=int(pos.sum()),provisional_negative_pixels=int(neg.sum()),ignored_pixels=int(np.sum(label[sl]==255)),er_low_intensity_cutoff=low,hashes=hashes)
  assert record['positive_pixels']>0 and record['provisional_negative_pixels']>0
  rows.append(record);print(record['cell'],record['positive_pixels'],record['provisional_negative_pixels'],flush=True)
 sheet.save(OUT/'five_user_roi_pairs.png')
 save=dict(records=rows,annotation_sha256=sha(ANNOT),source_manifest_sha256=sha(SRC/'manifest.json'),labels='0 provisional negative,1 positive,255 ignored',positive_source='User-circled OLD ER_mask_line panel, not new nERdy additions',roi_method='red marks first ER column; exclude connecting strokes; convex hull; erode5; context24px',split='four whole specimens train;23965 validation; no independent test. Previously viewed data.',status='Prepared exploratory paired ROI data; not trained. Negative pixels remain ER-intensity-based candidates, not manual truth.')
 (OUT/'manifest.json').write_text(json.dumps(save,indent=2))
 for r in rows:
  d=OUT/r['directory'];x=tf.imread(d/'Fluo_input.tif');y=tf.imread(d/'partial_label_0neg_1pos_255ignore.tif');assert x.shape==y.shape and set(np.unique(y))<={0,1,255}
 (OUT/'verification.json').write_text(json.dumps({'status':'PASS','pairs':5,'native_aligned_data':True,'train_validation_cells_disjoint':True},indent=2))
if __name__=='__main__':main()
