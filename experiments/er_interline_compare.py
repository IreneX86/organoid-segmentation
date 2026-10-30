"""Compare frozen train-calibrated models on identical original validation labels."""
from pathlib import Path
import json,hashlib
import numpy as np
import tifffile as tif
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];OLD=R/'outputs/er_roi_bn_recalibration_20261029';NEW=R/'outputs/er_interline_bn_calibrated_20261029';DATA=R/'outputs/er_roi_student_ready_20261029'
def metrics(y,p):
 v=y!=255;t=y==1;b=p>=.5;tp=int((v&t&b).sum());fp=int((v&~t&b).sum());fn=int((v&t&~b).sum())
 return dict(tp=tp,fp=fp,fn=fn,dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),area=float(b.mean()),unknown_area=float(b[~v].mean()))
def main():
 rows=[r for r in json.loads((DATA/'manifest.json').read_text())['records'] if r['split']=='validation'];records=[];views=[];ya=[];pa=[];pb=[]
 for r in rows:
  name=r['file'][:-4]
  with np.load(DATA/r['file']) as z:x=np.clip((z['fluo'].astype('float32')-z['low'])/(z['high']-z['low']),0,1);y=z['label']
  a=tif.imread(OLD/name/'probability.tif');b=tif.imread(NEW/name/'probability.tif');ma=metrics(y,a);mb=metrics(y,b)
  ya.append(y);pa.append(a);pb.append(b);records.append(dict(file=name,old=ma,new=mb,delta_dice=mb['dice']-ma['dice']))
  gray=np.repeat((x*255).astype('uint8')[...,None],3,2);ov=gray.copy();ov[b>=.5]=[255,220,0]
  labels=np.full((320,320,3),65,np.uint8);labels[y==1]=[255,220,0];labels[y==0]=[0,130,255]
  arrays=[gray,labels,np.repeat((a>=.5)[...,None].astype('uint8')*255,3,2),np.repeat((b>=.5)[...,None].astype('uint8')*255,3,2),ov]
  v=Image.new('RGB',(1600,355));draw=ImageDraw.Draw(v)
  for j,(im,title) in enumerate(zip(arrays,['Fluo','Unchanged validation labels','Old + train-only BN','Interline + train-only BN','New overlay'])):
   v.paste(Image.fromarray(im),(j*320,35));draw.text((j*320+3,4),title,fill='white')
  v.save(NEW/name/'paired_review.png');views.append(v)
 sheet=Image.new('RGB',(1600,len(views)*355))
 for i,v in enumerate(views):sheet.paste(v,(0,i*355))
 sheet.save(NEW/'paired_all8.png');out=dict(old=metrics(np.stack(ya),np.stack(pa)),new=metrics(np.stack(ya),np.stack(pb)),improved_images=sum(r['delta_dice']>0 for r in records),rows=records,scope='One previously seen specimen; incomplete pseudo-label agreement; exploratory only')
 (NEW/'paired_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
