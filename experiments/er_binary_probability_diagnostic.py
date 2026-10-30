"""Read-only probability diagnosis; fixed illustrative thresholds, no model selection."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
from PIL import Image, ImageDraw
R=Path(__file__).resolve().parents[1]
SRC=R/'outputs/er_binary_roi_20261030/extracted/results/comparison'
OUT=R/'outputs/er_binary_probability_20261030'
def main():
 OUT.mkdir(exist_ok=False)
 rows=[]; panel=Image.new('RGB',(1920,2800));d=ImageDraw.Draw(panel)
 for i,f in enumerate(sorted((SRC/'after').glob('*.npz'))):
  with np.load(f) as a:x=a['fluo'];y=a['label'].astype(bool);p=a['probability']
  pos=p[y];neg=p[~y];fn=p[y&(p<.5)]
  # Rank AUC includes exact ties, avoids interpreting low prevalence ROC alone.
  ranks=rankdata(p.ravel());n=int(y.sum());nn=int((~y).sum());auc=(ranks[y.ravel()].sum()-n*(n+1)/2)/(n*nn)
  row=dict(file=f.name,positive_median=float(np.median(pos)),negative_median=float(np.median(neg)),roc_auc=float(auc),positive_quantiles=np.quantile(pos,[.1,.25,.5,.75,.9]).tolist(),missed_positive_quantiles=np.quantile(fn,[.1,.5,.9]).tolist(),thresholds={})
  for t in [.5,.3,.1]:
   pr=p>=t;tp=int((pr&y).sum());fp=int((pr&~y).sum());fnn=int((~pr&y).sum())
   row['thresholds'][str(t)]=dict(dice=2*tp/max(2*tp+fp+fnn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fnn,1),positive_fraction=float(pr.mean()),fp=fp,fn=fnn)
  rows.append(row)
  ims=[np.uint8(np.clip(x,0,1)*255),np.uint8(y)*255,np.uint8(np.clip(p,0,1)*255),np.uint8(p>=.5)*255,np.uint8(p>=.3)*255,np.uint8(p>=.1)*255]
  for j,(im,title) in enumerate(zip(ims,['Fluo','Binary target','Probability 0-1','Threshold 0.5','Threshold 0.3','Threshold 0.1'])):
   panel.paste(Image.fromarray(im).convert('RGB'),(j*320,i*350+30));d.text((j*320+4,i*350+6),title,fill='white')
 panel.save(OUT/'probability_threshold_review.png')
 summary={str(t):{k:float(np.mean([r['thresholds'][str(t)][k] for r in rows])) for k in ['dice','precision','recall','positive_fraction','fp','fn']} for t in [.5,.3,.1]}
 result=dict(scope='Exploratory reused validation specimen; fixed diagnostic thresholds, not selected production settings',threshold_summary=summary,mean_roc_auc=float(np.mean([r['roc_auc'] for r in rows])),rows=rows)
 (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
