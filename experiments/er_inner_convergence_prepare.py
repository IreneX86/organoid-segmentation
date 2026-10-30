"""Freeze a paired inner-validation study without consulting past scores."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
S=R/'outputs/er_diverse_cv_20261030/payload/data/manifest.json'
O=R/'outputs/er_inner_convergence_20261030'
def main():
 m=json.loads(S.read_text());records=m['records'];cells=sorted({r['cell'] for r in records});assert len(cells)==5
 folds=[]
 for i,outer in enumerate(cells):
  inner=cells[(i+1)%5];train=[c for c in cells if c not in [outer,inner]]
  select=lambda cs,old:[r['file'] for r in records if r['cell'] in cs and (not old or r['role']=='old_anchor')]
  f=dict(fold=i,outer_specimen=outer,inner_specimen=inner,training_specimens=train,control_train=select(train,True),expanded_train=select(train,False),inner_validation=select([inner],True),outer_reserved=select([outer],True),excluded_inner_new=[r['file'] for r in records if r['cell']==inner and r['role']!='old_anchor'])
  assert len(f['control_train'])==24 and len(f['inner_validation'])==len(f['outer_reserved'])==8
  assert not(set(f['expanded_train'])&set(f['inner_validation']) or set(f['expanded_train'])&set(f['outer_reserved']) or set(f['inner_validation'])&set(f['outer_reserved']))
  folds.append(f)
 protocol=dict(status='PREPARED_NOT_STARTED',source_manifest_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),records=records,folds=folds,settings=dict(seed_base=20261031,base_filters=16,learning_rate=.0003,batch=4,minimum_steps=1000,maximum_steps=4000,evaluate_every=250,patience_evaluations=4,min_delta=.002,threshold=.5,selection='mean inner-validation crop Dice; candidates from step1000 only; earliest wins ties',batch_sampling='one crop per each of three training specimens; fourth rotates deterministically among the three',normalization='unchanged per-crop p1/p99.8',loss='unchanged BCE+Dice',augmentation='none',outer_evaluation='disabled in selection runner; only after all ten model decisions are locked'),production_approved=False)
 target=O/'protocol.json';assert not target.exists();target.write_text(json.dumps(protocol,indent=2));(O/'protocol.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'  protocol.json\n')
 print([(f['fold'],f['outer_specimen'],f['inner_specimen'],len(f['expanded_train'])) for f in folds])
if __name__=='__main__':main()
