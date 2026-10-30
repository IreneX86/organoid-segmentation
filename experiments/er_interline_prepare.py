"""Package one train-label-only ablation; validation NPZs are byte-identical."""
from pathlib import Path
import json,hashlib,shutil,sys,tarfile
import numpy as np
import tifffile as tif
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from er_roi_student import verify
SRC=R/'outputs/er_roi_student_ready_20261029';CAND=R/'outputs/er_interline_negative_review_20261029';OUT=R/'outputs/er_interline_student_ready_20261029'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((SRC/'manifest.json').read_text());c=json.loads((CAND/'manifest.json').read_text());lookup={r['directory']:r for r in c['records']};changed=0
 for r in m['records']:
  f=SRC/r['file'];assert sha(f)==r['sha256'];oldsha=r['sha256']
  if r['split']=='validation':shutil.copyfile(f,OUT/r['file']);assert sha(OUT/r['file'])==oldsha;continue
  name=r['file'][:-4];p=CAND/name/'partial_label_candidate.tif';assert sha(p)==lookup[name]['candidate_sha256']
  y=tif.imread(p)
  with np.load(f) as a:
   old=a['label'];assert np.array_equal(y==1,old==1) and np.all(y[old==0]==0);assert np.all((y==old)|((old==255)&(y==0)))
   changed+=int((y!=old).sum());np.savez_compressed(OUT/r['file'],fluo=a['fluo'],label=y,low=a['low'],high=a['high'])
  r['previous_npz_sha256']=oldsha;r['sha256']=sha(OUT/r['file']);r['interline_candidate_sha256']=sha(p)
 m['ablation']=dict(factor='additional training inter-line weak negatives',changed_pixels=changed,previous_manifest_sha256=sha(SRC/'manifest.json'),validation_identical=True)
 (OUT/'manifest.json').write_text(json.dumps(m,indent=2));print(verify(OUT))
 with tarfile.open(R/'outputs/er_interline_bundle_20261029.tar.gz','w:gz') as t:
  for p in [*OUT.glob('*.npz'),OUT/'manifest.json',R/'code/er_roi_student.py',R/'code/er_partial_supervision.py',R/'code/model/unet.py']:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
 p=R/'outputs/er_interline_bundle_20261029.tar.gz';Path(str(p)+'.sha256').write_text(sha(p)+'  '+p.name+'\n')
 print('changed_pixels',changed,'archive_bytes',p.stat().st_size)
if __name__=='__main__':main()
