"""Fixed, out-of-domain nERdy+ Fluo-only pilot; outputs are NOT ground truth.
External AGPL model is imported from an independent clone, not vendored.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from er_nerdy_reference_pilot import sha, normalize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/er_nerdy_fluo_five_20261029'
CLONE=Path('/private/tmp/er-nerdy-reference-20261029')
SELECT=('23298','23299','23961','23965','23966')

def main():
    OUT.mkdir(exist_ok=False)
    np.int=int; np.float=float; np.complex=complex
    sys.path.insert(0,str(CLONE/'nERdy+'))
    import torch
    from model import D4nERdy
    from postprocessing import postprocessing
    torch.set_num_threads(4)
    weights=CLONE/'nERdy+'/ 'NNet_groupy_p4m_v2_VecAdam.pth'
    model=D4nERdy(1,1)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)); model.eval()
    manifests=list((ROOT/'outputs/er_coherent_fiber_full_20261028').glob('*/*/manifest.json'))
    overview=Image.new('RGB',(1200,430*5),'black')
    rows=[]
    for i,key in enumerate(SELECT):
        mp=next(p for p in manifests if f'_{key}_' in p.parent.name)
        m=json.loads(mp.read_text()); z=m['source_shape'][0]//2-1
        raw=tifffile.imread(m['source_fluo'],key=z)
        assert raw.shape==(1200,1200)
        folder=OUT/mp.parent.name; folder.mkdir()
        tifffile.imwrite(folder/'Fluo_input.tif',raw)
        a=normalize(raw)
        # One fixed full-plane normalization, overlapping tiles with tapered blending.
        accum=np.zeros_like(a); denom=np.zeros_like(a)
        tile=384
        positions=[0,272,544,816]
        window=np.maximum(np.hanning(tile),0.05).astype(np.float32)
        weight=window[:,None]*window[None,:]
        t=time.monotonic()
        with torch.inference_mode():
            for y in positions:
                for x in positions:
                    pred=torch.sigmoid(model(torch.from_numpy(a[None,None,y:y+tile,x:x+tile]))).squeeze().numpy()
                    accum[y:y+tile,x:x+tile]+=pred*weight
                    denom[y:y+tile,x:x+tile]+=weight
        prob=accum/denom
        scaled=(prob-prob.min())/max(float(np.ptp(prob)),1e-8)
        mask=postprocessing(scaled)>0
        tifffile.imwrite(folder/'candidate_mask_NOT_GT.tif',mask.astype('uint8')*255)
        tifffile.imwrite(folder/'model_sigmoid_NOT_calibrated.tif',prob)
        rgb=np.repeat((a*255).astype('uint8')[...,None],3,2)
        over=rgb.copy(); over[mask]=(0.45*over[mask]+0.55*np.array([255,220,0])).astype('uint8')
        panels=[Image.fromarray(rgb),Image.fromarray(over),Image.fromarray(mask.astype('uint8')*255).convert('RGB')]
        strip=Image.new('RGB',(3600,1240),'black')
        for j,panel in enumerate(panels):
            strip.paste(panel,(1200*j,40))
            overview.paste(panel.resize((400,400)),(400*j,430*i+30))
        caption=f'{mp.parent.name} Z{z+1} | Fluo / nERdy+ overlay / candidate mask (OOD, NOT GT)'
        ImageDraw.Draw(strip).text((10,10),caption,fill='white'); strip.save(folder/'review.png')
        ImageDraw.Draw(overview).text((5,i*430+8),caption,fill='white')
        rows.append(dict(cell=mp.parent.name,z_one_based=z+1,source_fluo=m['source_fluo'],input_sha256=sha(folder/'Fluo_input.tif'),mask_sha256=sha(folder/'candidate_mask_NOT_GT.tif'),positive_fraction=float(mask.mean()),seconds=time.monotonic()-t))
        print(json.dumps(rows[-1]),flush=True)
        overview.save(OUT/'five_fluo_review.png')
    (OUT/'manifest.json').write_text(json.dumps(dict(model='public nERdy+',weights_sha256=sha(weights),repo='https://github.com/NanoscopyAI/nERdy',input='Fluo ONLY; no ER channel or masks read',preprocessing='full-plane p1-p99.8 clipped;384 tiles stride272 tapered blending',postprocessing='authors rolling-ball then Otsu after minmax sigmoid rescaling',warning='ER-trained model applied out of domain. Not validated ER identity; not training labels.',selection='five training cells; middle slice fixed before inference; no validation/test',records=rows),indent=2))

if __name__=='__main__': main()
