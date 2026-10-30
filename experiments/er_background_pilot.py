"""Fixed six-training-plane background pilot; no fitting, test data, or GPU."""
import sys
import json
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from er_partial_supervision import exterior_background
from er_fluo_student import sha256, TEST_CELL, VALIDATION_CELL
from er_positive_paired_crops import display


def run(dataset, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((dataset/'manifest.json').read_text())
    output.mkdir(parents=True)
    rows, figures = [], []
    for source in manifest['sources']:
        cell = source['cell']
        assert cell not in {TEST_CELL, VALIDATION_CELL}
        path = Path(source['manifest'])
        assert sha256(path) == source['manifest_sha256']
        info = json.loads(path.read_text())
        z = info['source_shape'][0]//2
        arrays = []
        paths = [Path(info['source_er']), Path(info['source_fluo']), path.parent/'ER_mask_line.tif']
        hashes = [info['source_er_sha256'], info['source_fluo_sha256'], info['outputs']['ER_mask_line']['sha256']]
        for file, expected in zip(paths, hashes):
            assert sha256(file) == expected
            with tifffile.TiffFile(file) as t:
                arrays.append(t.pages[z].asarray())
        er, fluo, old = arrays
        assert old.dtype == np.uint8 and np.isin(old, [0,255]).all()
        positive = old > 0
        background, thresholds = exterior_background(er, fluo, positive)
        labels = np.full(old.shape, 255, np.uint8)
        labels[background] = 0
        labels[positive] = 1
        target = output/f'{cell}_z{z:03d}.npz'
        np.savez_compressed(target, fluo=fluo, partial_label=labels)
        with np.load(target, allow_pickle=False) as saved:
            assert np.array_equal(saved['fluo'], fluo)
            assert np.array_equal(saved['partial_label']==1, positive)
            assert not (background & positive).any()
        row = {'cell':cell,'split':'train','z_zero_based':z,'file':target.name,
               'sha256':sha256(target),'positive_pixels':int(positive.sum()),
               'weak_negative_pixels':int(background.sum()),
               'ignore_pixels':int((labels==255).sum()),'thresholds_ER_Fluo':thresholds}
        rows.append(row)
        rgb = np.repeat(display(fluo)[...,None],3,axis=-1)
        overlay = rgb.copy()
        overlay[positive]=[255,220,0]
        overlay[background]=[40,140,255]
        panel = Image.new('RGB',(800,430))
        panel.paste(Image.fromarray(rgb).resize((400,400)),(0,30))
        panel.paste(Image.fromarray(overlay).resize((400,400)),(400,30))
        ImageDraw.Draw(panel).text((4,8),f'{cell} Z{z+1}: Fluo | yellow positive, blue weak exterior negative',fill='white')
        figures.append(panel)
    canvas = Image.new('RGB',(800,430*len(figures)))
    for i, panel in enumerate(figures):
        canvas.paste(panel,(0,i*430))
    canvas.save(output/'background_review.png')
    result = {'training_started':False,'training_ready':False,
              'limitation':'Exterior negatives alone do not supervise intra-tissue false positives.',
              'source_manifest_sha256':sha256(dataset/'manifest.json'),'records':rows}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    run(Path('outputs/er_positive_patch_dataset_20261029'), Path('outputs/er_background_pilot_20261029'))
