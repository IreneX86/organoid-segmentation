"""Separate Fluo-only U-Net workflow for ER-positive / exterior-negative labels.

Preparation is local; training explicitly requires one GPU. Test organoid is
excluded. No unmarked intracellular pixel becomes a negative label.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import sys
import numpy as np
import tifffile
from er_fluo_student import SEED, TEST_CELL, VALIDATION_CELL, sha256
from er_positive_paired_crops import best_tile
from er_partial_supervision import exterior_background, masked_bce_dice

CONFIG = dict(model='unet_2d', base_filters=16, patch_size=384, batch_size=4,
              steps_per_epoch=48, max_epochs=20, patience=5, learning_rate=.001,
              threshold=.5, seed=SEED, normalization='full-plane Fluo p1-p99.8',
              loss='masked_bce_dice', augmentation='paired flips and rot90 only',
              sampling='two positive-rich and two exterior-negative-rich patches per batch',
              negative_rule='sigma2; dual-channel p20; edge-connected; erode4; teacher-distance>12',
              evaluation='partial-label agreement only; no biological full-mask accuracy')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(mask_root, output):
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(mask_root.glob('*/*/manifest.json'))
    if len(paths) != 8:
        raise ValueError('Expected eight source manifests')
    output.mkdir(parents=True)
    dump(output/'config.json', CONFIG)  # Locked before validation pixels are read.
    records, sources, skipped = [], [], []
    for path in paths:
        cell = path.parent.name
        if cell == TEST_CELL:
            continue  # Do not open test manifest, channels or labels.
        split = 'validation' if cell == VALIDATION_CELL else 'train'
        info = json.loads(path.read_text())
        shape = tuple(info['source_shape'])
        if info['source_z_start_zero_based'] != 0 or info['source_z_stop_exclusive'] != shape[0]:
            raise ValueError('Incomplete source')
        files = [Path(info['source_er']), Path(info['source_fluo']), path.parent/'ER_mask_line.tif']
        hashes = [info['source_er_sha256'], info['source_fluo_sha256'], info['outputs']['ER_mask_line']['sha256']]
        for file, expected in zip(files, hashes):
            if sha256(file) != expected:
                raise ValueError(f'Changed source: {file}')
        sources.append(dict(cell=cell, split=split, shape=list(shape), source_hashes=hashes,
                            manifest_sha256=sha256(path)))
        with tifffile.TiffFile(files[0]) as ef, tifffile.TiffFile(files[1]) as ff, tifffile.TiffFile(files[2]) as mf:
            if any(t.series[0].shape != shape or len(t.pages) != shape[0] for t in (ef,ff,mf)):
                raise ValueError('Alignment mismatch')
            for z in range(shape[0]):
                er, fluo, old = [t.pages[z].asarray() for t in (ef,ff,mf)]
                if old.dtype != np.uint8 or not np.isin(old,[0,255]).all():
                    raise ValueError('Invalid old mask')
                pos = old > 0
                bg, thresholds = exterior_background(er, fluo, pos)
                labels = np.full(shape[1:],255,np.uint8)
                labels[bg] = 0
                labels[pos] = 1
                lo, hi = np.percentile(fluo.astype(np.float32),[1,99.8])
                for stream, selection in [('positive',pos),('negative',bg)]:
                    if not selection.any():
                        skipped.append(dict(cell=cell,z=z,stream=stream))
                        continue
                    y,x,_ = best_tile(selection)
                    region = np.s_[y:y+384,x:x+384]
                    image, target = fluo[region], labels[region]
                    name = f'{cell}_z{z:03d}_{stream}.npz'
                    np.savez_compressed(output/name, fluo=image, labels=target,
                                        low=np.float32(lo), high=np.float32(hi))
                    with np.load(output/name,allow_pickle=False) as d:
                        assert np.array_equal(d['fluo'],image)
                        assert np.array_equal(d['labels']==1,pos[region])
                        assert np.array_equal(d['labels']==0,bg[region])
                    records.append(dict(file=name,sha256=sha256(output/name),cell=cell,
                                        split=split,stream=stream,z=z,y=y,x=x,
                                        positive=int((target==1).sum()),negative=int((target==0).sum()),
                                        ignore=int((target==255).sum()),negative_thresholds=thresholds))
        print(f'PREPARED {cell}: {shape[0]} planes',flush=True)
    dump(output/'dataset_manifest.json',dict(config=CONFIG,sources=sources,records=records,
         skipped=skipped,excluded_test=TEST_CELL,training_started=False,
         semantics='1 old pseudo-positive; 0 weak exterior negative; 255 unknown'))
    return verify(output)


def load_patch(root, row):
    with np.load(root/row['file'],allow_pickle=False) as d:
        image = d['fluo'].astype(np.float32)
        low, high = float(d['low']),float(d['high'])
        image = np.clip((image-low)/(high-low),0,1) if high>low else np.zeros_like(image)
        labels = d['labels'].copy()
    if labels.shape != (384,384) or image.shape != labels.shape or not np.isin(labels,[0,1,255]).all():
        raise ValueError('Invalid patch')
    if not np.isfinite(image).all():
        raise ValueError('Nonfinite image')
    return image[...,None], labels[...,None].astype(np.float32)


def verify(root):
    m=json.loads((root/'dataset_manifest.json').read_text())
    if m['config'] != CONFIG:
        raise ValueError('Configuration mismatch')
    cells={split:{r['cell'] for r in m['records'] if r['split']==split} for split in ['train','validation']}
    if len(cells['train'])!=6 or cells['validation']!={VALIDATION_CELL} or TEST_CELL in cells['train']:
        raise ValueError('Split changed')
    for r in m['records']:
        if sha256(root/r['file'])!=r['sha256']:
            raise ValueError('Patch hash mismatch')
        _,y=load_patch(root,r)
        for code,key in [(1,'positive'),(0,'negative'),(255,'ignore')]:
            if int((y==code).sum())!=r[key]:
                raise ValueError('Label count mismatch')
    for split in cells:
        for stream in ['positive','negative']:
            if not any(r['split']==split and r['stream']==stream for r in m['records']):
                raise ValueError('Missing sampling stream')
    result=dict(status='PASS',patches=len(m['records']),
                counts={s:sum(r['split']==s for r in m['records']) for s in cells},
                skipped=len(m['skipped']),test_opened=False)
    dump(root/'verification.json',result)
    return result


def train(root, output):
    import tensorflow as tf
    from model.unet import build_unet_2d
    if output.exists():
        raise FileExistsError(output)
    verify(root)
    m=json.loads((root/'dataset_manifest.json').read_text())
    gpu=tf.config.list_physical_devices('GPU')
    if len(gpu)!=1:
        raise RuntimeError('Training requires exactly one GPU; CPU training forbidden')
    tf.config.experimental.set_memory_growth(gpu[0],True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    with tf.device('/GPU:0'):
        a=tf.matmul(tf.ones((128,128)),tf.ones((128,128)))
    if 'GPU' not in a.device or float(a[0,0])!=128:
        raise RuntimeError('GPU preflight failed')
    print('GPU_PREFLIGHT_PASS',flush=True)
    output.mkdir(parents=True)
    dump(output/'config.json',CONFIG)
    dump(output/'environment.json',dict(python=sys.version,tensorflow=tf.__version__,
         numpy=np.__version__,platform=platform.platform(),build=tf.sysconfig.get_build_info()))
    dump(output/'dataset_manifest.json',m)

    class Training(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.epoch=0
            self.groups={s:[r for r in m['records'] if r['split']=='train' and r['stream']==s]
                         for s in ['positive','negative']}
        def __len__(self): return CONFIG['steps_per_epoch']
        def on_epoch_end(self): self.epoch+=1
        def __getitem__(self,index):
            rng=np.random.default_rng(SEED+self.epoch*100003+index)
            xx,yy=[],[]
            for s in ['positive','positive','negative','negative']:
                group=self.groups[s]
                # Uniform organoid sampling prevents longer stacks dominating.
                cell=rng.choice(sorted({r['cell'] for r in group}))
                subset=[r for r in group if r['cell']==cell]
                x,y=load_patch(root,subset[int(rng.integers(len(subset)))])
                k=int(rng.integers(4)); x,y=np.rot90(x,k),np.rot90(y,k)
                if rng.random()<.5: x,y=np.flip(x,0),np.flip(y,0)
                xx.append(x); yy.append(y)
            return np.stack(xx),np.stack(yy)

    class Validation(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.rows=[r for r in m['records'] if r['split']=='validation']
        def __len__(self): return (len(self.rows)+3)//4
        def __getitem__(self,index):
            pairs=[load_patch(root,r) for r in self.rows[index*4:index*4+4]]
            return np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])

    model=build_unet_2d(1,16)
    model.compile(optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),loss=masked_bce_dice)
    checkpoint=output/'er_partial_student.keras'
    history=model.fit(Training(),validation_data=Validation(),epochs=CONFIG['max_epochs'],verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint),monitor='val_loss',save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=CONFIG['patience'],restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(output/'training_history.csv'))])
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError('Nonfinite training history')
    model=tf.keras.models.load_model(checkpoint,compile=False)
    tp=fp=fn=0
    rows=[]
    validation_rows=[r for r in m['records'] if r['split']=='validation']
    preview_indices={0,len(validation_rows)//2,len(validation_rows)-1}
    for index,r in enumerate(validation_rows):
        x,y=load_patch(root,r)
        p=model(x[None],training=False).numpy()[0]
        if not np.isfinite(p).all(): raise RuntimeError('Nonfinite output')
        pred=p>=.5; valid=y!=255; truth=y==1
        a=int((pred & truth & valid).sum()); b=int((pred & ~truth & valid).sum()); c=int((~pred & truth & valid).sum())
        tp+=a;fp+=b;fn+=c
        rows.append(dict(file=r['file'],tp=a,fp=b,fn=c,
                    unknown_predicted_fraction=float(pred[~valid].mean()) if (~valid).any() else None))
        if index in preview_indices:
            # Fixed examples, not cherry-picked after inspecting performance.
            preview=output/'validation_previews'
            preview.mkdir(exist_ok=True)
            np.savez_compressed(preview/r['file'],fluo_normalized=x[...,0],
                                partial_label=y[...,0].astype(np.uint8),probability=p[...,0])
    dump(output/'validation_partial_metrics.json',dict(tp=tp,fp=fp,fn=fn,
         dice=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
         note='Overlapping sampled patches; known-pixel agreement, not whole-image ER accuracy',rows=rows))
    dump(output/'result.json',dict(epochs=len(history.history['loss']),
         best_epoch=int(np.argmin(history.history['val_loss']))+1,checkpoint_sha256=sha256(checkpoint),
         test_opened=False,production_approved=False))
    print('ER_PARTIAL_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','train'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode!='verify' and a.output is None: p.error('--output required')
    if a.mode=='prepare': print(json.dumps(prepare(a.data,a.output)))
    elif a.mode=='verify': print(json.dumps(verify(a.data)))
    else: train(a.data,a.output)
