"""mvs_test 작업 폴더 고치기: (1) 영상 링크 → 회색조 JPEG 사본, (2) patch-match.cfg · fusion.cfg 를 작업 폴더 영상끼리의 짝으로.
사용: venv-mvs/bin/python -I mvs_fix_workspace.py <작업 폴더> [--src 20]"""
import os, sys
import numpy as np
from PIL import Image
import pycolmap
ws = sys.argv[1]; nsrc = int(sys.argv[sys.argv.index('--src') + 1]) if '--src' in sys.argv else 20
imgs = []
for f in sorted(os.listdir(os.path.join(ws, 'images'))):
    for g in sorted(os.listdir(os.path.join(ws, 'images', f))):
        imgs.append(f'{f}/{g}')
for n in imgs:
    p = os.path.join(ws, 'images', n)
    if os.path.islink(p):
        src = os.path.realpath(p); os.unlink(p)
        Image.open(src).convert('L').save(p, quality=95)
rec = pycolmap.Reconstruction(os.path.join(ws, 'sparse'))
name2id = {im.name: iid for iid, im in rec.images.items()}
ids = [name2id[n] for n in imgs if n in name2id]
idset = set(ids)
co = {i: {} for i in ids}
for p in rec.points3D.values():
    obs = [e.image_id for e in p.track.elements if e.image_id in idset]
    for a in obs:
        for b in obs:
            if a != b:
                co[a][b] = co[a].get(b, 0) + 1
lines_pm, lines_f, nsrc_list = [], [], []
for i in ids:
    src = [b for b, c in sorted(co[i].items(), key=lambda kv: -kv[1]) if c >= 15][:nsrc]
    if len(src) < 2:
        continue
    nm = rec.images[i].name
    lines_pm += [nm, ', '.join(rec.images[b].name for b in src)]
    lines_f.append(nm); nsrc_list.append(len(src))
open(os.path.join(ws, 'stereo', 'patch-match.cfg'), 'w').write('\n'.join(lines_pm) + '\n')
open(os.path.join(ws, 'stereo', 'fusion.cfg'), 'w').write('\n'.join(lines_f) + '\n')
print('영상', len(imgs), '모델 안', len(ids), '기준 영상', len(lines_f), '원본 짝 수 중앙', int(np.median(nsrc_list)) if nsrc_list else 0)
