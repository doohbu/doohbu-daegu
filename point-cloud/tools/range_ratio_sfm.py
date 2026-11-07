"""COLMAP 점 거리 ÷ 네이버 메시 거리 (같은 광선). 그 파노라마 면이 본 점 (트랙) 만 쓴다.
사용: ~/apt-recon/venv/bin/python -I range_ratio_sfm.py <panos> <sfm 폴더> <model id> <out.json>
- 맞춤 변환 · 파노라마 자세는 eval_<id>.json · aligned_<id>.npz (eval_sfm.py) 의 것을 쓴다. 메시 규약 H2.
"""
import json, os, sys
import numpy as np
import open3d as o3d
import pycolmap
panos, sfm, mid, outp = sys.argv[1:5]
ev = json.load(open(os.path.join(sfm, f'eval_{mid}.json'))); al = np.load(os.path.join(sfm, f'aligned_{mid}.npz'))
M = np.array(ev['transform']['M']); T = np.array(ev['transform']['T'])
pose = {str(p): (R, C) for p, R, C in zip(al['pano_pids'], al['pano_R'], al['pano_C'])}
rec = pycolmap.Reconstruction(os.path.join(sfm, 'sparse', mid))
by_pano = {}
for pid3, p in rec.points3D.items():
    if p.track.length() < 3:
        continue
    X = M @ np.asarray(p.xyz) + T
    for e in p.track.elements:
        nm = rec.images[e.image_id].name; pid = nm.split('/')[1][:-4]
        by_pano.setdefault(pid, []).append(X)
allr = {'5-10': [], '10-20': [], '20-40': []}; rows = []
for pid, (Rw2p, C) in pose.items():
    f = os.path.join(panos, pid, 'model.npz')
    if not os.path.exists(f) or pid not in by_pano:
        continue
    X = np.unique(np.array(by_pano[pid]), axis=0)
    if len(X) < 100:
        continue
    z = np.load(f); V = z['vertices'].astype(np.float64); F = z['faces'].astype(np.int32)
    Vw = np.stack([-V[:, 0], V[:, 2], V[:, 1]], 1) @ Rw2p + C
    sc = o3d.t.geometry.RaycastingScene(); m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(Vw.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(F); sc.add_triangles(m)
    d = X - C; r = np.linalg.norm(d, axis=1); u = d / r[:, None]
    rm = sc.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(C, X.shape), u]).astype(np.float32)))['t_hit'].numpy()
    ok = np.isfinite(rm) & (rm > 2) & (r > 2) & (np.abs(np.log(r / np.maximum(rm, 1e-6))) < 0.5)
    ratio = r[ok] / rm[ok]; row = {'pid': pid, 'n': int(ok.sum())}
    for k, (a, b) in {'5-10': (5, 10), '10-20': (10, 20), '20-40': (20, 40)}.items():
        mk = (rm[ok] >= a) & (rm[ok] < b)
        if mk.sum() >= 20:
            row[k] = round(float(np.median(ratio[mk])), 4); allr[k].append(row[k])
    rows.append(row)
summ = {k: {'panos': len(v), 'median_ratio': round(float(np.median(v)), 4) if v else None,
            'p25': round(float(np.percentile(v, 25)), 4) if v else None, 'p75': round(float(np.percentile(v, 75)), 4) if v else None} for k, v in allr.items()}
json.dump({'summary': summ, 'panos': rows}, open(outp, 'w'), indent=1); print(json.dumps(summ))
