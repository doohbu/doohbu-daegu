"""VGGT 자세의 재투영 오차: COLMAP 모델의 점 트랙 (옆면 관측) 을 VGGT 세계 자세 (views_world.npz) 로 다시 삼각측량해 2048 px 면에서 잰다.
사용: ~/apt-recon/venv/bin/python -I reproj_check.py <COLMAP sparse 폴더> <views_world.npz> <out.json> [--max-points 50000]
- 큐브 옆면 front/right/back/left = VGGT 뷰 v000+00 / v090+00 / v180+00 / v270+00 (같은 회전 · 90° 화각) 이라 그 자세를 그대로 쓴다.
- 내부값은 참값 (f = 1024, 주점 1024) 을 쓴다 → 자세만의 오차.
- 삼각측량: 선형 (DLT) 후 관측 2 개 이상, 모든 카메라 앞에 있는 점만. 같은 트랙의 COLMAP 오차 (같은 관측) 도 함께 잰다.
"""
import json, sys
import numpy as np
import pycolmap

sp, vwp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
maxp = int(sys.argv[sys.argv.index('--max-points') + 1]) if '--max-points' in sys.argv else 50000
FACE2VIEW = {'front': 'v000+00', 'right': 'v090+00', 'back': 'v180+00', 'left': 'v270+00'}
rec = pycolmap.Reconstruction(sp)
v = np.load(vwp)
pose = {(str(p), str(w)): (R, t) for p, w, R, t in zip(v['pid'], v['view'], v['R'], v['t'])}
K = np.array([[1024.0, 0, 1024], [0, 1024, 1024], [0, 0, 1]])
imgs = {}
for im in rec.images.values():
    if not im.has_pose:
        continue
    f, pid = im.name.split('/')[0], im.name.split('/')[1][:-4]
    if f in FACE2VIEW and (pid, FACE2VIEW[f]) in pose:
        R, t = pose[(pid, FACE2VIEW[f])]
        cfw = im.cam_from_world()
        imgs[im.image_id] = {'P': K @ np.hstack([R, t[:, None]]), 'R': R, 't': t,
                             'Pc': K @ cfw.matrix()[:3, :], 'Rc': cfw.rotation.matrix(), 'tc': np.asarray(cfw.translation), 'im': im}
pts = list(rec.points3D.values())
rng = np.random.default_rng(0)
if len(pts) > maxp:
    pts = [pts[i] for i in rng.choice(len(pts), maxp, replace=False)]


def tri(Ps, xs):
    A = []
    for P, x in zip(Ps, xs):
        A.append(x[0] * P[2] - P[0]); A.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A)); X = Vt[-1]
    return X[:3] / X[3]


ev, ec, nobs = [], [], 0
for p in pts:
    obs = [(e.image_id, e.point2D_idx) for e in p.track.elements if e.image_id in imgs]
    if len(obs) < 2:
        continue
    xs = [np.asarray(imgs[i]['im'].points2D[j].xy) for i, j in obs]
    for key, store in (('P', ev), ('Pc', ec)):
        Ps = [imgs[i][key] for i, _ in obs]
        X = tri(Ps, xs)
        ok = True; errs = []
        for P, x in zip(Ps, xs):
            h = P @ np.append(X, 1)
            if h[2] <= 0:
                ok = False; break
            errs.append(np.linalg.norm(h[:2] / h[2] - x))
        if ok:
            store.extend(errs)
    nobs += len(obs)
ev = np.array(ev); ec = np.array(ec)
res = {'images_with_vggt_pose': len(imgs), 'tracks_used': len(pts), 'obs': nobs,
       'vggt_reproj_px': {'n': int(len(ev)), 'median': round(float(np.median(ev)), 2), 'mean': round(float(ev.mean()), 2),
                          'p90': round(float(np.percentile(ev, 90)), 2), 'within_2px': round(float((ev < 2).mean()), 3)} if len(ev) else None,
       'colmap_reproj_px_same_obs': {'n': int(len(ec)), 'median': round(float(np.median(ec)), 2), 'mean': round(float(ec.mean()), 2),
                                     'p90': round(float(np.percentile(ec, 90)), 2)} if len(ec) else None}
json.dump(res, open(outp, 'w'), indent=1)
print(json.dumps(res, ensure_ascii=False))
