"""같은 광선에서 복원 점 거리 ÷ 네이버 메시 거리 (축척 대조). 자세 = poses.npz 의 파노라마 자세, 메시 규약 H2.
사용: ~/apt-recon/venv/bin/python -I range_ratio.py <panos> <poses.npz> <fused.npz> <out.json> [--n 60]
- fused.npz 의 pano 번호 · pids 로 그 파노라마 자기 뷰의 점만 쓴다 (다른 파노라마 점 섞임 없음).
- 거리 구간 (5–10, 10–20, 20–40 m) 마다 비 중앙값.
"""
import json, os, sys
import numpy as np
import open3d as o3d
panos, posep, fp, outp = sys.argv[1:5]
N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 60
po = np.load(posep); fz = np.load(fp)
pose = {str(p): (R, C) for p, R, C in zip(po['pano_pids'], po['pano_R'], po['pano_C'])}
fpids = [str(p) for p in fz['pids']]
P = fz['P']; PN = fz['pano']
order = np.argsort(PN, kind='stable'); PN_s = PN[order]
rng = np.random.default_rng(1)
ids = [p for p in fpids if p in pose and os.path.exists(os.path.join(panos, p, 'model.npz'))]
ids = list(rng.choice(ids, min(N, len(ids)), replace=False))
rows = []; allr = {'5-10': [], '10-20': [], '20-40': []}
for p in ids:
    i = fpids.index(p)
    lo, hi = np.searchsorted(PN_s, i), np.searchsorted(PN_s, i, 'right')
    X = P[order[lo:hi]].astype(np.float64)
    if len(X) < 500:
        continue
    X = X[rng.choice(len(X), min(20000, len(X)), replace=False)]
    Rw2p, C = pose[p]
    z = np.load(os.path.join(panos, p, 'model.npz')); V = z['vertices'].astype(np.float64); F = z['faces'].astype(np.int32)
    vp = np.stack([-V[:, 0], V[:, 2], V[:, 1]], 1); Vw = vp @ Rw2p + C
    sc = o3d.t.geometry.RaycastingScene(); m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(Vw.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(F); sc.add_triangles(m)
    d = X - C; r = np.linalg.norm(d, axis=1); u = d / r[:, None]
    ans = sc.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(C, X.shape), u]).astype(np.float32)))
    rm = ans['t_hit'].numpy()
    ok = np.isfinite(rm) & (rm > 2) & (r > 2) & (np.abs(np.log(r / np.maximum(rm, 1e-6))) < 0.5)
    ratio = r[ok] / rm[ok]
    row = {'pid': p, 'n': int(ok.sum())}
    for k, (a, b) in {'5-10': (5, 10), '10-20': (10, 20), '20-40': (20, 40)}.items():
        mk = (rm[ok] >= a) & (rm[ok] < b)
        if mk.sum() >= 50:
            row[k] = round(float(np.median(ratio[mk])), 4); allr[k].append(row[k])
    rows.append(row)
summ = {k: {'panos': len(v), 'median_ratio': round(float(np.median(v)), 4) if v else None,
            'p25': round(float(np.percentile(v, 25)), 4) if v else None, 'p75': round(float(np.percentile(v, 75)), 4) if v else None} for k, v in allr.items()}
json.dump({'summary': summ, 'panos': rows}, open(outp, 'w'), indent=1)
print(json.dumps(summ))
