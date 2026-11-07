"""네이버 3D 메시 (panos/<id>/model.npz, 수집 세션이 받은 것) 의 축 규약을 정하고, 복원 점과의 거리를 잰다 (독립 형상 대조).

사용: ~/apt-recon/venv/bin/python -I naver_mesh_check.py <panos 폴더> <poses.npz> <points.npz> <out.json> [--n 40] [--radius 25] [--yaw-search 6]
- poses.npz: pano_pids, pano_R (세계 → 파노라마 틀: x 오른쪽 · y 앞 · z 위), pano_C (세계 m). (vggt_graph.py 의 views_world.npz 또는 SfM 판)
- points.npz: P (세계 m) — 비교할 복원 점.
- 메시 틀 가설 4개 (메시 y = 위): H1 앞 −z · 오른쪽 +x, H2 앞 +z · 오른쪽 −x, H3 앞 −z · 오른쪽 −x (거울), H4 앞 +z · 오른쪽 +x (거울).
- 파노라마마다: 카메라에서 --radius m 안 복원 점 → 메시 면까지 거리. 가설마다, 방위 ±--yaw-search° (1° 칸) 를 더 돌려 가장 좋은 값.
"""
import json, os, sys
import numpy as np
import open3d as o3d

panos, posep, ptsp, outp = sys.argv[1:5]
N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 40
RAD = float(sys.argv[sys.argv.index('--radius') + 1]) if '--radius' in sys.argv else 25.0
YS = float(sys.argv[sys.argv.index('--yaw-search') + 1]) if '--yaw-search' in sys.argv else 6.0
H = {'H1': lambda v: np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1), 'H2': lambda v: np.stack([-v[:, 0], v[:, 2], v[:, 1]], 1),
     'H3': lambda v: np.stack([-v[:, 0], -v[:, 2], v[:, 1]], 1), 'H4': lambda v: np.stack([v[:, 0], v[:, 2], v[:, 1]], 1)}
po = np.load(posep); P = np.load(ptsp)['P'].astype(np.float64)
pose = {str(p): (R, C) for p, R, C in zip(po['pano_pids'], po['pano_R'], po['pano_C'])}
ids = [p for p in sorted(pose) if os.path.exists(os.path.join(panos, p, 'model.npz'))]
rng = np.random.default_rng(0)
ids = list(rng.choice(ids, min(N, len(ids)), replace=False))
res = []
for pid in ids:
    z = np.load(os.path.join(panos, pid, 'model.npz')); V = z['vertices'].astype(np.float64); F = z['faces'].astype(np.int64)
    Rw2p, C = pose[pid]
    near = P[np.linalg.norm(P - C, axis=1) < RAD]
    if len(near) < 200:
        continue
    near = near[rng.choice(len(near), min(4000, len(near)), replace=False)]
    best = {}
    for hn, f in H.items():
        vp = f(V)
        for dy in np.arange(-YS, YS + 0.5, 1.0):
            a = np.radians(dy); Rz = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])   # 파노라마 틀에서 시계 방향 dy
            Vw = (Rz @ vp.T).T @ Rw2p + C          # 세계 = R^T p + C
            sc = o3d.t.geometry.RaycastingScene()
            m = o3d.t.geometry.TriangleMesh(); m.vertex.positions = o3d.core.Tensor(Vw.astype(np.float32)); m.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
            sc.add_triangles(m)
            d = sc.compute_distance(o3d.core.Tensor(near.astype(np.float32))).numpy()
            med = float(np.median(d))
            if hn not in best or med < best[hn][0]:
                best[hn] = (med, float(dy), float(np.mean(d < 0.5)), float(np.mean(d < 1.0)))
    hb = min(best, key=lambda k: best[k][0])
    res.append({'pid': pid, 'n': int(len(near)), 'best': hb, 'median_m': round(best[hb][0], 3), 'yaw_fix_deg': best[hb][1],
                'within_0.5m': round(best[hb][2], 3), 'within_1m': round(best[hb][3], 3),
                'all': {k: round(v[0], 3) for k, v in best.items()}})
    print(res[-1], flush=True)
import collections
cnt = collections.Counter(r['best'] for r in res)
summ = {'panos': len(res), 'best_hypothesis_counts': dict(cnt),
        'median_dist_m': round(float(np.median([r['median_m'] for r in res])), 3) if res else None,
        'within_0.5m_median': round(float(np.median([r['within_0.5m'] for r in res])), 3) if res else None,
        'within_1m_median': round(float(np.median([r['within_1m'] for r in res])), 3) if res else None,
        'yaw_fix_abs_median_deg': round(float(np.median([abs(r['yaw_fix_deg']) for r in res])), 1) if res else None}
json.dump({'summary': summ, 'panos': res}, open(outp, 'w'), indent=1)
print(json.dumps(summ))
