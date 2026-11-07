"""네이버 3D 메시 → 세계 좌표 점 (면 위 고르게 뽑음). 자세는 복원 (VGGT 또는 SfM) 의 파노라마 자세를 쓴다.
사용: ~/apt-recon/venv/bin/python -I naver_mesh_world.py <panos 폴더> <poses.npz> <out.npz> [--radius 45] [--density 4] [--hyp H2] [--yaw-fix json]
- 규약 (naver_mesh_check.py 로 정함): H2 = 메시 +z 가 파노라마 앞, −x 가 오른쪽, +y 가 위 (40곳 중 38곳).
- 점: 삼각형 넓이에 비례해 ㎡ 당 --density 개, 카메라에서 --radius m 안.
- --yaw-fix: naver_mesh_check 결과 json 을 주면 파노라마마다 찾은 방위 보정을 쓴다 (없으면 0).
"""
import json, os, sys
import numpy as np

panos, posep, outp = sys.argv[1:4]
RAD = float(sys.argv[sys.argv.index('--radius') + 1]) if '--radius' in sys.argv else 45.0
DEN = float(sys.argv[sys.argv.index('--density') + 1]) if '--density' in sys.argv else 4.0
yfix = {}
if '--yaw-fix' in sys.argv:
    for r in json.load(open(sys.argv[sys.argv.index('--yaw-fix') + 1]))['panos']:
        yfix[r['pid']] = r['yaw_fix_deg']
po = np.load(posep)
rng = np.random.default_rng(0)
Ps, PN, Cs, pids = [], [], [], []
for p, Rw2p, C in zip(po['pano_pids'], po['pano_R'], po['pano_C']):
    p = str(p); f = os.path.join(panos, p, 'model.npz')
    if not os.path.exists(f):
        continue
    z = np.load(f); V = z['vertices'].astype(np.float64); F = z['faces'].astype(np.int64)
    vp = np.stack([-V[:, 0], V[:, 2], V[:, 1]], 1)            # H2 → 파노라마 틀 (x 오른쪽 · y 앞 · z 위)
    a = np.radians(yfix.get(p, 0.0)); Rz = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Vw = (Rz @ vp.T).T @ Rw2p + C
    A, B, Cc = Vw[F[:, 0]], Vw[F[:, 1]], Vw[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, Cc - A), axis=1)
    cen = (A + B + Cc) / 3
    ok = (np.linalg.norm(cen - C, axis=1) < RAD) & (area > 0)
    n = rng.poisson(area[ok] * DEN)
    idx = np.repeat(np.where(ok)[0], n)
    if len(idx) == 0:
        continue
    r1 = np.sqrt(rng.random(len(idx))); r2 = rng.random(len(idx))
    X = (1 - r1)[:, None] * A[idx] + (r1 * (1 - r2))[:, None] * B[idx] + (r1 * r2)[:, None] * Cc[idx]
    Ps.append(X.astype(np.float32)); PN.append(np.full(len(X), len(pids), np.int32)); Cs.append(C); pids.append(p)
P = np.concatenate(Ps)
np.savez_compressed(outp, P=P, pano=np.concatenate(PN), C=np.array(Cs), pids=np.array(pids), track=np.full(len(P), 3, np.int16))
print('메시', len(pids), '곳, 점', len(P))
