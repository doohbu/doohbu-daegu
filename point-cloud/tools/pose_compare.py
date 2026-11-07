"""두 복원의 파노라마 자세 견주기 (같은 v5 좌표에 맞춘 뒤). 사용: python pose_compare.py <A poses npz> <B poses npz> <out.json>
poses npz: pano_pids, pano_R (세계 → 파노라마 틀), pano_C. 결과: 공통 파노라마의 수평 · 높이 차, 회전 차, 방위 차."""
import json, sys
import numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
ia = {str(p): i for i, p in enumerate(A['pano_pids'])}; ib = {str(p): i for i, p in enumerate(B['pano_pids'])}
com = sorted(set(ia) & set(ib))
dh, dz, dr, dy = [], [], [], []
for p in com:
    Ca, Cb = A['pano_C'][ia[p]], B['pano_C'][ib[p]]; Ra, Rb = A['pano_R'][ia[p]], B['pano_R'][ib[p]]
    dh.append(np.linalg.norm(Ca[:2] - Cb[:2])); dz.append(Ca[2] - Cb[2])
    D = Ra @ Rb.T; dr.append(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))))
    fa = Ra.T @ np.array([0, 1.0, 0]); fb = Rb.T @ np.array([0, 1.0, 0])
    dy.append((np.degrees(np.arctan2(fa[0], fa[1]) - np.arctan2(fb[0], fb[1])) + 180) % 360 - 180)
dh, dz, dr, dy = map(np.array, (dh, dz, dr, dy))
q = lambda v: {'median': round(float(np.median(v)), 3), 'p90': round(float(np.percentile(v, 90)), 3), 'max': round(float(np.max(v)), 3)}
res = {'common_panos': len(com), 'horiz_m': q(dh), 'abs_dz_m': q(np.abs(dz)), 'dz_median_m': round(float(np.median(dz)), 3),
       'rot_deg': q(dr), 'abs_yaw_deg': q(np.abs(dy)), 'within_1m': round(float((dh < 1).mean()), 3), 'within_2deg': round(float((dr < 2).mean()), 3)}
json.dump(res, open(sys.argv[3], 'w'), indent=1); print(json.dumps(res, ensure_ascii=False))
