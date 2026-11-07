"""v5 동 메시를 수평으로 잘라 윤곽 선분을 얻는다 (v5 의 동 벽 = V-World dt_d198 · LT_C_SPBD 윤곽을 세운 것이라, 자른 선 = 그 윤곽 + 모델이 붙인 돌출).
사용: ~/apt-recon/venv/bin/python tower_outlines.py <v5_scene.npz> <v5 scene.json> <out.json>
높이: 바닥 + 1.5 m (1층) 와 바닥 + 20 m (중간층). 닫힌 고리 넓이 (신발끈 공식, 절댓값 합) 를 scene.json 의 area_m2 와 견준다."""
import json, sys
import numpy as np
import trimesh

z = np.load(sys.argv[1]); s = json.load(open(sys.argv[2]))
V = z['V']; F = z['F']; G = z['G']; T = z['tower']
out = {}


def shoelace(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


for t in s['info']['towers']:
    n = int(t['name']); mk = (G == 0) & (T == n)
    f = F[mk]; used = np.unique(f); remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    m = trimesh.Trimesh(V[used].astype(np.float64), remap[f], process=False)
    res = {'area_m2_scene': t['area_m2'], 'base_z': t['base_z'], 'height_m': t['height_m'], 'floors': t['floors']}
    cands = []
    for dzz in (10.0, 11.5, 13.0, 14.5, 16.0, 17.5, 20.0, 22.5, 25.0, 28.0):
        sec = m.section(plane_origin=[0, 0, t['base_z'] + dzz], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        loops = [np.asarray(d)[:, :2] for d in sec.discrete]
        closed = [l for l in loops if len(l) > 3 and np.linalg.norm(l[0] - l[-1]) < 1e-3]
        a = max([abs(shoelace(l)) for l in closed] or [0])
        cands.append((abs(a - t['area_m2']) / t['area_m2'], dzz))
    best_dz = min(cands)[1] if cands else 20.0
    res['wall_dz'] = best_dz
    for name, dzz in (('z1', 1.5), ('wall', best_dz)):
        sec = m.section(plane_origin=[0, 0, t['base_z'] + dzz], plane_normal=[0, 0, 1])
        if sec is None:
            res[name] = None; continue
        loops = [np.asarray(d)[:, :2] for d in sec.discrete]
        closed = [l for l in loops if len(l) > 3 and np.linalg.norm(l[0] - l[-1]) < 1e-3]
        areas = [shoelace(l) for l in closed]
        res[name] = {'loops': len(loops), 'closed': len(closed), 'area_abs_max': round(max([abs(a) for a in areas] or [0]), 1),
                     'segments': [[[round(float(a), 3) for a in l[i]], [round(float(a), 3) for a in l[i + 1]]]
                                  for l in loops for i in range(len(l) - 1)]}
    res['top_z'] = round(float(m.vertices[:, 2].max()), 2)
    out[t['name']] = res
    print(t['name'], res['area_m2_scene'], res['z1']['area_abs_max'] if res['z1'] else None, res['z1']['closed'] if res['z1'] else None,
          res['wall']['area_abs_max'] if res['wall'] else None, res['wall_dz'], 'top', res['top_z'], 'base+H', round(t['base_z'] + t['height_m'], 2))
json.dump(out, open(sys.argv[3], 'w'))
