"""3D 뷰어용 자료 (내부 전용, 이 컴퓨터 안에서만 본다). 기존 결과 파일은 읽기만 하고 work/viewer/ 에 새로 쓴다.

사용: venv/bin/python -I viewer_prep.py [--max-vggt 3000000]
- vggt.npz: vggt_graph.py 의 합친 점과 같은 규칙 (파노라마마다 가장 가운데 덩어리, 신뢰도 70 분위 위, 60 m 안, 3 화소마다)
  으로 다시 뽑으며, 그 화소의 색을 GPU 로 다시 그린 뷰 (views_gpu) 에서 붙인다. --max-vggt 개로 고르게 솎는다.
- colmap.npz: merged.json 에 쓰인 COLMAP 조각들의 맞춘 점 + COLMAP 색.
- mesh.npz: 네이버 메시 점 (COLMAP 자세, naver_mesh_world_sfmpose0.npz) 을 100만 개로 솎음.
- lines.npz: 단지 동 V-World 윤곽을 v5 바닥 높이 · v5 꼭대기 높이 두 고리 + 꼭짓점 세로선으로 (v5 높이 상자), 주변 건물 윤곽 (땅 높이 근사).
- cams.npz: VGGT 파노라마 자리 · 회전, COLMAP 파노라마 자리.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import torch
import cube, views, views_gpu

R0 = os.path.join(HERE, '..'); W = os.path.join(R0, 'work'); OUT = os.path.join(W, 'viewer')
PANOS = os.path.expanduser('~/lightlab/runs/streetview/naver/panos')
SCENE = os.path.expanduser('~/lightlab/runs/apt-recon/cheongnim-hyundai/v5/work/scene.json')
maxv = int(sys.argv[sys.argv.index('--max-vggt') + 1]) if '--max-vggt' in sys.argv else 3_000_000
IDS = set(sys.argv[sys.argv.index('--ids') + 1].split(',')) if '--ids' in sys.argv else None
STRIDE = int(sys.argv[sys.argv.index('--stride') + 1]) if '--stride' in sys.argv else 3
OUT_V = sys.argv[sys.argv.index('--out-vggt') + 1] if '--out-vggt' in sys.argv else 'vggt.npz'
ONLY_V = '--only-vggt' in sys.argv
rng = np.random.default_rng(0)
t0 = time.time()
# ---- VGGT (vggt_graph.py 와 같은 규칙) ----
d = os.path.join(W, 'vggt_full')
g = np.load(os.path.join(d, 'graph_sim3.npz')); s, R, T, kept = g['s'], g['R'], g['T'], [int(c) for c in g['ci']]
CH = []
for i, ci in enumerate(kept):
    c = np.load(os.path.join(d, f'chunk_{ci:03d}.npz'))
    E = c['extrinsic'].astype(np.float64); Rk = E[:, :3, :3]; tk = E[:, :3, 3]
    Cc = -np.einsum('sji,sj->si', Rk, tk); pano = c['pano']
    ups = sorted(set(pano), key=list(pano).index)
    Cp = {p: Cc[pano == p].mean(0) for p in ups}
    cf = c['conf'].astype(np.float32)
    CH.append({'i': i, 'ci': ci, 'K': c['intrinsic'].astype(np.float64), 'R': Rk, 't': tk, 'pano': pano, 'face': c['face'],
               'dep': c['depth'].astype(np.float32), 'cf': cf, 'thr': np.quantile(cf, 0.7), 'Cp': Cp, 'ups': ups, 'H': c['depth'].shape[1]})
best = {}
for c in CH:
    cen = np.mean([c['Cp'][p] for p in c['ups']], axis=0)
    for p in c['ups']:
        dd = np.linalg.norm(c['Cp'][p] - cen) * s[c['i']]
        if p not in best or dd < best[p][1]:
            best[p] = (c, dd)
print(f'덩어리 {len(CH)}, 파노라마 {len(best)} ({time.time() - t0:.0f}초)', flush=True)
Ps, Cs = [], []
for n, (p, (c, _)) in enumerate(sorted(best.items())):
    if IDS is not None and p not in IDS:
        continue
    i = c['i']; ks = np.where(c['pano'] == p)[0]
    cw = s[i] * R[i] @ c['Cp'][p] + T[i]
    vl = [(int(str(c['face'][k])[1:4]), int(str(c['face'][k])[4:])) for k in ks]
    F = views.load_faces(os.path.join(PANOS, p), ('front', 'right', 'back', 'left', 'top', 'bottom'))
    img = (views_gpu.render_batch(views_gpu.faces_tensor(F), vl, 90.0, c['H']) * 255).round().byte().cpu().numpy()   # (V,3,H,W)
    for j, k in enumerate(ks):
        vv, uu = np.mgrid[0:c['H']:STRIDE, 0:c['H']:STRIDE]
        uv = np.stack([uu.ravel(), vv.ravel()], 1)
        ok = c['cf'][k][uv[:, 1], uv[:, 0]] >= c['thr']
        uv = uv[ok]
        Ki = np.linalg.inv(c['K'][k]); rays = (Ki @ np.stack([uv[:, 0] + 0.5, uv[:, 1] + 0.5, np.ones(len(uv))])).T
        X = rays * c['dep'][k][uv[:, 1], uv[:, 0]][:, None].astype(np.float64)
        X = (X - c['t'][k]) @ c['R'][k]
        Xw = s[i] * X @ R[i].T + T[i]
        ok2 = np.linalg.norm(Xw - cw, axis=1) < 60.0
        Ps.append(Xw[ok2].astype(np.float32)); Cs.append(img[j][:, uv[ok2, 1], uv[ok2, 0]].T.copy())
    if n % 50 == 0:
        print(f'  {n}/{len(best)} ({time.time() - t0:.0f}초)', flush=True)
P = np.concatenate(Ps); C = np.concatenate(Cs)
print('VGGT 점', len(P), '(전체 · 3 화소마다면 합친 점구름과 같은 수여야 함: 44,760,027)', flush=True)
ix = np.sort(rng.choice(len(P), min(maxv, len(P)), replace=False))
np.savez_compressed(os.path.join(OUT, OUT_V), P=P[ix], C=C[ix], n_all=len(P), stride=STRIDE)
del P, C, Ps, Cs
if ONLY_V:
    print(f'끝 (VGGT 만) {time.time() - t0:.0f}초'); sys.exit(0)
# ---- COLMAP ----
mj = json.load(open(os.path.join(W, 'sfm_full', 'merged.json')))
Pc, Cc_ = [], []
for u in mj['used']:
    a = np.load(os.path.join(W, 'sfm_full', f"aligned_{u['model']}.npz"))
    ok = a['track'] >= 3
    Pc.append(a['P'][ok]); Cc_.append(a['color'][ok])
np.savez_compressed(os.path.join(OUT, 'colmap.npz'), P=np.concatenate(Pc), C=np.concatenate(Cc_))
# ---- 네이버 메시 점 ----
m = np.load(os.path.join(W, 'naver_mesh_world_sfmpose0.npz'))
ix = np.sort(rng.choice(len(m['P']), min(1_000_000, len(m['P'])), replace=False))
np.savez_compressed(os.path.join(OUT, 'mesh.npz'), P=m['P'][ix])
# ---- 선: 동 윤곽 · v5 높이 상자 · 주변 건물 ----
vw = json.load(open(os.path.join(W, 'vworld_outlines.json'))); sc = json.load(open(SCENE))
tw = {t['name']: t for t in sc['info']['towers']}
seg_t, seg_c, lab = [], [], []
for k, v in vw['towers'].items():
    if k not in tw:
        continue
    b = tw[k]['base_z']; h = b + tw[k]['height_m']
    for r in v['rings']:
        r = np.asarray(r)
        for z in (b, h):
            seg_t += [[[*r[j], z], [*r[j + 1], z]] for j in range(len(r) - 1)]
        seg_t += [[[*q, b], [*q, h]] for q in r[:-1]]
    cxy = np.asarray(v['rings'][0]).mean(0); lab.append((k, [cxy[0], cxy[1], h + 2]))
zg = {}
for c in vw['context']:
    r = np.asarray(c['xy'])
    if np.abs(r).max() > 320:
        continue
    seg_c += [[[*r[j], 0.0], [*r[j + 1], 0.0]] for j in range(len(r) - 1)]
np.savez_compressed(os.path.join(OUT, 'lines.npz'), towers=np.array(seg_t, np.float32), context=np.array(seg_c, np.float32),
                    labels=np.array([l[0] for l in lab]), label_pos=np.array([l[1] for l in lab], np.float32))
# ---- 카메라 ----
vwv = np.load(os.path.join(d, 'views_world.npz'))
a0 = np.load(os.path.join(W, 'sfm_full', 'merged.npz'))
np.savez_compressed(os.path.join(OUT, 'cams.npz'), vggt_C=vwv['pano_C'], vggt_R=vwv['pano_R'], colmap_C=a0['C'])
print(f'끝 {time.time() - t0:.0f}초', flush=True)
