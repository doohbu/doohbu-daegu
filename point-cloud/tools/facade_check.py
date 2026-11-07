"""입면 면 자리 · 높이 점검 (변 단위). 입력: 맞춘 점 npz (P, C), vworld_outlines.json, v5 scene.json.

- 벽 변: 단지 동 V-World 윤곽의 변 중 길이 ≥ --min-len (기본 12 m). 점은 가장 가까운 변 (짧은 꺾임 변 포함) 하나에만 붙인다.
- 그 변에 붙는 점: 변에 수직 거리 |d| ≤ 5 m, 변 방향으로 변 안쪽 (양 끝 1 m 뺌), 높이 (점 z − 동 base_z) 10–40 m (나무 · 차 · 1층 차양 위).
- 입면 자리 = 부호 거리 (밖 +) 의 최빈값 (0.25 m 칸, 1 m 창으로 다듬은 히스토그램 꼭대기). 퍼짐 = 최빈값 ± 2 m 안 점의 MAD.
- 점이 60 개 넘는 변만 센다.
- 벽 앞 땅: 그 변 밖 1–12 m 점 z 의 낮은 10 % 중앙값 (점구름 자체의 땅, SRTM 아님). 높이 = 입면 꼭대기 − 벽 앞 땅.
- 높이: 그 동의 모든 변 점 (|d−최빈값| ≤ 1.5 m) 의 높이 분포에서, 10–30 m 밀도 중앙값의 15 % 아래로 떨어지기 시작하는 가장 낮은 높이 = 보인 입면 꼭대기 [추정 방법].
결과: <접두>.json, <접두>.png (변별 입면 자리 지도 · 히스토그램, 형상만).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a = np.load(sys.argv[1]); vw = json.load(open(sys.argv[2])); sc = json.load(open(sys.argv[3])); pre = sys.argv[4]
title = sys.argv[sys.argv.index('--title') + 1] if '--title' in sys.argv else ''
P = a['P'].astype(np.float64)
_mx = int(sys.argv[sys.argv.index('--max-points') + 1]) if '--max-points' in sys.argv else 4_000_000
if len(P) > _mx:
    P = P[np.sort(np.random.default_rng(0).choice(len(P), _mx, replace=False))]
towers = {t['name']: t for t in sc['info']['towers']}
# 대장 높이가 있는 이웃 동 (DATA.md: 상도현대아파트 2003-06-16 사용승인, 101동 13층 37.3 m · 102동 15층 42.4 m) — 높이 측정 검증용
EXTRA = {'상도현대아파트|101동': ('S101', 13, 37.3), '상도현대아파트|102동': ('S102', 15, 42.4)}
bld = {k: {'rings': v['rings']} for k, v in vw['towers'].items() if k in towers}
for c in vw['context']:
    key = f"{c['name']}|{c['dc']}"
    if key in EXTRA:
        nm, fl, hreg = EXTRA[key]
        bld.setdefault(nm, {'rings': []})['rings'].append(c['xy'])
        towers[nm] = {'name': nm, 'base_z': None, 'height_m': hreg, 'floors': fl, 'registered': True}
segs = []
for k, v in bld.items():
    for r in v['rings']:
        r = np.asarray(r)
        area2 = np.dot(r[:-1, 0], r[1:, 1]) - np.dot(r[1:, 0], r[:-1, 1])
        for i in range(len(r) - 1):
            p0, p1 = r[i], r[i + 1]; L = np.linalg.norm(p1 - p0)
            if L < 0.5:
                continue
            u = (p1 - p0) / L
            nrm = np.array([u[1], -u[0]]) if area2 > 0 else np.array([-u[1], u[0]])   # 밖 방향 (고리 방향에 따라)
            segs.append({'tower': k, 'p0': p0, 'p1': p1, 'u': u, 'n': nrm, 'L': L})
res = {'segments': [], 'towers': {}}
xy = P[:, :2]
# 점마다 가장 가까운 변 (모든 동 · 모든 변, 짧은 꺾임 변 포함)
S0 = np.array([s['p0'] for s in segs]); S1 = np.array([s['p1'] for s in segs])
near_seg = np.full(len(P), -1); near_d = np.full(len(P), np.inf)
dd_ = S1 - S0; L2 = np.maximum((dd_ ** 2).sum(1), 1e-12)
for b in range(0, len(P), 2000):
    Q = xy[b:b + 2000]
    tt = np.clip(((Q[:, None, :] - S0[None]) * dd_[None]).sum(2) / L2[None], 0, 1)
    D = np.linalg.norm(S0[None] + tt[..., None] * dd_[None] - Q[:, None, :], axis=2)
    near_seg[b:b + 2000] = np.argmin(D, 1); near_d[b:b + 2000] = D.min(1)
MIN_L = float(sys.argv[sys.argv.index('--min-len') + 1]) if '--min-len' in sys.argv else 12.0
for si, sg in enumerate(segs):
    if sg['L'] < MIN_L:
        continue
    t = towers[sg['tower']]
    rel = xy - sg['p0']
    along = rel @ sg['u']; dist = rel @ sg['n']
    # 벽 앞 땅 (점구름 자체) — 먼저 구해 높이 원점으로 쓴다. 없으면 v5 base_z.
    gmk = (along > 1) & (along < sg['L'] - 1) & (dist >= 1) & (dist <= 12)
    zg = P[gmk, 2]
    ground_local = float(np.median(np.sort(zg)[:max(5, len(zg) // 10)])) if len(zg) >= 30 else None
    base = ground_local if ground_local is not None else t['base_z']
    if base is None:
        continue
    h = P[:, 2] - base
    mk = (near_seg == si) & (along > 1) & (along < sg['L'] - 1) & (np.abs(dist) <= 5) & (h >= 10) & (h <= 40)
    if mk.sum() < 60:
        continue
    dd = dist[mk]
    hist, edges = np.histogram(dd, bins=np.arange(-5, 5.25, 0.25))
    sm = np.convolve(hist, np.ones(4) / 4, 'same')
    mode = float(edges[np.argmax(sm)] + 0.125)
    near = np.abs(dd - mode) <= 2
    mad = float(np.median(np.abs(dd[near] - mode))) if near.any() else None
    sg_out = {'tower': sg['tower'], 'len_m': round(sg['L'], 1), 'n': int(mk.sum()), 'offset_m': round(mode, 2), 'mad_m': round(mad, 2) if mad is not None else None,
              'frac_within_1m_of_mode': round(float((np.abs(dd - mode) <= 1).mean()), 3), 'mid': ((sg['p0'] + sg['p1']) / 2).round(1).tolist(),
              'n_dir': sg['n'].round(3).tolist()}
    sg_out['ground_local_z'] = round(ground_local, 2) if ground_local is not None else None
    sg_out['ground_local_minus_base_m'] = round(ground_local - t['base_z'], 2) if (ground_local is not None and t['base_z'] is not None) else None
    sg_out['height_origin'] = 'local' if ground_local is not None else 'v5_base'
    sg['ground_local'] = ground_local
    # 높이용 점 (그 변 입면 둘레)
    mk2 = (near_seg == si) & (along > 1) & (along < sg['L'] - 1) & (np.abs(dist - mode) <= 1.5) & (h > 0) & (h < 80)
    sg['h'] = h[mk2]
    res['segments'].append(sg_out)
    sg['used'] = True
for k in sorted({s['tower'] for s in res['segments']}, key=lambda q: (q.startswith('S'), q)):
    hs = np.concatenate([sg['h'] for sg in segs if sg['tower'] == k and sg.get('used')])
    hist, edges = np.histogram(hs, bins=np.arange(0, 80.5, 0.5))
    band = hist[(edges[:-1] >= 10) & (edges[:-1] < 30)]
    ref = np.median(band) if len(band) else 0
    top = None
    if ref > 0:
        for i in range(len(hist)):
            if edges[i] >= 30 and hist[i:i + 4].max() < 0.15 * ref:
                top = float(edges[i]); break
    so = [s for s in res['segments'] if s['tower'] == k]
    gl = [s['ground_local_minus_base_m'] for s in so if s.get('ground_local_minus_base_m') is not None]
    loc = sum(1 for s in so if s['height_origin'] == 'local')
    hs_ok = hs[(hs > 0) & (hs < 80)]
    res['towers'][k] = {'segments': len(so), 'offset_median_m': round(float(np.median([s['offset_m'] for s in so])), 2),
                        'mad_median_m': round(float(np.median([s['mad_m'] for s in so])), 2),
                        'height_m': top, 'height_p995_m': round(float(np.percentile(hs_ok, 99.5)), 1) if len(hs_ok) > 100 else None,
                        'height_origin_local_segments': f'{loc}/{len(so)}',
                        'ground_local_minus_v5base_m': round(float(np.median(gl)), 2) if gl else None,
                        'ref_height_m': towers[k]['height_m'], 'ref_is_registered': bool(towers[k].get('registered')),
                        'floors': towers[k]['floors'], 'n_height_pts': int(len(hs))}
off = np.array([s['offset_m'] for s in res['segments'] if not s['tower'].startswith('S')]); mads = np.array([s['mad_m'] for s in res['segments'] if not s['tower'].startswith('S')])
if len(off):
    res['all'] = {'segments': int(len(off)), 'offset_median_m': round(float(np.median(off)), 2), 'offset_abs_median_m': round(float(np.median(np.abs(off))), 2),
                  'offset_within_1m': round(float((np.abs(off) <= 1).mean()), 3), 'offset_within_2m': round(float((np.abs(off) <= 2).mean()), 3),
                  'mad_median_m': round(float(np.median(mads)), 2)}
    hv = [(v['height_m'] - v['ref_height_m']) for v in res['towers'].values() if v['height_m'] is not None and not v['ref_is_registered']]
    if hv:
        res['all']['height_minus_v5_m'] = {'n': len(hv), 'median': round(float(np.median(hv)), 2), 'min': round(float(min(hv)), 2), 'max': round(float(max(hv)), 2)}
    hp = [(v['height_p995_m'] - v['ref_height_m']) for v in res['towers'].values() if v.get('height_p995_m') is not None and not v['ref_is_registered']]
    if hp:
        res['all']['height_p995_minus_v5_m'] = {'n': len(hp), 'median': round(float(np.median(hp)), 2), 'min': round(float(min(hp)), 2), 'max': round(float(max(hp)), 2)}
    hr = {k: round(v['height_m'] - v['ref_height_m'], 2) for k, v in res['towers'].items() if v['height_m'] is not None and v['ref_is_registered']}
    if hr:
        res['all']['height_minus_registered_m'] = hr
json.dump(res, open(pre + '.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res.get('all'), ensure_ascii=False))
for k, v in res['towers'].items():
    print(k, v)
# 그림: 변마다 입면 자리 화살 (색 = offset)
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1.25, 1]})
for k, v in bld.items():
    for r in v['rings']:
        r = np.asarray(r); ax[0].plot(r[:, 0], r[:, 1], color='k', lw=0.8)
    cxy = np.asarray(v['rings'][0]).mean(0); ax[0].text(cxy[0], cxy[1], k, fontsize=7, ha='center', va='center', color='0.3')
for c in vw['context']:
    r = np.asarray(c['xy']); ax[0].plot(r[:, 0], r[:, 1], color='0.85', lw=0.4)
cm = plt.get_cmap('coolwarm')
for s in res['segments']:
    m = np.array(s['mid']); n = np.array(s['n_dir'])
    col = cm(np.clip((s['offset_m'] + 3) / 6, 0, 1))
    ax[0].plot([m[0], m[0] + n[0] * s['offset_m'] * 3], [m[1], m[1] + n[1] * s['offset_m'] * 3], color=col, lw=2.5)
    ax[0].plot(m[0], m[1], 'o', color=col, ms=3)
C = a['C']; ax[0].plot(C[:, 0], C[:, 1], '.', color='g', ms=2, label='camera')
ax[0].set_aspect('equal'); ax[0].set_xlabel('east (m, v5)'); ax[0].set_ylabel('north (m)')
lo = C.min(0)[:2] - 60; hi = C.max(0)[:2] + 60; ax[0].set_xlim(lo[0], hi[0]); ax[0].set_ylim(lo[1], hi[1])
ax[0].set_title(f'{title}: facade offset per wall (line = offset x3, red = outside)', fontsize=9); ax[0].legend(loc='lower left', fontsize=8)
if len(off):
    ax[1].hist(off, bins=np.arange(-5, 5.25, 0.25), color='0.3')
    ax[1].axvline(0, color='r')
    ax[1].set_xlabel('facade plane offset from V-World outline (m, outside +), per wall >= 8 m')
    ax[1].set_title(f"walls {len(off)}: |offset| median {res['all']['offset_abs_median_m']} m, <=1 m {res['all']['offset_within_1m']*100:.0f} %, spread(MAD) {res['all']['mad_median_m']} m", fontsize=9)
plt.tight_layout(); plt.savefig(pre + '.png', dpi=110)
