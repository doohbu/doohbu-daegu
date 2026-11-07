"""맞춘 SfM 점 ↔ V-World 동 윤곽 (수평 거리) · v5 높이 대조, 그림 (형상만: 점 · 윤곽 · 카메라 자리).

사용: ~/apt-recon/venv/bin/python -I outline_check.py <aligned.npz> <vworld_outlines.json> <v5 scene.json> <out 접두> [--title 글]
- 벽 점: 가장 가까운 단지 동 윤곽 변까지 수평 거리 ≤ 6 m, 그 동 바닥 (scene base_z) 보다 3 m 넘게 위, 트랙 길이 ≥ 3.
- 부호: 윤곽 밖 +, 안 −.
- 높이: 벽 점 z − 그 동 base_z 의 99 분위와 최댓값 (사진에 보인 가장 높은 곳). v5 높이 = height_m (층수 × 2.9 m + 파라펫 1.5 m 규칙 [규칙]).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

a = np.load(sys.argv[1]); vw = json.load(open(sys.argv[2])); sc = json.load(open(sys.argv[3])); pre = sys.argv[4]
title = sys.argv[sys.argv.index('--title') + 1] if '--title' in sys.argv else ''
maxn = int(sys.argv[sys.argv.index('--max-points') + 1]) if '--max-points' in sys.argv else 400000
P = a['P'].astype(np.float64); tl = a['track']; C = a['C']
if len(P) > maxn:                      # 고르게 솎기 (재현 가능하게 고정 씨앗)
    ix = np.random.default_rng(0).choice(len(P), maxn, replace=False); ix.sort()
    P = P[ix]; tl = tl[ix]
towers = {t['name']: t for t in sc['info']['towers']}
segs = []; owner = []
paths = {}
for k, v in vw['towers'].items():
    if k not in towers:
        continue
    paths[k] = [MPath(np.asarray(r)) for r in v['rings']]
    for r in v['rings']:
        r = np.asarray(r)
        for i in range(len(r) - 1):
            segs.append((r[i], r[i + 1])); owner.append(k)
S0 = np.array([s[0] for s in segs]); S1 = np.array([s[1] for s in segs]); owner = np.array(owner)


def seg_dist_block(Q):
    d = S1 - S0; L2 = np.maximum((d ** 2).sum(1), 1e-12)
    t = np.clip(((Q[:, None, :] - S0[None]) * d[None]).sum(2) / L2[None], 0, 1)
    q = S0[None] + t[..., None] * d[None]
    dd = np.linalg.norm(q - Q[:, None, :], axis=2)
    i = np.argmin(dd, 1)
    return dd[np.arange(len(Q)), i], i


res = {'towers': {}}
dist = np.full(len(P), np.inf); seg_i = np.full(len(P), -1)
for b in range(0, len(P), 2000):
    dd, ii = seg_dist_block(P[b:b + 2000, :2])
    dist[b:b + 2000] = dd; seg_i[b:b + 2000] = ii
own = owner[seg_i]
inside = np.zeros(len(P), bool)
for k, ps in paths.items():
    mk = own == k
    if mk.any():
        inside[mk] = np.any([pp.contains_points(P[mk, :2]) for pp in ps], axis=0)
sd = np.where(inside, -dist, dist)
base = np.array([towers[o]['base_z'] if o in towers else np.nan for o in own])
hz = P[:, 2] - base
wall = (dist <= 6.0) & (hz > 3.0) & (tl >= 3)
allw = []
for k in sorted(set(own[wall]), key=int):
    mk = wall & (own == k)
    if mk.sum() < 20:
        continue
    t = towers[k]; s = sd[mk]; h = hz[mk]
    # 변마다 (점 30 개 이상) 부호 거리 중앙값
    per_seg = []
    for i in sorted(set(seg_i[mk])):
        m2 = mk & (seg_i == i)
        if m2.sum() >= 30:
            L = float(np.linalg.norm(S1[i] - S0[i]))
            per_seg.append({'seg': int(i), 'len_m': round(L, 1), 'n': int(m2.sum()), 'signed_median_m': round(float(np.median(sd[m2])), 2),
                            'abs_median_m': round(float(np.median(np.abs(sd[m2]))), 2)})
    res['towers'][k] = {'n_wall_pts': int(mk.sum()), 'signed_median_m': round(float(np.median(s)), 2),
                        'abs_median_m': round(float(np.median(np.abs(s))), 2), 'within_1m': round(float((np.abs(s) < 1).mean()), 3),
                        'height_obs_p99_m': round(float(np.percentile(h, 99)), 1), 'height_obs_max_m': round(float(h.max()), 1),
                        'height_v5_m': t['height_m'], 'floors': t['floors'], 'per_seg': per_seg}
    allw.append(s)
if allw:
    s = np.concatenate(allw)
    res['all_wall'] = {'n': int(len(s)), 'signed_median_m': round(float(np.median(s)), 2), 'abs_median_m': round(float(np.median(np.abs(s))), 2),
                       'within_0.5m': round(float((np.abs(s) < 0.5).mean()), 3), 'within_1m': round(float((np.abs(s) < 1).mean()), 3),
                       'within_2m': round(float((np.abs(s) < 2).mean()), 3)}
json.dump(res, open(pre + '.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'towers'}, ensure_ascii=False))
for k, v in res['towers'].items():
    print(k, {kk: vv for kk, vv in v.items() if kk != 'per_seg'})
# 그림: 위에서 본 점 (높이 색) + 윤곽 + 카메라
lo = np.percentile(np.vstack([P[:, :2], C[:, :2]]), 1, axis=0) - 30; hi = np.percentile(np.vstack([P[:, :2], C[:, :2]]), 99, axis=0) + 30
fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1.15, 1]})
for c in vw['context']:
    r = np.asarray(c['xy'])
    if ((r >= lo) & (r <= hi)).all(1).any():
        ax[0].plot(r[:, 0], r[:, 1], color='0.75', lw=0.6)
for k, v in vw['towers'].items():
    if k not in towers:
        continue
    for r in v['rings']:
        r = np.asarray(r); ax[0].plot(r[:, 0], r[:, 1], color='k', lw=1.2)
    cxy = np.asarray(v['rings'][0]).mean(0)
    if (cxy >= lo).all() and (cxy <= hi).all():
        ax[0].text(cxy[0], cxy[1], k, fontsize=8, ha='center', va='center', color='0.2')
ok = tl >= 3
zz = P[ok, 2]
im = ax[0].scatter(P[ok, 0], P[ok, 1], c=zz, s=0.6, cmap='viridis', vmin=np.percentile(zz, 2), vmax=np.percentile(zz, 98))
ax[0].plot(C[:, 0], C[:, 1], 'r^', ms=5, label='camera (SfM)')
ax[0].set_xlim(lo[0], hi[0]); ax[0].set_ylim(lo[1], hi[1]); ax[0].set_aspect('equal')
ax[0].set_xlabel('east (m, v5 local)'); ax[0].set_ylabel('north (m)'); ax[0].legend(loc='lower left', fontsize=8)
plt.colorbar(im, ax=ax[0], fraction=0.035, label='z (m, v5)')
ax[0].set_title(title + '  points (track>=3) over V-World outlines', fontsize=9)
if allw:
    ax[1].hist(np.clip(s, -6, 6), bins=96, color='0.3')
    ax[1].axvline(0, color='r', lw=1)
    ax[1].set_xlabel('signed horizontal distance to V-World outline (m, outside +)'); ax[1].set_ylabel('wall points')
    ax[1].set_title(f"wall points n={res['all_wall']['n']}: median {res['all_wall']['signed_median_m']} m, |d|<1 m {res['all_wall']['within_1m']*100:.0f} %", fontsize=9)
plt.tight_layout(); plt.savefig(pre + '.png', dpi=110); print('그림', pre + '.png')
