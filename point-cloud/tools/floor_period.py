"""입면 점의 높이 분포 주기 (층높이) 를 자기상관으로 잰다. facade_check 와 같은 변 · 같은 벽 앞 땅 규칙.
사용: ~/apt-recon/venv/bin/python -I floor_period.py <points.npz> <facade.json> <out.json> [--band 6,36]
- 변마다: 입면 자리 (facade.json offset) ± 0.8 m 안, 벽 앞 땅 위 band 높이 점의 z 히스토그램 (5 cm 칸) → 1.5 m 이동 평균 빼기 → 자기상관 → 2.2–3.6 m 에서 가장 높은 봉우리.
- 동마다: 변 값의 점 수 무게 중앙값. 봉우리 세기 (자기상관 값) 도 적는다 (0.1 아래는 믿지 않음).
"""
import json, sys
import numpy as np

a = np.load(sys.argv[1]); fc = json.load(open(sys.argv[2])); outp = sys.argv[3]
band = tuple(map(float, sys.argv[sys.argv.index('--band') + 1].split(','))) if '--band' in sys.argv else (6.0, 36.0)
P = a['P'].astype(np.float64)
_mx = int(sys.argv[sys.argv.index('--max-points') + 1]) if '--max-points' in sys.argv else 4_000_000
if len(P) > _mx:
    P = P[np.sort(np.random.default_rng(0).choice(len(P), _mx, replace=False))]
res = {'segments': [], 'towers': {}}
for s in fc['segments']:
    if s.get('ground_local_z') is None:
        continue
    m = np.array(s['mid']); n = np.array(s['n_dir']); u = np.array([-n[1], n[0]])
    rel = P[:, :2] - m
    along = rel @ u; dist = rel @ n
    h = P[:, 2] - s['ground_local_z']
    mk = (np.abs(along) < s['len_m'] / 2 - 1) & (np.abs(dist - s['offset_m']) <= 0.8) & (h >= band[0]) & (h <= band[1])
    if mk.sum() < 2000:
        continue
    hist, edges = np.histogram(h[mk], bins=np.arange(band[0], band[1] + 0.05, 0.05))
    hist = hist.astype(float)
    k = int(1.5 / 0.05); trend = np.convolve(hist, np.ones(k) / k, 'same')
    x = hist - trend; x -= x.mean()
    ac = np.correlate(x, x, 'full')[len(x) - 1:]; ac /= ac[0] if ac[0] > 0 else 1
    lags = np.arange(len(ac)) * 0.05
    w = (lags >= 2.2) & (lags <= 3.6)
    i = np.argmax(np.where(w, ac, -9))
    # 봉우리 꼭대기 다듬기 (포물선)
    if 0 < i < len(ac) - 1:
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]; dl = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0
    else:
        dl = 0
    per = float((i + dl) * 0.05)
    res['segments'].append({'tower': s['tower'], 'n': int(mk.sum()), 'period_m': round(per, 3), 'peak_ac': round(float(ac[i]), 3)})
for t in sorted({q['tower'] for q in res['segments']}, key=lambda q: (q.startswith('S'), q)):
    q = [r for r in res['segments'] if r['tower'] == t and r['peak_ac'] >= 0.1]
    if not q:
        continue
    ps = np.repeat([r['period_m'] for r in q], [max(1, r['n'] // 1000) for r in q])
    res['towers'][t] = {'segments': len(q), 'period_m': round(float(np.median(ps)), 3), 'peak_ac_max': max(r['peak_ac'] for r in q)}
json.dump(res, open(outp, 'w'), ensure_ascii=False, indent=1)
for t, v in res['towers'].items():
    print(t, v)
print('변', len(res['segments']))
