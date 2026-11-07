"""두 facade_check 결과의 같은 벽 (동 + 변 가운데 자리) 의 입면 자리를 견준다.
사용: python compare_walls.py <A.json> <B.json> [이름A 이름B]"""
import json, sys
import numpy as np
A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ('A', 'B')
ib = {(s['tower'], tuple(s['mid'])): s for s in B['segments']}
pairs = []
for s in A['segments']:
    k = (s['tower'], tuple(s['mid']))
    if k in ib and not s['tower'].startswith('S'):
        pairs.append((s['offset_m'], ib[k]['offset_m'], s['tower']))
a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
d = a - b
same = np.sign(a) == np.sign(b)
big = (np.abs(a) > 1.5) | (np.abs(b) > 1.5)
print(json.dumps({'common_walls': len(pairs), f'{na}_minus_{nb}_median_m': round(float(np.median(d)), 2),
                  'abs_diff_median_m': round(float(np.median(np.abs(d))), 2), 'within_1m': round(float((np.abs(d) <= 1).mean()), 3),
                  'corr': round(float(np.corrcoef(a, b)[0, 1]), 3) if len(a) > 2 else None,
                  'big_offset_walls': int(big.sum()), 'big_and_agree_within_1m': int((big & (np.abs(d) <= 1)).sum())}, ensure_ascii=False))
for (x, y, t) in sorted(pairs, key=lambda p: -abs(p[0])):
    if abs(x) > 1.5 or abs(y) > 1.5:
        print(f'  {t}: {na} {x:+.2f}  {nb} {y:+.2f}')
