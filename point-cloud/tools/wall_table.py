"""벽마다 세 근거의 입면 자리 (V-World 윤곽 기준) 를 한 표로. 사용: python wall_table.py <vggt.json> <colmap.json> <mesh.json> <out.csv>
- '윤곽 오차 후보' = COLMAP 과 메시가 1 m 안에서 같고, 둘 다 |차| > 1.5 m 인 벽."""
import csv, json, sys
import numpy as np
V, C, M = (json.load(open(f)) for f in sys.argv[1:4])
key = lambda s: (s['tower'], tuple(s['mid']))
iv = {key(s): s for s in V['segments']}; ic = {key(s): s for s in C['segments']}; im = {key(s): s for s in M['segments']}
rows = []
for k in sorted(set(iv) | set(ic) | set(im), key=lambda q: (q[0].startswith('S'), q[0], q[1])):
    if k[0].startswith('S'):
        continue
    g = lambda d: d[k]['offset_m'] if k in d else None
    v, c, m = g(iv), g(ic), g(im)
    L = (iv.get(k) or ic.get(k) or im.get(k))['len_m']
    cand = c is not None and m is not None and abs(c - m) <= 1 and abs(c) > 1.5 and abs(m) > 1.5
    rows.append({'동': k[0], '벽 가운데 x': k[1][0], '벽 가운데 y': k[1][1], '길이_m': L, 'VGGT_m': v, 'COLMAP_m': c, '메시_m': m, '윤곽오차후보': cand})
with open(sys.argv[4], 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
n3 = [r for r in rows if None not in (r['VGGT_m'], r['COLMAP_m'], r['메시_m'])]
print('벽', len(rows), '세 근거 모두', len(n3), '윤곽 오차 후보', sum(r['윤곽오차후보'] for r in rows))
for r in rows:
    if r['윤곽오차후보']:
        print(' ', r)
