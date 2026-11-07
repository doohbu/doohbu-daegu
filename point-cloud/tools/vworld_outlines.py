"""v5 빌더와 같은 규칙으로 V-World 동 윤곽을 캐시 파일에서 읽는다 (네트워크 안 씀, 원본 읽기만).
- 단지 26동: LT_C_SPBD 중 건물명 == '현대아파트' 이고 동 이름이 '…동' (201동 제외). spec-v5 outline_from 10동은 outlines-v2.geojson (dt_d198).
- 주변 건물: 나머지 LT_C_SPBD (층수 gro_flo_co).
사용: ~/apt-recon/venv/bin/python -I vworld_outlines.py <cheongnim-hyundai 폴더> <out.json>
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cube

D = sys.argv[1]; OUT = sys.argv[2]
OUTLINE_FROM = {"102", "105", "106", "107", "110", "111", "112", "115", "116", "125"}
EXCLUDE = {"현대아파트|201동"}


def feats(d):
    if isinstance(d, dict):
        if 'features' in d:
            return d['features']
        for v in d.values():
            r = feats(v)
            if r:
                return r
    return None


def rings(geom):
    t = geom['type']; c = geom['coordinates']
    polys = c if t == 'MultiPolygon' else [c]
    return [p[0] for p in polys]          # 바깥 고리만


def dong(s):
    m = re.match(r'^\s*(\d+)\s*동', s or '')
    return m.group(1) if m else None


cache = os.path.join(D, 'src/runs/_cache/vworld')
spbd = None
for f in os.listdir(cache):
    if not f.endswith('.json'):
        continue
    fs = feats(json.load(open(os.path.join(cache, f))))
    if fs and 'buld_nm_dc' in fs[0].get('properties', {}) and len(fs) >= 500:
        spbd = (f, fs)
assert spbd, 'SPBD 캐시 없음'
towers, context = {}, []
for ft in spbd[1]:
    p = ft['properties']; nm = p.get('buld_nm') or ''; dc = p.get('buld_nm_dc') or ''
    for r in rings(ft['geometry']):
        x, y = cube.local_xy(np.array([q[0] for q in r]), np.array([q[1] for q in r]))
        xy = np.stack([x, y], 1).round(3).tolist()
        if nm == '현대아파트' and dc.endswith('동') and f'{nm}|{dc}' not in EXCLUDE and dong(dc):
            towers.setdefault(dong(dc), {'spbd': [], 'floors_spbd': p.get('gro_flo_co')})['spbd'].append(xy)
        else:
            context.append({'name': nm, 'dc': dc, 'floors': p.get('gro_flo_co'), 'xy': xy})
dt = json.load(open(os.path.join(D, 'outlines-v2.geojson')))
for ft in dt['features']:
    k = dong(ft['properties'].get('buld_dong_nm'))
    if k in OUTLINE_FROM:
        for r in rings(ft['geometry']):
            x, y = cube.local_xy(np.array([q[0] for q in r]), np.array([q[1] for q in r]))
            towers.setdefault(k, {'spbd': []}).setdefault('dt', []).append(np.stack([x, y], 1).round(3).tolist())


def area(xy):
    p = np.asarray(xy); return abs(0.5 * (np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


out = {'source': {'spbd_cache': spbd[0], 'dt_d198': 'outlines-v2.geojson', 'outline_from': sorted(OUTLINE_FROM)}, 'towers': {}, 'context': context}
for k, v in sorted(towers.items(), key=lambda kv: int(kv[0])):
    use = 'dt' if k in OUTLINE_FROM and v.get('dt') else 'spbd'
    rr = v[use]
    out['towers'][k] = {'used': 'dt_d198' if use == 'dt' else 'LT_C_SPBD', 'rings': rr, 'area_m2': round(sum(area(r) for r in rr), 1)}
json.dump(out, open(OUT, 'w'), ensure_ascii=False)
print(len(out['towers']), '동', len(context), '주변')
for k, v in out['towers'].items():
    print(k, v['used'], v['area_m2'], len(v['rings']))
