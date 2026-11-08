#!/usr/bin/env python3
"""대구역 중심 1km × 1km OpenStreetMap buildings/roads -> local JSON.
Fetch is only run intentionally in Codespaces; the browser never calls Overpass.
"""
import json
import math
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site' / 'data' / 'daegu-1km.json'
LAT, LON = 35.8759306, 128.5960528  # Daegu Station (Korail), WGS84
R = 6378137.0
HALF = 500.0
S = LAT - math.degrees(HALF/R)
N = LAT + math.degrees(HALF/R)
W = LON - math.degrees(HALF/(R*math.cos(math.radians(LAT))))
E = LON + math.degrees(HALF/(R*math.cos(math.radians(LAT))))
BBOX = f'{S:.7f},{W:.7f},{N:.7f},{E:.7f}'
QUERY = f'''[out:json][timeout:75];
(
 way["building"]({BBOX});
 way["highway"]({BBOX});
);
out body geom;
'''
ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.nchc.org.tw/api/interpreter',
]

def fetch_osm():
    errors = []
    payload = urlencode({'data': QUERY}).encode('ascii')
    for url in ENDPOINTS:
        print('OSM 요청:', url, flush=True)
        try:
            req = Request(url, data=payload, headers={'User-Agent':'HanaLightSimulator/1.0 (local educational 1km test)', 'Content-Type':'application/x-www-form-urlencoded'})
            with urlopen(req, timeout=100) as r:
                raw = r.read(32*1024*1024 + 1)
                if len(raw)>32*1024*1024:raise RuntimeError('32MB 제한을 초과했습니다.')
            data = json.loads(raw)
            if not isinstance(data.get('elements'),list):raise RuntimeError('OSM 응답에 elements가 없습니다.')
            return data, url
        except Exception as exc:
            errors.append(f'{url}: {exc}')
    raise RuntimeError('모든 OSM 서버 호출 실패.\n'+'\n'.join(errors))

def local_xy(lat, lon):
    east=R*math.radians(lon-LON)*math.cos(math.radians(LAT))
    north=R*math.radians(lat-LAT)
    return [round(east,3),round(north,3)]

def estimate_height(tags):
    raw=str(tags.get('height','')).strip().lower().replace(',','.')
    found=re.search(r'([-+]?\d+(?:\.\d+)?)\s*(ft|feet|m|meters?)?',raw)
    if found:
        height=float(found.group(1))*(0.3048 if found.group(2) in ('ft','feet') else 1)
        if 2<=height<=300:return round(height,2),'osm:height'
    raw=str(tags.get('building:levels','')).strip().replace(',','.')
    found=re.search(r'\d+(?:\.\d+)?',raw)
    if found:
        levels=float(found.group())
        if 1<=levels<=100:return round(levels*3.1,2),'estimated:levels'
    t=tags.get('building','')
    return (6.2 if t in ('house','detached','garage','garages') else 12.4),'estimated:default'

def transform(data, endpoint='fixture'):
    buildings,roads=[],[]
    for item in data.get('elements',[]):
        if item.get('type')!='way':continue
        geo=item.get('geometry') or []
        if not (3<=len(geo)<=1000):continue
        try:pts=[local_xy(float(p['lat']),float(p['lon'])) for p in geo]
        except (KeyError, ValueError, TypeError):continue
        tags=item.get('tags') or {}
        if 'building' in tags and len(pts)>=4:
            height,hs=estimate_height(tags)
            # Only simple OSM way outlines, not relation multipolygons/holes
            buildings.append({'id':item['id'],'points':pts,'height':height,'height_source':hs,'tags':{'building':tags.get('building','yes')},'name':tags.get('name','')})
        if 'highway' in tags and len(pts)>=2:
            roads.append({'id':item['id'],'type':tags.get('highway','road'),'points':pts})
    return {
        'meta': {'center_lat':LAT,'center_lon':LON,'bbox':[S,W,N,E],'width_m':1000,'height_m':1000,
                 'source':'OpenStreetMap Overpass / '+endpoint,'is_real_osm':True,
                 'fetched_at':datetime.now(timezone.utc).isoformat(),
                 'attribution':'© OpenStreetMap contributors (ODbL)',
                 'notes':'OSM footprints; height may be estimated; no exact terrain elevations or facade imagery'},
        'buildings':buildings,'roads':roads,
    }

def main():
    raw,endpoint=fetch_osm()
    result=transform(raw,endpoint)
    if len(result['buildings'])<10:
        raise RuntimeError(f"건물 {len(result['buildings'])}개만 반환되어 파일을 변경하지 않았습니다.")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    temp=OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    os.replace(temp,OUT)
    print(f"완료: {OUT} / 건물 {len(result['buildings'])}개 / 도로 {len(result['roads'])}개")
    print('참고: 층수 없는 건물은 높이 추정치이며, 실제 지형 고도는 V1에 미구현입니다.')

if __name__=='__main__':main()
