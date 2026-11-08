#!/usr/bin/env python3
"""Creates clearly identified SAMPLE geometries if OSM fetch is unavailable."""
import json
from pathlib import Path
import random
random.seed(7)
root=Path(__file__).resolve().parents[1]
b=[];roads=[]
for i in range(14):
    x=-420+i*65
    roads.append({'id':i,'type':'residential','points':[[x,-510],[x,510]]})
for j in range(12):
    n=-400+j*78
    roads.append({'id':j+30,'type':'secondary','points':[[-510,n],[510,n]]})
count=0
for i in range(13):
    for j in range(11):
        x=-405+i*65;n=-389+j*78
        w=random.uniform(27,43);d=random.uniform(29,48)
        count+=1
        b.append({'id':count,'points':[[x,n],[x+w,n],[x+w,n+d],[x,n+d],[x,n]],'height':random.choice([10,16,22,28,44,65,82]),'height_source':'demo','tags':{'building':'yes'}})
obj={'meta':{'is_real_osm':False,'source':'FICTIONAL DEMO - NOT DAEGU','width_m':1000,'height_m':1000},'buildings':b,'roads':roads}
out=root/'site'/'data'/'daegu-1km.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
print('가상 건물 데모 생성:',len(b),'개 (실제 대구 데이터 아님)')
