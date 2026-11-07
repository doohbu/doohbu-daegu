#!/bin/bash
# 같은 33곳 (work/proto_ids.txt) 으로 VGGT 입력 구성 4가지를 다시 돌려 견준다.
cd ~/lightlab/runs/streetview-recon
P=~/lightlab/runs/streetview/naver; IDS=$(cat work/proto_ids.txt)
V8=0:0,45:0,90:0,135:0,180:0,225:0,270:0,315:0
run() { venv/bin/python -I tools/vggt_chunks.py --panos $P/panos --index $P/index.csv --ids $IDS --bf16-weights "$@" 2>&1 | grep -E "파노라마" ; }
run --out work/var_cube5 --faces front,right,back,left,top --k 8 --seed-spacing 30
run --out work/var_cube4 --faces front,right,back,left --k 10 --seed-spacing 30
run --out work/var_view8 --views $V8 --k 5 --seed-spacing 20
run --out work/var_view12 --views $V8,0:45,90:45,180:45,270:45 --k 4 --seed-spacing 15
for v in cube5 cube4 view8 view12; do ~/apt-recon/venv/bin/python -I tools/vggt_align.py work/var_$v work/v5_scene.npz > work/var_$v/align.out 2>&1; done
~/apt-recon/venv/bin/python tools/summ_align.py work/var_cube5/align_raw.json work/var_cube4/align_raw.json work/var_view8/align_raw.json work/var_view12/align_raw.json
