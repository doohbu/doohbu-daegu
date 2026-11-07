#!/bin/bash
# 전체 SfM 이 끝난 뒤 대조 (6절). 사용: bash after_sfm.sh
set -e
cd ~/lightlab/runs/streetview-recon
PY="$HOME/apt-recon/venv/bin/python -I"; S=~/lightlab/runs/apt-recon/cheongnim-hyundai/v5/work/scene.json; N=~/lightlab/runs/streetview/naver
for m in $(ls work/sfm_full/sparse); do OMP_NUM_THREADS=4 $PY tools/eval_sfm.py work/sfm_full work/v5_scene.npz --model $m > work/sfm_full/eval_$m.out 2>&1 || echo "eval $m 실패"; done
$PY tools/merge_sfm.py work/sfm_full work/sfm_full/merged.npz --min-frames 5 --max-gps 3.0
$PY tools/facade_check.py work/sfm_full/merged.npz work/vworld_outlines.json $S figs/full_sfm_facade --title "COLMAP 전체" 2>/dev/null | head -1
BIG=$(ls work/sfm_full/sparse | head -1)
for m in $(ls work/sfm_full/sparse); do echo "$m $(python3 -c "import json;print(json.load(open('work/sfm_full/eval_$m.json'))['frames_registered_front'])" 2>/dev/null)"; done | sort -k2 -n -r | head -3
# 가장 큰 조각 (틀 수) 으로 자세 대조 · 재투영 · 메시 축척
BIG=$(for m in $(ls work/sfm_full/sparse); do echo "$m $($PY -c "import json;print(json.load(open('work/sfm_full/eval_$m.json'))['frames_registered_front'])")"; done | sort -k2 -n -r | head -1 | cut -d' ' -f1)
echo "가장 큰 조각: $BIG"
$PY tools/reproj_check.py work/sfm_full/sparse/$BIG work/vggt_full/views_world.npz work/sfm_full/reproj_vggt_vs_colmap.json --max-points 60000
$PY tools/pose_compare.py work/vggt_full/views_world.npz work/sfm_full/aligned_$BIG.npz work/sfm_full/pose_vggt_vs_colmap.json
OMP_NUM_THREADS=4 $PY tools/range_ratio_sfm.py $N/panos work/sfm_full $BIG work/sfm_full/range_ratio_colmap.json
$PY tools/naver_mesh_world.py $N/panos work/sfm_full/aligned_$BIG.npz work/naver_mesh_world_sfmpose.npz
$PY tools/facade_check.py work/naver_mesh_world_sfmpose.npz work/vworld_outlines.json $S figs/naver_mesh_sfmpose_facade --title "네이버 메시 (COLMAP 자세)" 2>/dev/null | head -1
$PY tools/compare_walls.py figs/full_vggt_facade.json figs/full_sfm_facade.json VGGT COLMAP
$PY tools/compare_walls.py figs/full_sfm_facade.json figs/naver_mesh_sfmpose_facade.json COLMAP 메시
