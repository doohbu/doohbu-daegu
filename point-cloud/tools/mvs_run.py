"""고친 작업 폴더에서 PatchMatch → 합치기만. 사용: venv-mvs/bin/python -I mvs_run.py <작업 폴더> [--max-size 1600]"""
import json, os, sys, time
import pycolmap
ws = os.path.abspath(sys.argv[1]); mx = int(sys.argv[sys.argv.index('--max-size') + 1]) if '--max-size' in sys.argv else 1600
t1 = time.time()
po = pycolmap.PatchMatchOptions(); po.max_image_size = mx; po.gpu_index = '0'; po.geom_consistency = True; po.num_iterations = 5; po.filter = True
pycolmap.patch_match_stereo(ws, options=po)
t2 = time.time(); print(f'PatchMatch {t2 - t1:.0f}초', flush=True)
fo = pycolmap.StereoFusionOptions(); fo.max_image_size = mx; fo.min_num_pixels = 3
fused = pycolmap.stereo_fusion(os.path.join(ws, 'fused.ply'), ws, options=fo, output_type='PLY')
t3 = time.time()
json.dump({'max_image_size': mx, 'seconds': {'patch_match': round(t2 - t1), 'fusion': round(t3 - t2)},
           'points': fused.num_points3D()}, open(os.path.join(ws, 'mvs.json'), 'w'), indent=1)
print(f'합치기 {t3 - t2:.0f}초, 점 {fused.num_points3D()}', flush=True)
