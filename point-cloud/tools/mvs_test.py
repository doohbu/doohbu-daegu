"""시험 구역 밀집 복원 (COLMAP PatchMatch, CUDA). 자세 = 전체 SfM 모델 0 (정밀, 재투영 ~1 px). 사진은 링크만 (복사 안 함).
사용: venv-mvs/bin/python -I mvs_test.py <ids 파일> <out 폴더> [--max-size 1600]
- 대상: ids 의 파노라마 옆면 4장 (front · right · back · left). top 은 하늘이 많아 뺀다.
- 결과: <out>/fused.ply (COLMAP 좌표), 시간 · 점 수는 <out>/mvs.json.
"""
import json, os, sys, time
import pycolmap
ids = open(sys.argv[1]).read().strip().split(','); out = os.path.abspath(sys.argv[2])
mx = int(sys.argv[sys.argv.index('--max-size') + 1]) if '--max-size' in sys.argv else 1600
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, '..', 'work', 'sfm_full')
os.makedirs(out, exist_ok=True)
rec = pycolmap.Reconstruction(os.path.join(W, 'sparse', '0'))
names = [im.name for im in rec.images.values() if im.has_pose and im.name.split('/')[1][:-4] in ids and im.name.split('/')[0] in ('front', 'right', 'back', 'left')]
print('영상', len(names), flush=True)
t0 = time.time()
pycolmap.undistort_images(out, os.path.join(W, 'sparse', '0'), os.path.join(W, 'images'), image_names=names,
                          copy_policy=pycolmap.FileCopyType.softlink, num_patch_match_src_images=20)
t1 = time.time(); print(f'작업 폴더 {t1 - t0:.0f}초', flush=True)
po = pycolmap.PatchMatchOptions(); po.max_image_size = mx; po.gpu_index = '0'
po.geom_consistency = True; po.num_iterations = 5; po.filter = True
pycolmap.patch_match_stereo(out, options=po)
t2 = time.time(); print(f'PatchMatch {t2 - t1:.0f}초', flush=True)
fo = pycolmap.StereoFusionOptions(); fo.max_image_size = mx; fo.min_num_pixels = 3
fused = pycolmap.stereo_fusion(os.path.join(out, 'fused.ply'), out, options=fo, output_type='PLY')
t3 = time.time()
json.dump({'images': len(names), 'max_image_size': mx, 'seconds': {'workspace': round(t1 - t0), 'patch_match': round(t2 - t1), 'fusion': round(t3 - t2)},
           'points': fused.num_points3D() if hasattr(fused, 'num_points3D') else None}, open(os.path.join(out, 'mvs.json'), 'w'), indent=1)
print(f'합치기 {t3 - t2:.0f}초, 끝', flush=True)
