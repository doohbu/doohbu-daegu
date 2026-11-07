"""이미 맞춘 database 로 지도 만들기만 다시 (사전값 · 설정 시험용).
사용: ~/apt-recon/venv/bin/python -I remap.py <sfm 폴더> <out 이름> [--prior] [--global] [--threads 8]"""
import os, sys, json, time
import pycolmap
sfm = sys.argv[1]; name = sys.argv[2]
db = os.path.join(sfm, 'database.db'); img = os.path.join(sfm, 'images')
out = os.path.join(sfm, name); os.makedirs(out, exist_ok=True)
th = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 8
t = time.time()
if '--global' in sys.argv:
    go = pycolmap.GlobalPipelineOptions(); go.num_threads = th
    recs = pycolmap.global_mapping(db, img, out, options=go)
else:
    io = pycolmap.IncrementalPipelineOptions(); io.num_threads = th
    io.ba_refine_focal_length = False; io.ba_refine_principal_point = False; io.ba_refine_extra_params = False
    io.ba_refine_sensor_from_rig = False
    io.mapper.abs_pose_refine_focal_length = False; io.mapper.abs_pose_refine_extra_params = False
    io.multiple_models = True; io.min_model_size = 3
    if '--prior' in sys.argv:
        io.use_prior_position = True; io.use_robust_loss_on_prior_position = True
    recs = pycolmap.incremental_mapping(db, img, out, options=io)
res = []
for k, r in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
    res.append({'id': int(k), 'reg_images': r.num_reg_images(), 'reg_frames': r.num_reg_frames(), 'points3D': r.num_points3D(),
                'mean_reproj_px': round(r.compute_mean_reprojection_error(), 3)})
print(json.dumps({'seconds': round(time.time() - t, 1), 'models': res}))
