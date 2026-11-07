"""네이버 거리뷰 큐브맵 → COLMAP SfM (pycolmap 4.2.1, CPU). 한 파노라마 = rig 한 틀 (면 5장, 중심 같음, 상대 회전 고정).

사용: ~/apt-recon/venv/bin/python -I sfm_rig.py --panos <panos 폴더> --index <index.csv> --out <폴더>
        [--bbox x0,y0,x1,y1] [--ids a,b,...] [--faces front,right,back,left,top] [--pair-radius 40]
        [--max-size 2048] [--features 8192] [--threads 12] [--prior-xy 3] [--prior-z 20] [--mapper incremental|global]
- 면: PINHOLE f = W/2, 주점 = 가운데 (face_check.py 로 90° 확인), 내부값은 고정 (BA 에서 안 고침).
- 자세 사전값: 위경도 → v5 지역 xy, z = 땅 높이 모름 → 0 에 큰 분산 (--prior-z). 네이버 고도 값 (40.0) 은 쓰지 않는다.
- 짝: 파노라마 중심 거리 ≤ --pair-radius 인 두 파노라마의 모든 면 짝 (같은 틀 안 짝은 뺀다).
- 결과: <out>/sparse/<n>/ (COLMAP 모델), <out>/stats.json. 사진은 <out>/images/ 에 원본 심볼릭 링크로만 둔다.
"""
import argparse, csv, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pycolmap
import cube


def rows_from_index(index):
    return [r for r in csv.DictReader(open(index, encoding='utf-8')) if r.get('폴더')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panos', required=True); ap.add_argument('--index', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bbox'); ap.add_argument('--ids')
    ap.add_argument('--faces', default='front,right,back,left,top')
    ap.add_argument('--pair-radius', type=float, default=40.0)
    ap.add_argument('--max-size', type=int, default=2048); ap.add_argument('--features', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--prior-xy', type=float, default=3.0); ap.add_argument('--prior-z', type=float, default=20.0)
    ap.add_argument('--no-prior', action='store_true')
    ap.add_argument('--mapper', default='incremental')
    ap.add_argument('--feature-type', default='SIFT')
    ap.add_argument('--prune-angle', type=float, default=0.0, help='0 = 모든 면 짝. 아니면 옆면 짝은 네이버 방향으로 본 광축 차가 이 각도 이하만, top 은 --top-radius 안만')
    ap.add_argument('--top-radius', type=float, default=20.0)
    a = ap.parse_args()
    faces = a.faces.split(',')
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    rows = rows_from_index(a.index)
    seen = set(); sel = []
    for r in rows:
        pid = r['pano_id']
        if pid in seen:
            continue
        x, y = cube.local_xy(float(r['경도']), float(r['위도'])); x, y = float(x), float(y)
        if a.ids and pid not in a.ids.split(','):
            continue
        if a.bbox:
            x0, y0, x1, y1 = map(float, a.bbox.split(','))
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
        if not all(os.path.exists(os.path.join(a.panos, pid, f + '.jpg')) for f in faces):
            continue
        seen.add(pid)
        sel.append({'pid': pid, 'x': x, 'y': y, 'heading': float(r['방향_deg']), 'date': r['촬영일시_네이버값'],
                    'type': r['종류'], 'is3d': r['3D']})
    print(f'파노라마 {len(sel)}개, 면 {len(faces)}장씩', flush=True)
    img_dir = os.path.join(out, 'images')
    for f in faces:
        os.makedirs(os.path.join(img_dir, f), exist_ok=True)
        for s in sel:
            dst = os.path.join(img_dir, f, s['pid'] + '.jpg')
            if not os.path.lexists(dst):
                os.symlink(os.path.join(os.path.abspath(a.panos), s['pid'], f + '.jpg'), dst)
    names = [f'{f}/{s["pid"]}.jpg' for s in sel for f in faces]
    db_path = os.path.join(out, 'database.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    t0 = time.time()
    W = a.max_size
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = 'PINHOLE'
    ro.camera_params = f'{W/2},{W/2},{W/2},{W/2}'
    eo = pycolmap.FeatureExtractionOptions()
    eo.max_image_size = W
    eo.num_threads = a.threads
    eo.use_gpu = False
    if a.feature_type == 'SIFT':
        eo.sift.max_num_features = a.features
    else:
        eo.type = getattr(pycolmap.FeatureExtractorType, a.feature_type)
    pycolmap.extract_features(db_path, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=ro, extraction_options=eo)
    t_ext = time.time() - t0
    print(f'특징 {t_ext:.0f}초', flush=True)
    # rig: front = 기준 센서. cam_from_rig = R_face · R_front^T (회전만)
    Rf = cube.face_R('front', 0.0)
    cams = []
    for f in faces:
        c = pycolmap.RigConfigCamera()
        c.image_prefix = f + '/'
        if f == 'front':
            c.ref_sensor = True
        else:
            R = cube.face_R(f, 0.0) @ Rf.T
            c.cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(R), np.zeros(3))
        cams.append(c)
    rc = pycolmap.RigConfig(); rc.cameras = cams
    db = pycolmap.Database.open(db_path)
    pycolmap.apply_rig_config([rc], db)
    # 자세 사전값 (front 사진에 붙인다)
    if not a.no_prior:
        by_name = {im.name: im for im in db.read_all_images()}
        cov = np.diag([a.prior_xy ** 2, a.prior_xy ** 2, a.prior_z ** 2])
        for s in sel:
            im = by_name[f'front/{s["pid"]}.jpg']
            pp = pycolmap.PosePrior()
            pp.corr_data_id = pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, im.camera_id), im.image_id)
            pp.position = np.array([s['x'], s['y'], 0.0])
            pp.position_covariance = cov
            pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            db.write_pose_prior(pp)
    nfr = db.num_frames(); nrig = db.num_rigs(); npp = db.num_pose_priors()
    db.close()
    print(f'rig {nrig}, 틀 {nfr}, 사전값 {npp}', flush=True)
    # 짝
    pairs = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            d = math.hypot(sel[i]['x'] - sel[j]['x'], sel[i]['y'] - sel[j]['y'])
            if d <= a.pair_radius:
                for fi in faces:
                    for fj in faces:
                        if a.prune_angle > 0:
                            if 'top' in (fi, fj):
                                if d > a.top_radius:
                                    continue
                            elif fi != 'bottom' and fj != 'bottom':
                                ai = sel[i]['heading'] + 90 * cube.FACES.index(fi); aj = sel[j]['heading'] + 90 * cube.FACES.index(fj)
                                if abs((ai - aj + 180) % 360 - 180) > a.prune_angle:
                                    continue
                        pairs.append((f'{fi}/{sel[i]["pid"]}.jpg', f'{fj}/{sel[j]["pid"]}.jpg'))
    pf = os.path.join(out, 'pairs.txt')
    open(pf, 'w').write('\n'.join(f'{p} {q}' for p, q in pairs) + '\n')
    mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = a.threads; mo.use_gpu = False
    mo.skip_image_pairs_in_same_frame = True
    if a.feature_type != 'SIFT':
        mo.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    po = pycolmap.ImportedPairingOptions(); po.match_list_path = pf
    t1 = time.time()
    pycolmap.match_image_pairs(db_path, matching_options=mo, pairing_options=po)
    t_match = time.time() - t1
    db = pycolmap.Database.open(db_path)
    n_verified = db.num_verified_image_pairs(); db.close()
    print(f'짝 {len(pairs)}, 검증된 짝 {n_verified}, 맞추기 {t_match:.0f}초', flush=True)
    sp = os.path.join(out, 'sparse'); os.makedirs(sp, exist_ok=True)
    t2 = time.time()
    if a.mapper == 'global':
        go = pycolmap.GlobalPipelineOptions(); go.num_threads = a.threads
        recs = pycolmap.global_mapping(db_path, img_dir, sp, options=go)
    else:
        io = pycolmap.IncrementalPipelineOptions()
        io.num_threads = a.threads
        io.ba_refine_focal_length = False; io.ba_refine_principal_point = False; io.ba_refine_extra_params = False
        io.ba_refine_sensor_from_rig = False
        io.mapper.abs_pose_refine_focal_length = False; io.mapper.abs_pose_refine_extra_params = False
        io.multiple_models = True; io.min_model_size = 3
        # 위치 사전값을 쓰는 맞춤은 촬영 자리가 거의 한 줄이면 실패한다 (시험 1 에서 'Alignment w.r.t. prior positions failed').
        # 그래서 지도는 사전값 없이 만들고, 맞춤은 eval_sfm.py (중력 + GPS 수평 닮음) 에서 한다. 사전값은 database 에만 남긴다.
        io.use_prior_position = False
        recs = pycolmap.incremental_mapping(db_path, img_dir, sp, options=io)
    t_map = time.time() - t2
    st = {'panos': len(sel), 'faces': faces, 'images': len(names), 'pairs': len(pairs), 'verified_pairs': n_verified,
          'seconds': {'extract': round(t_ext), 'match': round(t_match), 'map': round(t_map)},
          'options': vars(a), 'pycolmap': pycolmap.__version__, 'models': []}
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        st['models'].append({'id': int(k), 'reg_images': rec.num_reg_images(), 'reg_frames': rec.num_reg_frames() if hasattr(rec, 'num_reg_frames') else None,
                             'points3D': rec.num_points3D(), 'mean_reproj_px': round(rec.compute_mean_reprojection_error(), 3),
                             'mean_track_len': round(rec.compute_mean_track_length(), 2)})
    json.dump({'sel': sel, **st}, open(os.path.join(out, 'stats.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(st['models'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
