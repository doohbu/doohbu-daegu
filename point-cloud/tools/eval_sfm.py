"""SfM 결과 점검: 등록 비율 · 재투영 오차 · 중력 (수평) · GPS 와 맞춤 · 방위 차 · 카메라 높이 · v5 모델 면까지 거리.

사용: ~/apt-recon/venv/bin/python -I eval_sfm.py <sfm 폴더> <v5_scene.npz> [--model 0]
맞춤 순서
1. 중력: 파노라마는 수평이 맞춰진 사진 (네이버 pitch · roll 0) → 틀마다 front 카메라 y 축 (영상 아래) 이 세계 '아래'. 그 평균을 −z 로 돌린다.
2. 수평 닮음 변환 (축척 · 방위 · 이동): 틀 중심 xy ↔ 네이버 위경도의 v5 지역 xy. 강건 최소제곱 (3σ 밖 제외, 3 회).
3. 높이 원점: SfM 땅 점 (카메라 아래 반경 10 m, 1.5 m 넘게 낮은 점) 과 v5 모델 땅 높이의 중앙값 차로 맞춘다 → 카메라 높이도 이 값으로 잰다.
결과: <sfm>/eval.json, <sfm>/aligned_points.npz (점 xyz · 색 · 모델까지 거리 · 가장 가까운 무리), 그림은 figs_* 에 (형상만).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pycolmap
import open3d as o3d

GROUPS = ['building', 'site', 'context_bldg', 'ground', 'car', 'lamp', 'tree', 'other']


def unit(v):
    return v / np.linalg.norm(v)


def rot_a_to_b(a, b):
    a = unit(a); b = unit(b); v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def sim2(src, dst, iters=3):
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        s0, d0 = src[keep], dst[keep]
        ms, md = s0.mean(0), d0.mean(0)
        A = (d0 - md).T @ (s0 - ms)
        U, S, Vt = np.linalg.svd(A)
        D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        s = np.trace(np.diag(S) @ D) / ((s0 - ms) ** 2).sum()
        t = md - s * R @ ms
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        sig = 1.4826 * np.median(res[keep])
        keep = res <= max(3 * sig, 0.5)
    return s, R, t, res, keep


def main():
    sfm = sys.argv[1]; npz = sys.argv[2]
    mid = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else None
    st = json.load(open(os.path.join(sfm, 'stats.json' if os.path.exists(os.path.join(sfm, 'stats.json')) else 'stats_pre.json')))
    sel = {s['pid']: s for s in st['sel']}
    mid = mid if mid is not None else str(st['models'][0]['id'])
    rec = pycolmap.Reconstruction(os.path.join(sfm, 'sparse', mid))
    faces = st['faces']
    # 틀 (파노라마) 별 front 영상
    fr = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        f, pid = im.name.split('/')[0], im.name.split('/')[1][:-4]
        fr.setdefault(pid, {})[f] = im
    pids = sorted(p for p in fr if 'front' in fr[p])
    reg_images = rec.num_reg_images()
    C = np.array([fr[p]['front'].projection_center() for p in pids])
    Rcw = [fr[p]['front'].cam_from_world().rotation.matrix() for p in pids]
    down = np.array([R[1] for R in Rcw]); fwd = np.array([R[2] for R in Rcw])
    g = unit(down.mean(0))
    lev = np.degrees(np.arccos(np.clip(down @ g, -1, 1)))
    Rg = rot_a_to_b(g, np.array([0, 0, -1.0]))
    Cg = C @ Rg.T
    gps = np.array([[sel[p]['x'], sel[p]['y']] for p in pids])
    s, R2, t2, res, keep = sim2(Cg[:, :2], gps)
    R3 = np.eye(3); R3[:2, :2] = R2
    M = s * R3 @ Rg      # 점 변환: p' = M p + T
    T = np.array([t2[0], t2[1], 0.0])
    Ca = C @ M.T + T
    fa = (fwd @ (R3 @ Rg).T)
    az = (np.degrees(np.arctan2(fa[:, 0], fa[:, 1])) + 360) % 360
    hd = np.array([sel[p]['heading'] for p in pids])
    dh = (az - hd + 180) % 360 - 180
    # 점
    P = np.array([p.xyz for p in rec.points3D.values()]); col = np.array([p.color for p in rec.points3D.values()])
    err = np.array([p.error for p in rec.points3D.values()]); tl = np.array([p.track.length() for p in rec.points3D.values()])
    Pa = P @ M.T + T
    # 모델 장면
    z = np.load(npz)
    keepg = np.isin(z['G'], (0, 1, 2, 3))
    F = z['F'][keepg]; G = z['G'][keepg]; TW = z['tower'][keepg]
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh(); m.vertex.positions = o3d.core.Tensor(z['V']); m.triangle.indices = o3d.core.Tensor(F)
    sc.add_triangles(m)

    def ground_z(xy):
        o = np.hstack([xy, np.full((len(xy), 1), 600.0)]).astype(np.float32)
        d = np.tile(np.array([[0, 0, -1.0]], np.float32), (len(xy), 1))
        ans = sc.cast_rays(o3d.core.Tensor(np.hstack([o, d])))
        return 600.0 - ans['t_hit'].numpy()
    # 높이 원점: 카메라 아래 반경 3 m, 카메라보다 1 m 넘게 낮은 점 = 땅 점
    dz_list = []; h_list = []
    for i, c in enumerate(Ca):
        dxy = np.linalg.norm(Pa[:, :2] - c[:2], axis=1)
        sel_g = (dxy < 10.0) & (Pa[:, 2] < c[2] - 1.5) & (tl >= 3)
        if sel_g.sum() >= 5:
            gz_sfm = np.median(Pa[sel_g, 2])
            gz_model = ground_z(Pa[sel_g, :2])
            dz_list.append(np.median(gz_model - Pa[sel_g, 2]))
            h_list.append(c[2] - gz_sfm)
    dz = float(np.median(dz_list)) if dz_list else 0.0
    Pa[:, 2] += dz; Ca[:, 2] += dz; T[2] += dz
    gz_cam = ground_z(Ca[:, :2])
    # 모델 면까지 거리
    q = sc.compute_closest_points(o3d.core.Tensor(Pa.astype(np.float32)))
    cp = q['points'].numpy(); pidx = q['primitive_ids'].numpy()
    nrm = q['primitive_normals'].numpy()
    vec = Pa - cp
    dist = np.linalg.norm(vec, axis=1)
    sgn = np.sign((vec * nrm).sum(1))
    grp = G[pidx]; twr = TW[pidx]
    out = {
        'model_id': mid, 'images_total': st['images'], 'images_registered': reg_images,
        'images_registered_ratio': round(reg_images / st['images'], 3),
        'frames_total': st['panos'], 'frames_registered_front': len(pids),
        'frames_registered_ratio': round(len(pids) / st['panos'], 3),
        'points3D': int(len(P)), 'mean_reproj_px': round(float(rec.compute_mean_reprojection_error()), 3),
        'point_error_px_median': round(float(np.median(err)), 3), 'mean_track_len': round(float(tl.mean()), 2),
        'leveling_deg': {'median': round(float(np.median(lev)), 3), 'max': round(float(lev.max()), 3)},
        'sim2': {'scale': round(float(s), 4), 'rot_deg': round(float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))), 3),
                 'inliers': int(keep.sum())},
        'gps_residual_m': {'median': round(float(np.median(res)), 2), 'p90': round(float(np.percentile(res, 90)), 2),
                           'max': round(float(res.max()), 2)},
        'heading_minus_naver_deg': {'mean': round(float(dh.mean()), 2), 'std': round(float(dh.std()), 2),
                                    'min': round(float(dh.min()), 2), 'max': round(float(dh.max()), 2)},
        'z_offset_to_model_ground_m': round(dz, 2),
        'camera_height_above_sfm_ground_m': {'median': round(float(np.median(h_list)) * 1, 2) if h_list else None,
                                             'n': len(h_list)},
        'camera_z_minus_model_ground_m': {'median': round(float(np.median(Ca[:, 2] - gz_cam)), 2)},
        'dist_to_model_m': {},
    }
    for gi, gn in enumerate(GROUPS):
        mk = (grp == gi) & (tl >= 3)
        if mk.sum() < 10:
            continue
        d = dist[mk]
        out['dist_to_model_m'][gn] = {'n': int(mk.sum()), 'median': round(float(np.median(d)), 2),
                                      'p25': round(float(np.percentile(d, 25)), 2), 'p75': round(float(np.percentile(d, 75)), 2),
                                      'within_0.5m': round(float((d < 0.5).mean()), 3), 'within_1m': round(float((d < 1).mean()), 3),
                                      'signed_median': round(float(np.median(d * sgn[mk])), 2)}
    per_frame = [{'pid': p, 'gps_res_m': round(float(r), 2), 'dheading_deg': round(float(x), 2),
                  'faces_registered': sorted(fr[p].keys())} for p, r, x in zip(pids, res, dh)]
    out['per_frame'] = per_frame
    out['transform'] = {'M': M.tolist(), 'T': T.tolist()}
    json.dump(out, open(os.path.join(sfm, f'eval_{mid}.json'), 'w'), ensure_ascii=False, indent=1)
    # 파노라마 세계 자세 (세계 → 파노라마 틀: x 오른쪽 · y 앞 · z 위) — front 카메라 회전에서
    A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    Ra = R3 @ Rg
    pano_R = np.array([A @ Rc @ Ra.T for Rc in Rcw])
    np.savez_compressed(os.path.join(sfm, f'aligned_{mid}.npz'), P=Pa.astype(np.float32), color=col.astype(np.uint8),
                        pano_R=pano_R, pano_C=Ca, pano_pids=np.array(pids),
                        dist=dist.astype(np.float32), sgn=sgn.astype(np.int8), grp=grp, tower=twr, track=tl, err=err.astype(np.float32),
                        C=Ca, pids=np.array(pids))
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_frame', 'transform')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
