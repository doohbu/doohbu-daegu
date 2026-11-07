"""VGGT 덩어리 결과 → v5 지역 좌표로 맞춤 · 점검 · 합친 점구름.

사용: ~/apt-recon/venv/bin/python -I vggt_align.py <vggt 폴더> <v5_scene.npz> [--conf-q 0.5] [--stride 3] [--max-depth 120]
덩어리마다
1. rig 일관성: 한 파노라마 면들의 카메라 중심 퍼짐 (덩어리 단위 → 맞춘 뒤 m), 면 사이 상대 회전과 큐브 회전의 차 (°).
2. 중력: 옆면 카메라 y 축 (영상 아래) 평균 → −z. 수평 오차 = 각 옆면 y 축과 평균의 각.
3. 수평 닮음 변환: 파노라마 중심 (면 중심 평균) xy ↔ 네이버 위경도의 v5 xy (강건 최소제곱).
4. 높이: 카메라 둘레 10 m 안, 카메라보다 1.5 m 넘게 낮은 깊이 점의 중앙값 = 땅 → v5 모델 땅과 맞춘다 (카메라 높이도 잼).
5. 예측 화각: 내부값 f 로 본 화각 (실제 90°).
점: 깊이 신뢰도 덩어리 안 분위 --conf-q 위, 화소 --stride 칸마다, 깊이 --max-depth m 안 (맞춘 뒤).
결과: <폴더>/align.json, <폴더>/fused.npz (P, conf, chunk, face, pano 번호, C 파노라마 중심, pids).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import open3d as o3d
import cube

SIDE = ('front', 'right', 'back', 'left')


def arg(n, d):
    return type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


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
    d = sys.argv[1]; npz = sys.argv[2]
    conf_q = arg('--conf-q', 0.5); stride = arg('--stride', 3); max_depth = arg('--max-depth', 120.0)
    rig = '--rig' in sys.argv; tag = arg('--tag', 'rig' if rig else 'raw')
    meta = json.load(open(os.path.join(d, 'chunks.json')))
    sel = {s['pid']: s for s in meta['sel']}
    pid_list = [s['pid'] for s in meta['sel']]; pid_idx = {p: i for i, p in enumerate(pid_list)}
    z = np.load(npz)
    keepg = np.isin(z['G'], (0, 1, 2, 3))
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh(); m.vertex.positions = o3d.core.Tensor(z['V']); m.triangle.indices = o3d.core.Tensor(z['F'][keepg])
    sc.add_triangles(m)

    def ground_z(xy):
        o = np.hstack([xy, np.full((len(xy), 1), 600.0)]).astype(np.float32)
        dd = np.tile(np.array([[0, 0, -1.0]], np.float32), (len(xy), 1))
        return 600.0 - sc.cast_rays(o3d.core.Tensor(np.hstack([o, dd])))['t_hit'].numpy()

    Rf = cube.face_R('front', 0.0)
    out = {'chunks': [], 'params': {'conf_q': conf_q, 'stride': stride, 'max_depth': max_depth}}
    Ps, Cs, CHs, FCs, PNs = [], [], [], [], []
    centers = {}
    for ci in range(len(meta['chunks'])):
        f = os.path.join(d, f'chunk_{ci:03d}.npz')
        if not os.path.exists(f):
            continue
        c = np.load(f)
        E = c['extrinsic'].astype(np.float64); Kk = c['intrinsic'].astype(np.float64)
        R = E[:, :3, :3]; t = E[:, :3, 3]
        Cc = -np.einsum('sji,sj->si', R, t)                    # 카메라 중심 (덩어리 좌표)
        pano = c['pano']; face = c['face']
        res_px = c['depth'].shape[-1]
        fov = np.degrees(2 * np.arctan((res_px / 2) / Kk[:, 0, 0]))
        # 뷰마다 이상 회전 (파노라마 틀 → 카메라). 큐브 면이면 cube.face_R, 다시 그린 뷰면 저장된 R_view.
        RV = c['R_view'].astype(np.float64) if 'R_view' in c.files else np.array([cube.face_R(str(fc), 0.0) for fc in face])
        Rp_est = np.einsum('sji,sjk->sik', RV, R)              # R_view^T R = 덩어리 → 파노라마 틀 회전 (뷰마다 추정)
        ups = sorted(set(pano), key=list(pano).index)
        pc, spread, rigerr, Rpano = [], [], [], {}
        for p in ups:
            mk = np.where(pano == p)[0]
            U, _, Vt = np.linalg.svd(Rp_est[mk].sum(0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            Rpano[p] = Rm
            for k in mk:
                rigerr.append(float(np.degrees(np.arccos(np.clip((np.trace(Rp_est[k] @ Rm.T) - 1) / 2, -1, 1)))))
            cc = Cc[mk]; pc.append(cc.mean(0)); spread.append(float(np.linalg.norm(cc - cc.mean(0), axis=1).max()))
        # 중력: 파노라마 틀의 아래 (0,0,-1) 를 덩어리 좌표로
        down = np.array([Rpano[pano[k]].T @ np.array([0, 0, -1.0]) for k in range(len(E))])
        down_v = np.einsum('sji,j->si', Rp_est, np.array([0, 0, -1.0]))
        g = unit(down.mean(0)); lev = np.degrees(np.arccos(np.clip(down_v @ g, -1, 1)))
        Rg = rot_a_to_b(g, np.array([0, 0, -1.0]))
        pc = np.array(pc) @ Rg.T
        gps = np.array([[sel[p]['x'], sel[p]['y']] for p in ups])
        s, R2, t2, res, keep = sim2(pc[:, :2], gps)
        R3 = np.eye(3); R3[:2, :2] = R2
        M = s * R3 @ Rg; T = np.array([t2[0], t2[1], 0.0])
        Cw = (np.array([Cc[pano == p].mean(0) for p in ups]) @ M.T) + T
        Rw = R3 @ Rg
        dhead = []
        for p in ups:
            fwd = Rw @ (Rpano[p].T @ np.array([0, 1.0, 0]))
            az = (np.degrees(np.arctan2(fwd[0], fwd[1])) + 360) % 360
            dhead.append(float((az - sel[p]['heading'] + 180) % 360 - 180))
        # 깊이 점
        H = W = res_px
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        uu = uu.ravel() + 0.5; vv = vv.ravel() + 0.5
        dep = c['depth'][:, ::stride, ::stride].reshape(len(E), -1).astype(np.float64)
        cf = c['conf'][:, ::stride, ::stride].reshape(len(E), -1).astype(np.float64)
        thr = np.quantile(cf, conf_q)
        pts, pcf, pfc, ppn = [], [], [], []
        pc_raw = {p: Cc[pano == p].mean(0) for p in ups}
        for k in range(len(E)):
            Ki = np.linalg.inv(Kk[k])
            rays = (Ki @ np.stack([uu, vv, np.ones_like(uu)])).T
            Xc = rays * dep[k][:, None]
            if rig:   # 같은 파노라마의 뷰는 중심 하나 · 회전은 이상 회전 × 파노라마 평균 회전
                Rk = RV[k] @ Rpano[pano[k]]; Ck = pc_raw[pano[k]]
                Xw = Xc @ Rk + Ck
            else:
                Xw = (Xc - t[k]) @ R[k]                         # R^T (Xc - t)
            Xa = Xw @ M.T + T
            dist = np.linalg.norm(Xa - Cw[ups.index(pano[k])], axis=1)
            ok = (cf[k] >= thr) & (dist < max_depth)
            pts.append(Xa[ok]); pcf.append(cf[k][ok]); pfc.append(np.full(ok.sum(), k % 128, np.int8))   # 덩어리 안 뷰 번호 (0..)
            ppn.append(np.full(ok.sum(), pid_idx[pano[k]], np.int32))
        P = np.concatenate(pts)
        # 높이 원점: 카메라 둘레 땅 점
        dz_list, h_list = [], []
        for i, cw in enumerate(Cw):
            dxy = np.linalg.norm(P[:, :2] - cw[:2], axis=1)
            g_sel = (dxy < 10) & (P[:, 2] < cw[2] - 1.5 * s / max(s, 1e-9))
            if g_sel.sum() >= 50:
                gz = np.median(P[g_sel, 2])
                gzm = np.median(ground_z(P[g_sel][::max(1, g_sel.sum() // 400), :2]))
                dz_list.append(gzm - gz); h_list.append(cw[2] - gz)
        dz = float(np.median(dz_list)) if dz_list else 0.0
        P[:, 2] += dz; Cw[:, 2] += dz
        for p, cw in zip(ups, Cw):
            centers.setdefault(p, []).append((ci, cw))
        Ps.append(P.astype(np.float32)); CHs.append(np.full(len(P), ci, np.int16)); FCs.append(np.concatenate(pfc)); PNs.append(np.concatenate(ppn))
        Cs.append(Cw)
        out['chunks'].append({'chunk': ci, 'frames': int(len(E)), 'panos': len(ups), 'seconds': float(c['seconds']), 'peak_gib': float(c['peak_gib']),
                              'scale_m_per_unit': round(float(s), 3), 'gps_res_m': {'median': round(float(np.median(res)), 2), 'max': round(float(res.max()), 2)},
                              'rig_center_spread_m': {'median': round(float(np.median(spread) * s), 3), 'max': round(float(np.max(spread) * s), 3)},
                              'rig_rot_err_deg': {'median': round(float(np.median(rigerr)), 2), 'max': round(float(np.max(rigerr)), 2)},
                              'leveling_deg': {'median': round(float(np.median(lev)), 2), 'max': round(float(lev.max()), 2)},
                              'fov_pred_deg': {'median': round(float(np.median(fov)), 2), 'min': round(float(fov.min()), 2), 'max': round(float(fov.max()), 2)},
                              'cam_height_m': round(float(np.median(h_list)), 2) if h_list else None, 'z_offset_m': round(dz, 2),
                              'heading_minus_naver_deg': {'median': round(float(np.median(dhead)), 2), 'absmax': round(float(np.max(np.abs(dhead))), 2)},
                              'points': int(len(P))})
        print(out['chunks'][-1], flush=True)
    # 덩어리 사이: 같은 파노라마 중심 차
    diffs = []
    for p, lst in centers.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                diffs.append(np.linalg.norm(lst[i][1] - lst[j][1]))
    out['between_chunks_center_diff_m'] = {'n': len(diffs), 'median': round(float(np.median(diffs)), 2) if diffs else None,
                                           'max': round(float(np.max(diffs)), 2) if diffs else None}
    allc = {p: np.mean([c for _, c in lst], axis=0) for p, lst in centers.items()}
    pids = sorted(allc)
    json.dump(out, open(os.path.join(d, f'align_{tag}.json'), 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(d, f'fused_{tag}.npz'), P=np.concatenate(Ps), chunk=np.concatenate(CHs), face=np.concatenate(FCs),
                        pano=np.concatenate(PNs), C=np.array([allc[p] for p in pids]), pids=np.array(pids),
                        track=np.full(sum(len(x) for x in Ps), 3, np.int16))
    print('덩어리 사이 같은 파노라마 중심 차', out['between_chunks_center_diff_m'])


if __name__ == '__main__':
    main()
