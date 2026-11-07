"""VGGT 덩어리들을 한 좌표계로: 덩어리마다 닮음 변환 (축척 · 회전 · 이동) 을 GPS · 중력 · 땅 높이 · 덩어리 사이 겹침으로 함께 맞춘다.

사용: ~/apt-recon/venv/bin/python -I vggt_graph.py <vggt 폴더> <v5_scene.npz> [--sig-gps 1.0] [--sig-pt 0.3] [--px 24] [--conf-q 0.7] [--max-depth 60]
잔차 (Huber, 표준편차로 나눔)
- GPS: 덩어리 안 파노라마 중심 (뷰 중심 평균) 의 수평 위치 ↔ 네이버 위경도 (σ --sig-gps m).
- 중력: 파노라마 틀 위 (뷰 이상 회전으로 얻은 파노라마 회전) ↔ 세계 +z (σ 1°).
- 땅: 카메라 둘레 10 m · 1.5 m 넘게 아래 깊이 점 20 개의 z ↔ v5 모델 땅 (σ 1 m) [v5 땅 = SRTM 다듬은 땅, 자료 한계 있음].
- 겹침 회전: 같은 파노라마를 담은 두 덩어리의 세계 → 파노라마 회전 차 (σ 1°).
- 겹침 점: 같은 파노라마 · 같은 뷰 · 같은 화소를 두 덩어리에서 투영한 3D 점 차 (σ --sig-pt m), 뷰마다 --px 개 화소.
결과: <폴더>/graph.json (전후 수치), <폴더>/graph_sim3.npz, <폴더>/fused_graph.npz (파노라마마다 가장 가운데 덩어리 하나에서 투영).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot


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


def sim2(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    A = (dst - md).T @ (src - ms); U, S, Vt = np.linalg.svd(A)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((src - ms) ** 2).sum()
    return s, R, md - s * R @ ms


def unproj(c, k, uv):
    """덩어리 c 의 뷰 k 에서 화소 uv (정수 열 · 행) 를 덩어리 좌표 3D 점으로 (VGGT 가 낸 그 뷰의 자세 · 내부값 · 깊이)."""
    Ki = np.linalg.inv(c['K'][k]); rays = (Ki @ np.stack([uv[:, 0] + 0.5, uv[:, 1] + 0.5, np.ones(len(uv))])).T
    Xc = rays * c['dep'][k][uv[:, 1], uv[:, 0]][:, None].astype(np.float64)
    return (Xc - c['t'][k]) @ c['R'][k]


def main():
    d = sys.argv[1]; npz = sys.argv[2]
    sig_gps = arg('--sig-gps', 1.0); sig_pt = arg('--sig-pt', 0.3); npx = arg('--px', 24)
    conf_q = arg('--conf-q', 0.7); max_depth = arg('--max-depth', 60.0); sig_rot = np.radians(1.0); sig_g = np.radians(1.0)
    meta = json.load(open(os.path.join(d, 'chunks.json')))
    sel = {s['pid']: s for s in meta['sel']}
    z = np.load(npz)
    keepg = np.isin(z['G'], (0, 1, 2, 3))
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh(); m.vertex.positions = o3d.core.Tensor(z['V']); m.triangle.indices = o3d.core.Tensor(z['F'][keepg])
    sc.add_triangles(m)

    def ground_z(xy):
        o = np.hstack([xy, np.full((len(xy), 1), 600.0)]).astype(np.float32)
        dd = np.tile(np.array([[0, 0, -1.0]], np.float32), (len(xy), 1))
        return 600.0 - sc.cast_rays(o3d.core.Tensor(np.hstack([o, dd])))['t_hit'].numpy()

    rng = np.random.default_rng(0)
    CH = []
    t0 = time.time()
    for ci in range(len(meta['chunks'])):
        f = os.path.join(d, f'chunk_{ci:03d}.npz')
        if not os.path.exists(f):
            continue
        c = np.load(f)
        E = c['extrinsic'].astype(np.float64); K = c['intrinsic'].astype(np.float64)
        R = E[:, :3, :3]; t = E[:, :3, 3]; Cc = -np.einsum('sji,sj->si', R, t)
        RV = c['R_view'].astype(np.float64); pano = c['pano']; face = c['face']
        dep = c['depth'].astype(np.float32); cf = c['conf'].astype(np.float32)
        Rp_est = np.einsum('sji,sjk->sik', RV, R)
        ups = sorted(set(pano), key=list(pano).index)
        Rpano, Cpano = {}, {}
        for p in ups:
            mk = np.where(pano == p)[0]
            U, _, Vt = np.linalg.svd(Rp_est[mk].sum(0)); Rpano[p] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            Cpano[p] = Cc[mk].mean(0)
        g = unit(np.mean([Rpano[p].T @ np.array([0, 0, -1.0]) for p in ups], axis=0))
        Rg = rot_a_to_b(g, np.array([0, 0, -1.0]))
        pc = np.array([Cpano[p] for p in ups]) @ Rg.T
        gps = np.array([[sel[p]['x'], sel[p]['y']] for p in ups])
        s, R2, t2 = sim2(pc[:, :2], gps)
        R3 = np.eye(3); R3[:2, :2] = R2
        Rc = R3 @ Rg; tc = np.array([t2[0], t2[1], 0.0])
        thr = np.quantile(cf, conf_q)
        H = dep.shape[1]

        cd = {'K': K, 't': t, 'R': R, 'dep': dep}
        # 땅 점 (덩어리 좌표) 과 그 자리 모델 땅
        Xall = []
        for k in range(len(E)):
            uv = rng.integers(0, H, size=(400, 2))
            ok = cf[k][uv[:, 1], uv[:, 0]] >= thr
            Xall.append(unproj(cd, k, uv[ok]))
        Xall = np.concatenate(Xall)
        Xw0 = s * Xall @ Rc.T + tc
        grd = []
        for p in ups:
            cw = s * Rc @ Cpano[p] + tc
            mk = (np.linalg.norm(Xw0[:, :2] - cw[:2], axis=1) < 10) & (Xw0[:, 2] < cw[2] - 1.5)
            if mk.sum() >= 5:
                grd.append(Xall[mk][rng.choice(mk.sum(), min(5, mk.sum()), replace=False)])
        grd = np.concatenate(grd) if grd else np.zeros((0, 3))
        if len(grd):
            gz0 = s * grd @ Rc.T + tc
            zm = ground_z(gz0[:, :2])
            tc[2] += float(np.median(zm - gz0[:, 2]))
            zm = ground_z((s * grd @ Rc.T + tc)[:, :2])
        else:
            zm = np.zeros(0)
        # 덩어리 품질: GPS 맞춤 잔차 (수평), 뷰 회전 일관성 (각 뷰가 낸 파노라마 회전과 평균의 각 최댓값)
        gres = np.linalg.norm((s * (pc[:, :2] @ R2.T)) + t2 - gps, axis=1)
        rerr = []
        for p in ups:
            for k in np.where(pano == p)[0]:
                rerr.append(np.degrees(np.arccos(np.clip((np.trace(Rp_est[k] @ Rpano[p].T) - 1) / 2, -1, 1))))
        qual = {'gps_med': float(np.median(gres)), 'gps_max': float(gres.max()), 'rot_max': float(np.max(rerr)), 'scale': float(s)}
        CH.append({'ci': ci, 'E': E, 'K': K, 'R': R, 't': t, 'pano': pano, 'ups': ups, 'Rpano': Rpano, 'Cpano': Cpano, 'qual': qual,
                   'dep': dep, 'cf': cf, 'thr': thr, 'H': H, 'x0': np.concatenate([[np.log(s)], Rot.from_matrix(Rc).as_rotvec(), tc]),
                   'grd': grd, 'zm': zm, 'face': face})
    print(f'덩어리 {len(CH)} 읽음 {time.time() - t0:.0f}초', flush=True)
    # 나쁜 덩어리 거르기 [규칙]: GPS 잔차 중앙 > 3 m 또는 최대 > 6 m, 뷰 회전 일관성 최대 > 20°, 축척이 중앙값의 0.5–2 배 밖
    smed = float(np.median([c['qual']['scale'] for c in CH]))
    keep, dropped = [], []
    for c in CH:
        q = c['qual']
        bad = q['gps_med'] > 3 or q['gps_max'] > 6 or q['rot_max'] > 20 or not (0.5 * smed <= q['scale'] <= 2 * smed)
        (dropped if bad else keep).append(c)
    all_p = set(p for c in CH for p in c['ups']); kept_p = set(p for c in keep for p in c['ups'])
    print(f'거른 덩어리 {len(dropped)}/{len(CH)}, 남은 덩어리의 파노라마 {len(kept_p)}/{len(all_p)}', flush=True)
    drop_info = [{'chunk': c['ci'], **{k: round(v, 2) for k, v in c['qual'].items()}} for c in dropped]
    CH = keep
    # 겹침: 같은 파노라마를 담은 덩어리 쌍
    where = {}
    for i, c in enumerate(CH):
        for p in c['ups']:
            where.setdefault(p, []).append(i)
    pairs = []
    for p, lst in where.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                pairs.append((p, lst[a], lst[b]))
    PTS = []    # (i, j, Xi, Xj)
    for p, i, j in pairs:
        ci, cj = CH[i], CH[j]
        ki = np.where(ci['pano'] == p)[0]; kj = np.where(cj['pano'] == p)[0]
        for a_, b_ in zip(ki, kj):        # 같은 뷰 순서
            if ci['face'][a_] != cj['face'][b_]:
                continue
            uv = rng.integers(0, ci['H'], size=(npx * 4, 2))
            ok = (ci['cf'][a_][uv[:, 1], uv[:, 0]] >= ci['thr']) & (cj['cf'][b_][uv[:, 1], uv[:, 0]] >= cj['thr'])
            uv = uv[ok][:npx]
            if len(uv) < 4:
                continue
            PTS.append((i, j, unproj(ci, a_, uv), unproj(cj, b_, uv)))
    print(f'겹침 쌍 {len(pairs)}, 점 묶음 {len(PTS)}', flush=True)
    n = len(CH); x0 = np.concatenate([c['x0'] for c in CH])
    # 잔차 목록 만들기 (구조)
    gps_i, gps_C, gps_xy = [], [], []
    grav_i, grav_R = [], []
    for i, c in enumerate(CH):
        for p in c['ups']:
            gps_i.append(i); gps_C.append(c['Cpano'][p]); gps_xy.append([sel[p]['x'], sel[p]['y']])
            grav_i.append(i); grav_R.append(c['Rpano'][p])
    gps_i = np.array(gps_i); gps_C = np.array(gps_C); gps_xy = np.array(gps_xy); grav_i = np.array(grav_i); grav_R = np.array(grav_R)
    grd_i = np.concatenate([np.full(len(c['grd']), i) for i, c in enumerate(CH)]).astype(int)
    grd_X = np.concatenate([c['grd'] for c in CH]); grd_zm = np.concatenate([c['zm'] for c in CH])
    rot_pairs = [(i, j, CH[i]['Rpano'][p], CH[j]['Rpano'][p]) for p, i, j in pairs]
    rp_i = np.array([r[0] for r in rot_pairs]); rp_j = np.array([r[1] for r in rot_pairs])
    rp_Ri = np.array([r[2] for r in rot_pairs]); rp_Rj = np.array([r[3] for r in rot_pairs])
    pt_i = np.concatenate([np.full(len(q[2]), q[0]) for q in PTS]).astype(int); pt_j = np.concatenate([np.full(len(q[2]), q[1]) for q in PTS]).astype(int)
    pt_Xi = np.concatenate([q[2] for q in PTS]); pt_Xj = np.concatenate([q[3] for q in PTS])
    ez = np.array([0, 0, 1.0])

    def unpack(x):
        x = x.reshape(n, 7)
        return np.exp(x[:, 0]), Rot.from_rotvec(x[:, 1:4]).as_matrix(), x[:, 4:7]

    def resid(x):
        s, R, T = unpack(x)
        r1 = ((s[gps_i, None] * np.einsum('nij,nj->ni', R[gps_i], gps_C) + T[gps_i])[:, :2] - gps_xy).ravel() / sig_gps
        up = np.einsum('nij,nkj,k->ni', R[grav_i], grav_R, ez)          # R_c R_p^T e_z
        r2 = (up - ez).ravel() / sig_g
        r3 = ((s[grd_i, None] * np.einsum('nij,nj->ni', R[grd_i], grd_X) + T[grd_i])[:, 2] - grd_zm) / 1.0
        Wi = np.einsum('nij,nkj->nik', rp_Ri, R[rp_i]); Wj = np.einsum('nij,nkj->nik', rp_Rj, R[rp_j])   # 세계 → 파노라마
        D = np.einsum('nij,nkj->nik', Wi, Wj)
        r4 = Rot.from_matrix(D).as_rotvec().ravel() / sig_rot
        Pi = s[pt_i, None] * np.einsum('nij,nj->ni', R[pt_i], pt_Xi) + T[pt_i]
        Pj = s[pt_j, None] * np.einsum('nij,nj->ni', R[pt_j], pt_Xj) + T[pt_j]
        r5 = (Pi - Pj).ravel() / sig_pt
        return np.concatenate([r1, r2, r3, r4, r5])

    sizes = [len(gps_i) * 2, len(grav_i) * 3, len(grd_i), len(rp_i) * 3, len(pt_i) * 3]
    A = lil_matrix((sum(sizes), n * 7), dtype=np.int8)
    o = 0
    for idx, k in ((gps_i, 2), (grav_i, 3), (grd_i, 1)):
        for r, i in enumerate(idx):
            A[o + r * k:o + r * k + k, i * 7:i * 7 + 7] = 1
        o += len(idx) * k
    for idx_i, idx_j in ((rp_i, rp_j), (pt_i, pt_j)):
        for r, (i, j) in enumerate(zip(idx_i, idx_j)):
            A[o + r * 3:o + r * 3 + 3, i * 7:i * 7 + 7] = 1; A[o + r * 3:o + r * 3 + 3, j * 7:j * 7 + 7] = 1
        o += len(idx_i) * 3

    def report(x, tag):
        r = resid(x); out = {}
        names = ['gps_m', 'gravity_deg', 'ground_m', 'overlap_rot_deg', 'overlap_pt_m']
        scale = [sig_gps, np.degrees(sig_g), 1.0, np.degrees(sig_rot), sig_pt]
        dims = [2, 3, 1, 3, 3]
        o = 0
        for nm, sz, scl, dm in zip(names, sizes, scale, dims):
            v = r[o:o + sz].reshape(-1, dm) * scl; o += sz
            mag = np.linalg.norm(v, axis=1)
            out[nm] = {'median': round(float(np.median(mag)), 3), 'p90': round(float(np.percentile(mag, 90)), 3)} if len(mag) else None
        print(tag, out, flush=True)
        return out
    before = report(x0, '전')
    t1 = time.time()
    sol = least_squares(resid, x0, jac_sparsity=A, loss='huber', f_scale=1.0, x_scale='jac', method='trf', max_nfev=60, verbose=0)
    print(f'최적화 {time.time() - t1:.0f}초, 상태 {sol.status}', flush=True)
    after = report(sol.x, '후')
    s, R, T = unpack(sol.x)
    # 파노라마 세계 자세: 가장 가운데 덩어리 (덩어리 파노라마 중심 평균에 가장 가까운) 에서
    best = {}
    for i, c in enumerate(CH):
        cen = np.mean([c['Cpano'][p] for p in c['ups']], axis=0)
        for p in c['ups']:
            dd = np.linalg.norm(c['Cpano'][p] - cen) * s[i]
            if p not in best or dd < best[p][1]:
                best[p] = (i, dd)
    # 파노라마별 GPS 차 (가장 가운데 덩어리 값)
    gres = []
    for p, (i, _) in best.items():
        cw = s[i] * R[i] @ CH[i]['Cpano'][p] + T[i]
        gres.append(np.linalg.norm(cw[:2] - np.array([sel[p]['x'], sel[p]['y']])))
    gres = np.array(gres)
    Ps, PN, CI, Cs, pids = [], [], [], [], []
    VW_pid, VW_view, VW_R, VW_t, VW_C, VW_Rp = [], [], [], [], [], []
    for p, (i, _) in sorted(best.items()):
        c = CH[i]; ks = np.where(c['pano'] == p)[0]
        cw = s[i] * R[i] @ c['Cpano'][p] + T[i]
        for k in ks:     # 뷰마다 세계 자세 (m): cam_from_world R = R_k R_c^T, t = -R T + s t_k
            Rk = c['R'][k] @ R[i].T; tk = -Rk @ T[i] + s[i] * c['t'][k]
            VW_pid.append(p); VW_view.append(str(c['face'][k])); VW_R.append(Rk); VW_t.append(tk); VW_C.append(-Rk.T @ tk)
        VW_Rp.append(c['Rpano'][p] @ R[i].T)
        for k in ks:
            vv, uu = np.mgrid[0:c['H']:3, 0:c['H']:3]
            uv = np.stack([uu.ravel(), vv.ravel()], 1)
            ok = c['cf'][k][uv[:, 1], uv[:, 0]] >= c['thr']
            X = unproj(c, k, uv[ok]); Xw = s[i] * X @ R[i].T + T[i]
            ok2 = np.linalg.norm(Xw - cw, axis=1) < max_depth
            Ps.append(Xw[ok2].astype(np.float32)); PN.append(np.full(ok2.sum(), len(pids), np.int32)); CI.append(np.full(ok2.sum(), c['ci'], np.int16))
        Cs.append(cw); pids.append(p)
    out = {'chunks': len(CH), 'chunks_dropped': drop_info, 'panos_total': len(all_p), 'panos': len(best),
           'panos_ratio': round(len(best) / len(all_p), 3), 'pairs': len(pairs), 'point_groups': len(PTS), 'before': before, 'after': after,
           'gps_res_per_pano_m': {'median': round(float(np.median(gres)), 2), 'p90': round(float(np.percentile(gres, 90)), 2),
                                  'max': round(float(gres.max()), 2), 'within_2m': round(float((gres < 2).mean()), 3)},
           'scale_m_per_unit': {'median': round(float(np.median(s)), 2), 'min': round(float(s.min()), 2), 'max': round(float(s.max()), 2)},
           'params': {'sig_gps': sig_gps, 'sig_pt': sig_pt, 'px': npx, 'conf_q': conf_q, 'max_depth': max_depth}}
    json.dump(out, open(os.path.join(d, 'graph.json'), 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(d, 'graph_sim3.npz'), s=s, R=R, T=T, ci=np.array([c['ci'] for c in CH]))
    np.savez_compressed(os.path.join(d, 'views_world.npz'), pid=np.array(VW_pid), view=np.array(VW_view), R=np.array(VW_R), t=np.array(VW_t),
                        C=np.array(VW_C), pano_pids=np.array(sorted(best)), pano_R=np.array(VW_Rp), pano_C=np.array(Cs))
    P = np.concatenate(Ps)
    np.savez_compressed(os.path.join(d, 'fused_graph.npz'), P=P, pano=np.concatenate(PN), chunk=np.concatenate(CI), C=np.array(Cs), pids=np.array(pids),
                        track=np.full(len(P), 3, np.int16))
    print(json.dumps({k: v for k, v in out.items() if k not in ('before', 'after')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
