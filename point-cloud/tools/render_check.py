"""v5 모델을 파노라마 자리에서 360° 로 광선 추적해, 사진과 같은 등장방형 띠로 만든다 (방향 규약 · 자리 점검).

사용: ~/apt-recon/venv/bin/python -I render_check.py <v5_scene.npz> <panos 폴더> <out 폴더> [pano_id ...] [--h 2.5]
- 카메라 자리: 위경도 → v5 지역 xy, z = 모델 땅 (건물 · 단지 · 땅 무리만, 아래로 쏜 광선) + h (기본 2.5 m [추측]).
- 모델 띠: 세계 방위 −180..180 · 고도 −30..80°, 0.5° 칸. 값 = 거리 (m, 하늘 = inf), 무리 번호.
- 사진 띠: front 방위 = 0 으로 둔 같은 격자 (사진 쪽 방위는 상대값). 사진 띠는 내부 점검용 (out/private/), 보고서에 싣지 않는다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import open3d as o3d
from PIL import Image
import cube

KEEP = (0, 1, 2, 3)        # building · site · context_bldg · ground (차 · 가로등 · 나무는 자리가 [추측] 이라 뺀다)
STEP = 0.5
AZ = np.arange(-180, 180, STEP) + STEP / 2
EL = np.arange(80, -30, -STEP) - STEP / 2      # 위가 먼저


def build_scene(npz):
    z = np.load(npz)
    keep = np.isin(z['G'], KEEP)
    F = z['F'][keep]; G = z['G'][keep]; T = z['tower'][keep]
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(z['V'])
    m.triangle.indices = o3d.core.Tensor(F)
    sc.add_triangles(m)
    return sc, G, T


def cast(sc, orig, dirs):
    d = dirs.reshape(-1, 3).astype(np.float32)
    o = np.broadcast_to(np.asarray(orig, np.float32), d.shape)
    ans = sc.cast_rays(o3d.core.Tensor(np.hstack([o, d])))
    t = ans['t_hit'].numpy(); pid = ans['primitive_ids'].numpy()
    return t.reshape(dirs.shape[:-1]), pid.reshape(dirs.shape[:-1])


def ground_z(sc, x, y):
    t, _ = cast(sc, (x, y, 600.0), np.array([[0, 0, -1.0]]))
    return 600.0 - float(t[0])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    h = 2.5
    if '--h' in sys.argv:
        h = float(sys.argv[sys.argv.index('--h') + 1]); args.remove(str(sys.argv[sys.argv.index('--h') + 1]) if str(sys.argv[sys.argv.index('--h') + 1]) in args else None)
    npz, panos, out = args[0], args[1], args[2]
    ids = args[3:]
    os.makedirs(os.path.join(out, 'private'), exist_ok=True)
    sc, G, T = build_scene(npz)
    azg, elg = np.meshgrid(AZ, EL)
    D = cube.dirs_equirect(azg, elg)
    res = {}
    for pid in ids:
        meta = json.load(open(os.path.join(panos, pid, 'meta.json')))
        x, y = cube.local_xy(meta['lon'], meta['lat'])
        x, y = float(x), float(y)
        zg = ground_z(sc, x, y)
        C = (x, y, zg + h)
        t, prim = cast(sc, C, D)
        grp = np.where(np.isfinite(t), G[np.clip(prim, 0, len(G) - 1)], 255).astype(np.uint8)
        twr = np.where(np.isfinite(t), T[np.clip(prim, 0, len(T) - 1)], 0).astype(np.int16)
        # 사진 띠 (front 방위 0 기준 상대 방위)
        faces = {f: np.asarray(Image.open(os.path.join(panos, pid, f + '.jpg')).convert('RGB').resize((512, 512), Image.BILINEAR))
                 for f in cube.FACES}
        photo = cube.sample_cube(faces, D, 0.0)
        np.savez_compressed(os.path.join(out, f'model_{pid}.npz'), t=t.astype(np.float32), grp=grp, tower=twr, C=np.array(C),
                            az=AZ, el=EL)
        np.save(os.path.join(out, 'private', f'photo_{pid}.npy'), photo)
        res[pid] = {'xy': [round(x, 2), round(y, 2)], 'ground_z': round(zg, 2), 'cam_z': round(zg + h, 2),
                    'heading_deg': meta['heading_deg'], 'sky_frac_model': round(float(np.isinf(t).mean()), 3)}
        print(pid, res[pid], flush=True)
    json.dump(res, open(os.path.join(out, 'render_check.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
