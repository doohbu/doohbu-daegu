"""v5 장면 (meshes.npz + scene.json 물체 행렬) → 세계 좌표 삼각형 하나로 합친 npz. 원본은 읽기만 한다.

사용: ~/apt-recon/venv/bin/python scene_mesh.py <v5/work 폴더> <out.npz>
세계 좌표 = v5 지역 좌표: x 동쪽 m, y 북쪽 m, z 위 m (z 0 = 해발 datum_m, scene.json info.terrain.datum_m).
나오는 배열: V (n,3) float32, F (m,3) int32, G (m,) uint8 무리 번호, groups (이름 목록), tower (m,) int16 동 번호 (아니면 0).
무리: 0 building, 1 site, 2 context_bldg (ctx_*), 3 ground (ground*), 4 car, 5 lamp (slamp · lamp), 6 tree, 7 기타.
"""
import json, sys
import numpy as np

GROUPS = ['building', 'site', 'context_bldg', 'ground', 'car', 'lamp', 'tree', 'other']


def kind(ob):
    g = ob['group']; n = ob['name']
    if g.startswith('building:'):
        return 0
    if g == 'site':
        return 1
    if n.startswith('ctx'):
        return 2
    if n.startswith('ground'):
        return 3
    if n.startswith('car'):
        return 4
    if n.startswith('slamp') or n.startswith('lamp'):
        return 5
    if g == 'landscape':
        return 6
    return 7


def main():
    work, out = sys.argv[1], sys.argv[2]
    s = json.load(open(f'{work}/scene.json'))
    z = np.load(f'{work}/meshes.npz')
    Vs, Fs, Gs, Ts = [], [], [], []
    nv = 0
    for ob in s['objects']:
        M = np.asarray(ob['matrix'], np.float64)
        k = kind(ob)
        t = int(ob['name']) if k == 0 and ob['name'].isdigit() else 0
        for mat in s['meshes'][ob['mesh']]:
            kv, kf = f"{ob['mesh']}|{mat}|V", f"{ob['mesh']}|{mat}|F"
            if kv not in z.files:
                continue
            V = z[kv].astype(np.float64)
            F = z[kf].astype(np.int64)
            if len(F) == 0:
                continue
            Vw = V @ M[:3, :3].T + M[:3, 3]
            Vs.append(Vw.astype(np.float32)); Fs.append((F + nv).astype(np.int32))
            Gs.append(np.full(len(F), k, np.uint8)); Ts.append(np.full(len(F), t, np.int16))
            nv += len(V)
    V = np.concatenate(Vs); F = np.concatenate(Fs); G = np.concatenate(Gs); T = np.concatenate(Ts)
    np.savez_compressed(out, V=V, F=F, G=G, tower=T, groups=np.array(GROUPS),
                        origin=np.array(s['info']['origin']), datum_m=np.array(s['info']['terrain']['datum_m']))
    print('V', V.shape, 'F', F.shape, {GROUPS[i]: int((G == i).sum()) for i in range(len(GROUPS))})
    print('범위', V.min(0).round(1), V.max(0).round(1))


if __name__ == '__main__':
    main()
