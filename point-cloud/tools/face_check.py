"""큐브맵 6면의 이음매 점검 — 면 화각이 90° 인지, 위 · 아래 면이 어떤 방향으로 붙는지.

사용: ~/apt-recon/venv/bin/python -I face_check.py <panos 폴더> [pano_id ...]
- 옆면 이음매 (front→right→back→left→front): A 의 오른쪽 끝 열과 B 의 왼쪽 열 k (k=0..24) 의 NCC.
  k=0 에서 가장 높으면 겹침 없는 90° 면이다.
- 위 · 아래 면: 옆면 네 개의 위(아래) 끝 줄이 위(아래) 면의 네 변 중 어디에 붙는지 (정방향 · 역방향 NCC).
사진 내용은 출력하지 않는다 (숫자만).
"""
import os, sys, json
import numpy as np
from PIL import Image

SIDES = ['front', 'right', 'back', 'left']


def load(d, name):
    im = Image.open(os.path.join(d, name + '.jpg')).convert('L')
    a = np.asarray(im, np.float32)
    # 살짝 흐리게 (JPEG 블록 잡음 줄이기)
    k = np.array([1, 2, 1], np.float32) / 4
    a = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, a)
    a = np.apply_along_axis(lambda c: np.convolve(c, k, 'same'), 0, a)
    return a


def ncc(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def edges(a):
    # 변: 위(왼→오), 오른쪽(위→아래), 아래(왼→오), 왼쪽(위→아래); 끝 1 px 은 흐림 가장자리라 1 px 안쪽
    return {'top': a[1, :], 'right': a[:, -2], 'bottom': a[-2, :], 'left': a[:, 1]}


def main():
    root = sys.argv[1]
    ids = sys.argv[2:] or sorted(os.listdir(root))
    out = {}
    for pid in ids:
        d = os.path.join(root, pid)
        F = {n: load(d, n) for n in SIDES + ['top', 'bottom']}
        res = {'size': list(F['front'].shape)}
        # 옆면 이음매
        side = []
        for i, A in enumerate(SIDES):
            B = SIDES[(i + 1) % 4]
            a = F[A][:, -2]
            sc = [ncc(a, F[B][:, 1 + k]) for k in range(25)]
            # 반대로: A 의 안쪽 열 j 와 B 의 첫 열
            sc2 = [ncc(F[A][:, -2 - j], F[B][:, 1]) for j in range(25)]
            side.append({'pair': f'{A}->{B}', 'best_k_in_B': int(np.argmax(sc)), 'ncc_k0': round(sc[0], 3),
                         'ncc_max': round(max(sc), 3), 'best_j_in_A': int(np.argmax(sc2)),
                         'ncc_far': round(sc[24], 3)})
        res['side'] = side
        # 위 · 아래 면
        for cap, row in (('top', 1), ('bottom', -2)):
            E = edges(F[cap])
            m = {}
            for s in SIDES:
                r = F[s][row, :]
                cands = [(ncc(r, e), en, False) for en, e in E.items()] + \
                        [(ncc(r, e[::-1]), en, True) for en, e in E.items()]
                best = max(cands)
                m[s] = {'edge': best[1], 'reversed': best[2], 'ncc': round(best[0], 3)}
            res[cap] = m
        out[pid] = res
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
