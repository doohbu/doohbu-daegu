"""큐브맵 6면 → 임의 방위 · 고개 · 화각의 원근 뷰 (쌍선형). 뷰 틀 = 파노라마 틀 (front 방위 0) 기준.
view_R(yaw, pitch): 파노라마 틀 (x 오른쪽 = front 의 오른쪽, y 앞 = front, z 위) → 카메라 (OpenCV) 회전. yaw 는 front 에서 시계 방향 °, pitch 는 위로 +°.
"""
import math, os
import numpy as np
from PIL import Image
import cube


def view_R(yaw, pitch):
    """세계 → 카메라 회전 (세계 = 파노라마 틀, front 방위 0 이라 둘 때). cube.face_R 과 같은 규약."""
    a = math.radians(yaw); p = math.radians(pitch)
    z = np.array([math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), math.sin(p)])
    up = np.array([0, 0, 1.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)          # 오른쪽 (수평)
    y = np.cross(z, x)                                   # 영상 아래
    return np.stack([x, y, z])


def load_faces(d, names=('front', 'right', 'back', 'left', 'top', 'bottom')):
    return {n: np.asarray(Image.open(os.path.join(d, n + '.jpg')).convert('RGB'), np.float32) for n in names}


def render(faces, yaw, pitch, fov=90.0, size=518, ss=2):
    """ss 배로 크게 그린 뒤 줄인다 (계단 · 깜빡임 줄이기)."""
    if ss > 1:
        big, R, _ = render(faces, yaw, pitch, fov, size * ss, 1)
        img = np.asarray(Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS), np.float32)
        f = (size / 2) / math.tan(math.radians(fov) / 2)
        return img, R, np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    R = view_R(yaw, pitch)
    f = (size / 2) / math.tan(math.radians(fov) / 2)
    u = (np.arange(size) + 0.5 - size / 2) / f
    uu, vv = np.meshgrid(u, u)
    d_cam = np.stack([uu, vv, np.ones_like(uu)], -1)
    d = d_cam @ R                                        # 파노라마 틀 방향
    d = d.reshape(-1, 3)
    out = np.zeros((d.shape[0], 3), np.float32); done = np.zeros(d.shape[0], bool)
    for name, img in faces.items():
        H, W = img.shape[:2]
        p = d @ cube.face_R(name, 0.0).T
        ok = (p[:, 2] > 1e-9) & ~done
        x = np.where(ok, p[:, 0] / np.maximum(p[:, 2], 1e-9), 9); y = np.where(ok, p[:, 1] / np.maximum(p[:, 2], 1e-9), 9)
        ok &= (np.abs(x) <= 1) & (np.abs(y) <= 1)
        px = (x[ok] + 1) / 2 * W - 0.5; py = (y[ok] + 1) / 2 * H - 0.5
        x0 = np.clip(np.floor(px).astype(int), 0, W - 1); y0 = np.clip(np.floor(py).astype(int), 0, H - 1)
        x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
        wx = np.clip(px - x0, 0, 1)[:, None]; wy = np.clip(py - y0, 0, 1)[:, None]
        out[ok] = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy) + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
        done |= ok
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    return out.reshape(size, size, 3), R, K
