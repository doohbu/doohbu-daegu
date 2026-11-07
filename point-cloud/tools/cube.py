"""큐브맵 면 ↔ 원근 카메라 · 지역 좌표 공용 함수 (네이버 거리뷰 큐브맵, face_check.py 로 확인한 표준 십자 배치).

좌표계
- 세계 = v5 지역 좌표: x 동쪽, y 북쪽, z 위 (m). 원점 = v5 scene.json info.origin (경도, 위도).
- 카메라 = OpenCV/COLMAP: x 오른쪽, y 아래, z 앞. 면 화각 90°, f = W/2, 주점 = (W/2, H/2) (화소 모서리 기준).
- 파노라마 틀: front 면 방위 = az_front (북 0°, 시계 방향). right = +90°, back = +180°, left = +270°.
  top: 앞 = 위, 영상 아래쪽 = front 방위. bottom: 앞 = 아래, 영상 위쪽 = front 방위 (face_check 로 확인).
"""
import math
import numpy as np

FACES = ['front', 'right', 'back', 'left', 'top', 'bottom']
A_WGS84 = 6378137.0
E2_WGS84 = 6.69437999014e-3
ORIGIN = (126.95936, 37.492375)   # v5 info.origin (경도, 위도)


def meter_scale(lat0):
    phi = math.radians(lat0)
    s = math.sin(phi)
    M = A_WGS84 * (1 - E2_WGS84) / (1 - E2_WGS84 * s * s) ** 1.5
    N = A_WGS84 / math.sqrt(1 - E2_WGS84 * s * s)
    return math.radians(1.0) * N * math.cos(phi), math.radians(1.0) * M


def local_xy(lon, lat, origin=ORIGIN):
    """경위도 → v5 지역 (동 m, 북 m). lightlab.apt3d.vworld.local_meters 와 같은 식."""
    kx, ky = meter_scale(origin[1])
    return (np.asarray(lon) - origin[0]) * kx, (np.asarray(lat) - origin[1]) * ky


def lonlat_from_xy(x, y, origin=ORIGIN):
    kx, ky = meter_scale(origin[1])
    return origin[0] + np.asarray(x) / kx, origin[1] + np.asarray(y) / ky


def horiz(az_deg):
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a), 0.0])


def face_R(face, az_front_deg):
    """세계 → 카메라 회전 (행 = 카메라 x, y, z 축의 세계 방향)."""
    up = np.array([0.0, 0.0, 1.0])
    if face in ('front', 'right', 'back', 'left'):
        a = az_front_deg + 90.0 * FACES.index(face)
        z = horiz(a)
        y = -up
    elif face == 'top':
        z = up
        y = horiz(az_front_deg)
    elif face == 'bottom':
        z = -up
        y = -horiz(az_front_deg)
    else:
        raise ValueError(face)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def K(W, H=None):
    H = H or W
    return np.array([[W / 2.0, 0, W / 2.0], [0, H / 2.0, H / 2.0], [0, 0, 1.0]])


def pixel_rays(face, az_front_deg, n):
    """n×n 화소 가운데의 세계 방향 (n,n,3), 단위 벡터."""
    u = (np.arange(n) + 0.5) / n * 2 - 1           # -1..1
    uu, vv = np.meshgrid(u, u)
    d_cam = np.stack([uu, vv, np.ones_like(uu)], -1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    R = face_R(face, az_front_deg)
    return d_cam @ R                                 # 카메라 → 세계 = R^T d


def dirs_equirect(az_deg, el_deg):
    """세계 방위 · 고도각 격자 → 방향 (…,3)."""
    a = np.radians(az_deg); e = np.radians(el_deg)
    return np.stack([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)], -1)


def sample_cube(faces, dirs, az_front_deg):
    """faces: {이름: (H,W[,C]) 배열}. dirs: (…,3) 세계 방향. 가장 가까운 화소 값."""
    shp = dirs.shape[:-1]
    d = dirs.reshape(-1, 3)
    any_face = next(iter(faces.values()))
    out = np.zeros((d.shape[0],) + any_face.shape[2:], any_face.dtype)
    done = np.zeros(d.shape[0], bool)
    for name, img in faces.items():
        H, W = img.shape[:2]
        p = d @ face_R(name, az_front_deg).T
        ok = (p[:, 2] > 1e-9) & ~done
        x = np.where(ok, p[:, 0] / np.maximum(p[:, 2], 1e-9), 9); y = np.where(ok, p[:, 1] / np.maximum(p[:, 2], 1e-9), 9)
        ok &= (np.abs(x) <= 1) & (np.abs(y) <= 1)
        u = np.clip(((x[ok] + 1) / 2 * W).astype(int), 0, W - 1)
        v = np.clip(((y[ok] + 1) / 2 * H).astype(int), 0, H - 1)
        out[ok] = img[v, u]
        done |= ok
    return out.reshape(shp + any_face.shape[2:])
