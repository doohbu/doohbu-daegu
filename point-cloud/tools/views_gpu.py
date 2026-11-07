"""views.render 의 GPU 판 (torch grid_sample). 같은 규약 (views.view_R · cube.face_R), ss 배 크게 그린 뒤 평균으로 줄인다."""
import math
import numpy as np
import torch
import torch.nn.functional as Fn
import cube
import views

FACE_NAMES = ('front', 'right', 'back', 'left', 'top', 'bottom')
FR = torch.tensor(np.stack([cube.face_R(n, 0.0) for n in FACE_NAMES]), dtype=torch.float32)   # (6,3,3) 세계 → 면 카메라


def faces_tensor(faces_np, dev='cuda'):
    """{이름: (H,W,3) uint8/float} → (6,3,H,W) float16 GPU."""
    return torch.stack([torch.from_numpy(np.ascontiguousarray(faces_np[n])).permute(2, 0, 1) for n in FACE_NAMES]).to(dev).half() / 255.0


def render_batch(Ft, vlist, fov=90.0, size=518, ss=4):
    dev = Ft.device
    S = size * ss
    f = (S / 2) / math.tan(math.radians(fov) / 2)
    u = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5 - S / 2) / f
    vv, uu = torch.meshgrid(u, u, indexing='ij')
    d_cam = torch.stack([uu, vv, torch.ones_like(uu)], -1).reshape(-1, 3)          # (S*S,3)
    out = []
    FRd = FR.to(dev)
    for (yw, pt) in vlist:
        Rv = torch.tensor(views.view_R(yw, pt), dtype=torch.float32, device=dev)
        d = d_cam @ Rv                                                              # 파노라마 틀 방향
        p = torch.einsum('fij,nj->fni', FRd, d)                                     # (6,N,3) 면 카메라 좌표
        z = p[..., 2]
        x = p[..., 0] / z.clamp(min=1e-9); y = p[..., 1] / z.clamp(min=1e-9)
        ok = (z > 1e-9) & (x.abs() <= 1) & (y.abs() <= 1)
        fid = torch.argmax(ok.float() * (z + 2), 0)                                 # 맞는 면 중 아무거나 (경계는 같은 값)
        img = torch.zeros(3, S * S, device=dev, dtype=torch.float16)
        for k in range(6):
            m = fid == k
            if not m.any():
                continue
            grid = torch.stack([x[k, m], y[k, m]], -1)[None, None].half()           # (1,1,M,2), align_corners=False 규약 (-1..1 = 화소 가장자리)
            smp = Fn.grid_sample(Ft[k:k + 1], grid, mode='bilinear', padding_mode='border', align_corners=False)
            img[:, m] = smp[0, :, 0]
        img = img.reshape(1, 3, S, S).float()
        if ss > 1:
            img = Fn.avg_pool2d(img, ss)
        out.append(img[0])
    return torch.stack(out).clamp(0, 1)                                             # (V,3,size,size)
