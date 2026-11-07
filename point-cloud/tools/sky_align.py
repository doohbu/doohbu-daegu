"""사진 하늘 마스크 ↔ 모델 하늘 (광선이 아무것도 안 맞음) 을 방위 보정값마다 견줘, front 면 방위 보정값을 찾는다.

사용: ~/apt-recon/venv/bin/python -I sky_align.py <rc 폴더> pano_id ...
- 사진 하늘 [규칙]: 파란 하늘 (B 가 가장 크고 채도 > 0.12, 밝기 > 0.45) 또는 흰 하늘 (밝기 > 0.72, 채도 < 0.14), 고도 0° 위, 맨 위 줄과 이어진 덩어리.
- 점수: 고도 0–80° 띠의 하늘 IoU. 보정 −40..+40° 를 0.5° 칸으로.
- 하늘선 차: 열마다 위에서 처음 하늘이 아닌 고도. 모델 − 사진 의 중앙값 (나무 · 전선이 사진 쪽을 올린다).
"""
import os, sys, json
import numpy as np
from scipy import ndimage

rc = sys.argv[1]
info = json.load(open(os.path.join(rc, 'render_check.json')))
out = {}
for pid in sys.argv[2:]:
    m = np.load(os.path.join(rc, f'model_{pid}.npz'))
    ph = np.load(os.path.join(rc, 'private', f'photo_{pid}.npy')).astype(np.float32) / 255
    EL = m['el']; AZ = m['az']; step = 0.5
    R, G, B = ph[..., 0], ph[..., 1], ph[..., 2]
    v = ph.max(-1); s = (v - ph.min(-1)) / np.maximum(v, 1e-6)
    blue = (B >= G) & (B >= R) & (s > 0.12) & (v > 0.45)
    white = (v > 0.72) & (s < 0.14)
    cand = (blue | white) & (EL[:, None] > 0)
    lab, n = ndimage.label(cand)
    top = set(np.unique(lab[0][lab[0] > 0]))
    sky_p = np.isin(lab, list(top))
    upper = EL > 0
    sky_m_all = ~np.isfinite(m['t'])
    best = None; curve = []
    for off in np.arange(-40, 40.01, step):
        az_front = info[pid]['heading_deg'] + off
        k = np.round(((az_front + AZ + 180) % 360 - 180 - AZ[0]) / step).astype(int) % len(AZ)
        sm = sky_m_all[:, k]
        a = sky_p[upper]; b = sm[upper]
        iou = (a & b).sum() / max((a | b).sum(), 1)
        curve.append((float(off), float(iou)))
        if best is None or iou > best[1]:
            best = (float(off), float(iou), k)
    off, iou, k = best
    sm = sky_m_all[:, k]
    # 하늘선: 열마다 위에서부터 처음 '하늘 아님' 고도
    def skyline(sky):
        ns = ~sky & upper[:, None]
        idx = np.argmax(ns, 0)
        has = ns.any(0)
        return np.where(has, EL[idx], np.nan)
    sl_p = skyline(sky_p); sl_m = skyline(sm)
    d = sl_m - sl_p
    iou0 = dict(curve)[0.0]
    out[pid] = {'best_offset_deg': off, 'iou_best': round(iou, 3), 'iou_at_0': round(iou0, 3),
                'sky_frac_photo_upper': round(float(sky_p[upper].mean()), 3),
                'sky_frac_model_upper': round(float(sm[upper].mean()), 3),
                'skyline_model_minus_photo_deg_median': round(float(np.nanmedian(d)), 2),
                'skyline_abs_lt3deg_frac': round(float(np.nanmean(np.abs(d) < 3)), 3)}
    top5 = sorted(curve, key=lambda c: -c[1])[:5]
    out[pid]['top5'] = top5
    print(pid, out[pid])
json.dump(out, open(os.path.join(rc, 'sky_align.json'), 'w'), indent=1)
