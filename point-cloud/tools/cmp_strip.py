"""사진 띠와 모델 띠를 위아래로 붙인 내부 점검 그림 (front 방위 = heading + 보정 가정). 사진이 들어가므로 private/ 에만 둔다."""
import os, sys, json
import numpy as np
from PIL import Image

rc = sys.argv[1]; pid = sys.argv[2]; off = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
m = np.load(os.path.join(rc, f'model_{pid}.npz'))
photo = np.load(os.path.join(rc, 'private', f'photo_{pid}.npy'))
info = json.load(open(os.path.join(rc, 'render_check.json')))[pid]
step = 0.5
az_front = info['heading_deg'] + off
# 사진 띠 열 j = 상대 방위 AZ[j]; 같은 열에 세계 방위 az_front + AZ[j] 의 모델 값을 놓는다
AZ = m['az']
k = np.round(((az_front + AZ + 180) % 360 - 180 - AZ[0]) / step).astype(int) % len(AZ)
t = m['t'][:, k]
vis = np.zeros(t.shape + (3,), np.uint8)
fin = np.isfinite(t)
d = np.clip(np.log(np.where(fin, t, 1)) / np.log(400), 0, 1)
vis[fin] = (255 * (1 - d[fin]))[:, None].astype(np.uint8) * np.array([1, 1, 1], np.uint8)
grp = m['grp'][:, k]
vis[grp == 0] = (vis[grp == 0] * np.array([1.0, 0.6, 0.6])).astype(np.uint8)   # 단지 동은 붉게
vis[~fin] = (40, 80, 160)
img = np.concatenate([photo, vis], 0)
Image.fromarray(img).save(os.path.join(rc, 'private', f'cmp_{pid}_off{off:+.0f}.png'))
print('saved')
