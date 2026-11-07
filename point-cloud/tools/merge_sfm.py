"""SfM 모델 여러 개 (조각) 를 조각마다 eval_sfm.py 로 맞춘 뒤 (중력 + GPS), 점 · 카메라를 하나로 모은다.
사용: ~/apt-recon/venv/bin/python -I merge_sfm.py <sfm 폴더> <out.npz> [--min-frames 5] [--max-gps 3.0]
- 조각마다 eval_<id>.json · aligned_<id>.npz 가 있어야 한다 (eval_sfm.py 를 먼저 돌린다).
- 넣는 조건: 등록 틀 ≥ --min-frames 이고 GPS 차 중앙값 ≤ --max-gps m (틀린 조각을 거른다).
- 같은 파노라마가 두 조각에 있으면 틀 수가 많은 조각의 것을 쓴다 (점은 다 넣는다).
"""
import glob, json, os, sys
import numpy as np

d, outp = sys.argv[1], sys.argv[2]
mf = int(sys.argv[sys.argv.index('--min-frames') + 1]) if '--min-frames' in sys.argv else 5
mg = float(sys.argv[sys.argv.index('--max-gps') + 1]) if '--max-gps' in sys.argv else 3.0
evs = []
for f in glob.glob(os.path.join(d, 'eval_*.json')):
    e = json.load(open(f)); mid = str(e['model_id'])
    evs.append((e['frames_registered_front'], mid, e))
evs.sort(key=lambda t: -t[0])
P, T, C, pids, used, skipped = [], [], [], [], [], []
seen = set()
for nf, mid, e in evs:
    if nf < mf or e['gps_residual_m']['median'] > mg:
        skipped.append({'model': mid, 'frames': nf, 'gps_median': e['gps_residual_m']['median']}); continue
    a = np.load(os.path.join(d, f'aligned_{mid}.npz'))
    P.append(a['P']); T.append(a['track'])
    for c, p in zip(a['C'], a['pids']):
        if str(p) not in seen:
            seen.add(str(p)); C.append(c); pids.append(str(p))
    used.append({'model': mid, 'frames': nf, 'gps_median': e['gps_residual_m']['median'], 'points': int(len(a['P'])),
                 'reproj_px': e['mean_reproj_px']})
np.savez_compressed(outp, P=np.concatenate(P), track=np.concatenate(T), C=np.array(C), pids=np.array(pids))
json.dump({'used': used, 'skipped': skipped, 'frames_total': len(pids)}, open(outp.replace('.npz', '.json'), 'w'), indent=1)
print(json.dumps({'used': used, 'skipped': skipped[:10], 'frames_total': len(pids)}, ensure_ascii=False))
