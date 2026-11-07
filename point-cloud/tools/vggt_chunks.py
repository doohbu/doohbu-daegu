"""네이버 거리뷰 큐브 면 → VGGT-1B 덩어리 추론 (피드포워드). 덩어리 = 이웃 파노라마 K 곳 × 면 F 장.

사용: venv/bin/python -I vggt_chunks.py --panos <panos> --index <index.csv> --out <폴더> [--bbox x0,y0,x1,y1 | --ids a,b]
        [--faces front,right,back,left,top] [--k 8] [--seed-spacing 20] [--res 518] [--weights weights/VGGT-1B/model.safetensors]
- 면은 res×res 로 줄인다 (2048 → 518, 90° 그대로). 사진은 메모리에만, 결과에는 깊이 · 자세만 저장한다.
- 덩어리: 파노라마 자리 (v5 지역 xy) 에서 가장 먼 점 고르기로 씨앗 (간격 --seed-spacing m) → 씨앗마다 가장 가까운 K 곳.
- 저장: <out>/chunk_XXX.npz — names, pano, face, extrinsic (S,3,4: 덩어리 첫 면 기준 cam_from_world, OpenCV), intrinsic (S,3,3, res 화소),
  depth (S,res,res) float16, conf float16, 그리고 시간 · 최대 GPU 메모리. 덩어리 목록 <out>/chunks.json.
- GPU: 시작 전에 ComfyUI 큐 (127.0.0.1:8189) 가 비었는지 본다. 차 있으면 기다린다 (큐에는 넣지 않는다).
"""
import argparse, csv, json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'third_party', 'vggt'))
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
import cube
import views
import views_gpu
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def comfy_busy():
    try:
        q = json.load(urllib.request.urlopen('http://127.0.0.1:8189/queue', timeout=3))
        return bool(q.get('queue_running') or q.get('queue_pending'))
    except Exception:
        return False


def select(a):
    rows = [r for r in csv.DictReader(open(a.index, encoding='utf-8')) if r.get('폴더')]
    seen, sel = set(), []
    faces = a.faces.split(',')
    for r in rows:
        pid = r['pano_id']
        if pid in seen:
            continue
        x, y = cube.local_xy(float(r['경도']), float(r['위도'])); x, y = float(x), float(y)
        if a.ids and pid not in a.ids.split(','):
            continue
        if a.bbox:
            x0, y0, x1, y1 = map(float, a.bbox.split(','))
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
        if not all(os.path.exists(os.path.join(a.panos, pid, f + '.jpg')) for f in faces):
            continue
        seen.add(pid); sel.append({'pid': pid, 'x': x, 'y': y, 'heading': float(r['방향_deg']), 'date': r['촬영일시_네이버값']})
    return sel


def make_chunks(sel, k, spacing):
    P = np.array([[s['x'], s['y']] for s in sel])
    seeds = [0]
    dmin = np.linalg.norm(P - P[0], axis=1)
    while dmin.max() > spacing:
        i = int(np.argmax(dmin)); seeds.append(i); dmin = np.minimum(dmin, np.linalg.norm(P - P[i], axis=1))
    chunks = []
    for i in seeds:
        d = np.linalg.norm(P - P[i], axis=1)
        chunks.append(sorted(np.argsort(d)[:k].tolist()))
    covered = set(j for c in chunks for j in c)
    for j in range(len(sel)):
        if j not in covered:
            d = np.linalg.norm(P - P[j], axis=1); chunks.append(sorted(np.argsort(d)[:k].tolist()))
    # 같은 덩어리 지우기
    uniq = []
    for c in chunks:
        if c not in uniq:
            uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panos', required=True); ap.add_argument('--index', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bbox'); ap.add_argument('--ids')
    ap.add_argument('--faces', default='front,right,back,left,top')
    ap.add_argument('--k', type=int, default=8); ap.add_argument('--seed-spacing', type=float, default=20.0)
    ap.add_argument('--res', type=int, default=518)
    ap.add_argument('--weights', default=os.path.join(HERE, '..', 'weights', 'VGGT-1B', 'model.safetensors'))
    ap.add_argument('--only', type=int, default=-1, help='이 번호 덩어리만')
    ap.add_argument('--views', default='', help='예: 0:0,45:0,...  (방위:고개, front 기준 °). 주면 큐브 면 대신 다시 그린 뷰를 쓴다')
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--bf16-weights', action='store_true', help='가중치를 bf16 으로 (메모리 절약, 머리는 bf16 autocast)')
    a = ap.parse_args()
    faces = a.faces.split(',')
    vlist = [tuple(map(float, v.split(':'))) for v in a.views.split(',')] if a.views else []
    os.makedirs(a.out, exist_ok=True)
    sel = select(a)
    chunks = make_chunks(sel, a.k, a.seed_spacing)
    json.dump({'sel': sel, 'faces': faces, 'k': a.k, 'seed_spacing': a.seed_spacing, 'res': a.res,
               'chunks': [[sel[j]['pid'] for j in c] for c in chunks]}, open(os.path.join(a.out, 'chunks.json'), 'w'), ensure_ascii=False, indent=1)
    nper = len(vlist) if vlist else len(faces)
    print(f'파노라마 {len(sel)}, 덩어리 {len(chunks)} (K={a.k}, 파노라마마다 {nper}장 → {a.k * nper} 장/덩어리)', flush=True)
    dev = 'cuda'
    model = VGGT(enable_point=False, enable_track=False)
    sd = load_file(a.weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if not (m.startswith('point_head') or m.startswith('track_head'))], missing
    model = model.to(dev).eval()
    if a.bf16_weights:
        model = model.to(torch.bfloat16)
    del sd
    dtype = torch.bfloat16
    cache = {}
    t_all = time.time()
    for ci, c in enumerate(chunks):
        if a.only >= 0 and ci != a.only:
            continue
        outp = os.path.join(a.out, f'chunk_{ci:03d}.npz')
        if os.path.exists(outp):
            continue
        while comfy_busy():
            print('ComfyUI 큐가 차 있어 30초 기다림', flush=True); time.sleep(30)
        names, pids, fcs, imgs, Rv = [], [], [], [], []
        for j in c:
            if vlist:
                pid = sel[j]['pid']
                if pid not in cache:      # 그린 뷰는 메모리에만 (uint8), 파일로 남기지 않는다. GPU 로 그린다 (views_gpu, 4배 크게 → 평균).
                    F = views.load_faces(os.path.join(a.panos, pid), ('front', 'right', 'back', 'left', 'top', 'bottom'))
                    Ft = views_gpu.faces_tensor(F)
                    V = (views_gpu.render_batch(Ft, vlist, a.fov, a.res) * 255).round().byte().cpu()
                    cache[pid] = [(V[i], views.view_R(yw, pt)) for i, (yw, pt) in enumerate(vlist)]
                    del Ft, V
                for (yw, pt), (im, R) in zip(vlist, cache[pid]):
                    imgs.append(im.float() / 255.0)
                    nm = f'v{int(yw):03d}{int(pt):+03d}'
                    names.append(f'{nm}/{sel[j]["pid"]}'); pids.append(sel[j]['pid']); fcs.append(nm); Rv.append(R)
            else:
                for f in faces:
                    im = Image.open(os.path.join(a.panos, sel[j]['pid'], f + '.jpg')).convert('RGB').resize((a.res, a.res), Image.BICUBIC)
                    imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
                    names.append(f'{f}/{sel[j]["pid"]}'); pids.append(sel[j]['pid']); fcs.append(f); Rv.append(cube.face_R(f, 0.0))
        x = torch.stack(imgs).to(dev)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=dtype):
                tok, ps = model.aggregator(x[None])
            with torch.autocast('cuda', dtype=dtype, enabled=a.bf16_weights):
                pose = model.camera_head(tok)[-1].float()
                ext, intr = pose_encoding_to_extri_intri(pose, x.shape[-2:])
                dep, conf = model.depth_head(tok, images=x[None], patch_start_idx=ps)
        torch.cuda.synchronize(); dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        np.savez_compressed(outp, names=np.array(names), pano=np.array(pids), face=np.array(fcs),
                            extrinsic=ext[0].float().cpu().numpy(), intrinsic=intr[0].float().cpu().numpy(),
                            depth=dep[0, ..., 0].float().cpu().numpy().astype(np.float16),
                            conf=conf[0].float().cpu().numpy().astype(np.float16), seconds=dt, peak_gib=peak,
                            R_view=np.array(Rv), fov_true=a.fov)
        print(f'덩어리 {ci}/{len(chunks)}: {len(names)}장 {dt:.1f}초, 최대 GPU {peak:.2f} GiB, 누적 {time.time() - t_all:.0f}초', flush=True)
        del tok, pose, ext, intr, dep, conf, x
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
