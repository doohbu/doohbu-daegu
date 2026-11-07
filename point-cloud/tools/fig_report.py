"""보고서 그림 (형상 · 수치만, 사진 없음). 사용: ~/apt-recon/venv/bin/python -I fig_report.py <그림 이름> [인자 …]
- input <out.png> <이름=align.json> …        : 입력 구성별 VGGT 덩어리 수치 (작은 그림 4개, 축 하나씩)
- overview <out.png> <fused.npz> <vworld_outlines.json> <cams.npz> [제목] : 위에서 본 점구름 (높이 = 파랑 한 색조) + V-World 윤곽 + 카메라
- heights <out.png> <facade_vggt.json> [<facade_sfm.json>] : 동별 높이 (VGGT · COLMAP · v5 규칙 · 대장)
- offsets <out.png> <facade.json> <vworld_outlines.json> [제목] : 변별 입면 자리 (부호 = 파랑 ↔ 빨강)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

S.setup()
cmd = sys.argv[1]


def outlines(ax, vw, lo=None, hi=None, label_towers=True, lw=0.9):
    segs = []
    for c in vw['context']:
        r = np.asarray(c['xy'])
        if lo is not None and not (((r >= lo) & (r <= hi)).all(1).any()):
            continue
        segs.append(r)
    ax.add_collection(LineCollection(segs, colors='#c9c8c3', linewidths=0.4))
    for k, v in vw['towers'].items():
        if not k.isdigit() or int(k) < 100:
            continue
        for r in v['rings']:
            r = np.asarray(r); ax.plot(r[:, 0], r[:, 1], color=S.INK, lw=lw, zorder=3)
        if label_towers:
            cxy = np.asarray(v['rings'][0]).mean(0)
            ax.text(cxy[0], cxy[1], k, fontsize=7, ha='center', va='center', color=S.INK2, zorder=4)


if cmd == 'input':
    out = sys.argv[2]; items = [a.split('=', 1) for a in sys.argv[3:]]
    rows = []
    for name, f in items:
        d = json.load(open(f)); ch = d['chunks']
        rows.append({'name': name, 'gps': np.median([c['gps_res_m']['median'] for c in ch]),
                     'spread': np.median([c['rig_center_spread_m']['median'] for c in ch]),
                     'rot': np.median([c['rig_rot_err_deg']['median'] for c in ch]),
                     'between': d['between_chunks_center_diff_m']['median'],
                     'good': sum(1 for c in ch if c['gps_res_m']['median'] < 2 and c['rig_rot_err_deg']['max'] < 15) / len(ch) * 100})
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.2))
    specs = [('gps', 'GPS 차 중앙값 (m)'), ('spread', '한 파노라마 뷰 중심 퍼짐 (m)'), ('rot', '뷰 회전 오차 중앙값 (°)'),
             ('between', '덩어리 사이 같은 파노라마 차 (m)'), ('good', '쓸 만한 덩어리 (%)')]
    x = np.arange(len(rows))
    for ax, (k, t) in zip(axs, specs):
        v = [r[k] for r in rows]
        cols = [S.CAT[0] if i == len(rows) - 1 else '#9ec5f4' for i in range(len(rows))]
        ax.bar(x, v, color=cols, width=0.62)
        for xi, vi in zip(x, v):
            ax.text(xi, vi, f'{vi:.1f}' if k != 'good' else f'{vi:.0f}', ha='center', va='bottom', fontsize=8, color=S.INK)
        ax.set_xticks(x); ax.set_xticklabels([r['name'] for r in rows], fontsize=7.5)
        ax.set_title(t, fontsize=9); ax.grid(axis='x', visible=False)
    fig.suptitle('VGGT 입력 구성 비교 (같은 파노라마 33곳, 덩어리마다 GPS 로 맞춤; 진한 막대 = 고른 구성)', fontsize=10, color=S.INK)
    plt.tight_layout(); plt.savefig(out, dpi=130)

elif cmd == 'overview':
    out, fp, vwp, camp = sys.argv[2:6]; title = sys.argv[6] if len(sys.argv) > 6 else ''
    a = np.load(fp); P = a['P']; vw = json.load(open(vwp)); cams = np.load(camp)
    C = cams['pano_C'] if 'pano_C' in cams.files else cams['C']
    rng = np.random.default_rng(0)
    if len(P) > 1_500_000:
        P = P[rng.choice(len(P), 1_500_000, replace=False)]
    lo = C[:, :2].min(0) - 40; hi = C[:, :2].max(0) + 40
    fig, ax = plt.subplots(figsize=(11, 10))
    zz = P[:, 2]; o = np.argsort(zz)
    im = ax.scatter(P[o, 0], P[o, 1], c=zz[o], s=0.08, cmap=S.SEQ, vmin=np.percentile(zz, 2), vmax=np.percentile(zz, 99), linewidths=0, rasterized=True)
    outlines(ax, vw, lo, hi)
    ax.plot(C[:, 0], C[:, 1], 'o', ms=2.2, color=S.CAT[1], label=f'파노라마 자리 ({len(C)}곳)', zorder=5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
    ax.set_xlabel('동쪽 (m, v5 지역 좌표)'); ax.set_ylabel('북쪽 (m)'); ax.legend(loc='lower left', fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label('높이 z (m, v5)')
    ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=130)

elif cmd == 'heights':
    # 사용: heights <out.png> <facade_vggt.json> [<facade_mesh.json>] [<facade_sfm.json>]
    out = sys.argv[2]; fv = json.load(open(sys.argv[3]))
    fm = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != '-' else None
    fs = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    ks = sorted(set(fv['towers']) | ({'S101', 'S102'} & set((fm or {}).get('towers', {}))), key=lambda q: (q.startswith('S'), q))
    ref = {}
    for src in (fv, fm, fs):
        if src:
            for k, v in src['towers'].items():
                ref.setdefault(k, (v['ref_height_m'], v['ref_is_registered']))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(ks))
    ax.scatter([i for i, k in enumerate(ks) if not ref[k][1]], [ref[k][0] for k in ks if not ref[k][1]], marker='_', s=300, color=S.NEUTRAL, lw=3,
               label='v5 높이 (층수 × 2.9 m + 1.5 m [규칙])', zorder=2)
    ax.scatter([i for i, k in enumerate(ks) if ref[k][1]], [ref[k][0] for k in ks if ref[k][1]], marker='D', s=50, color=S.CAT[2],
               label='건축물대장 높이 (상도현대 2003)', zorder=3)

    def put(src, key, dx, col, lab, mk='o'):
        xs, ys = [], []
        for i, k in enumerate(ks):
            v = (src or {}).get('towers', {}).get(k, {}).get(key)
            if v is not None:
                xs.append(i + dx); ys.append(v)
        ax.scatter(xs, ys, s=36, color=col, label=lab, zorder=4, marker=mk, edgecolors=S.SURFACE, linewidths=1.2)
    put(fv, 'height_p995_m', -0.12, S.CAT[0], 'VGGT 입면 점 높이 99.5 분위')
    put(fm, 'height_p995_m', +0.12, S.CAT[1], '네이버 메시 높이 99.5 분위')
    if fs:
        put(fs, 'height_p995_m', 0.0, S.CAT[3], 'COLMAP 점 높이 99.5 분위', mk='s')
    ax.set_xticks(x); ax.set_xticklabels([k.replace('S', '상도\n') for k in ks], fontsize=7.5)
    ax.set_ylabel('벽 앞 땅에서 높이 (m)'); ax.set_ylim(10, 56); ax.legend(loc='lower left', fontsize=8, ncol=3)
    ax.set_title('동별 높이: 사진 · 메시로 잰 꼭대기 (99.5 분위) 와 v5 규칙 · 대장 — 검증 동 (상도)은 윗층이 비어 낮게 나온다', fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=130)

elif cmd == 'offsets':
    out, fp, vwp = sys.argv[2:5]; title = sys.argv[5] if len(sys.argv) > 5 else ''
    f = json.load(open(fp)); vw = json.load(open(vwp))
    seg = [s for s in f['segments'] if not s['tower'].startswith('S')]
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]
    mids = np.array([s['mid'] for s in seg])
    lo = mids.min(0) - 45; hi = mids.max(0) + 45
    outlines(ax, vw, lo, hi, lw=0.7)
    for s in seg:
        m = np.array(s['mid']); n = np.array(s['n_dir']); v = s['offset_m']
        col = S.DIV(np.clip((v + 4) / 8, 0, 1))
        ax.plot([m[0], m[0] + n[0] * v * 3], [m[1], m[1] + n[1] * v * 3], color=col, lw=3, solid_capstyle='round', zorder=5)
        ax.plot(m[0], m[1], 'o', ms=3.5, color=col, zorder=6)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
    ax.set_xlabel('동쪽 (m)'); ax.set_ylabel('북쪽 (m)')
    ax.set_title(title + '\n선 = 입면 자리 − V-World 윤곽 (3배로 그림), 빨강 = 밖 · 파랑 = 안', fontsize=9)
    off = np.array([s['offset_m'] for s in seg])
    axs[1].hist(off, bins=np.arange(-5, 5.25, 0.5), color=S.CAT[0], rwidth=0.88)
    axs[1].axvline(0, color=S.INK2, lw=1)
    axs[1].set_xlabel('입면 자리 − 윤곽 (m, 밖 +), 길이 12 m 넘는 벽'); axs[1].set_ylabel('벽 수')
    a_ = f['all']; axs[1].set_title(f"벽 {a_['segments']}개: |차| 중앙값 {a_['offset_abs_median_m']} m, 1 m 안 {a_['offset_within_1m']*100:.0f} %, 2 m 안 {a_['offset_within_2m']*100:.0f} %", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=130)
print('저장', sys.argv[2])

if cmd == 'walls':
    # 사용: walls <out.png> <wall_table.csv>
    import csv
    out, tp = sys.argv[2], sys.argv[3]
    rows = list(csv.DictReader(open(tp, encoding='utf-8')))
    f = lambda r, k: float(r[k]) if r[k] not in ('', 'None') else None
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (a, b, la, lb) in zip(axs, (('COLMAP_m', '메시_m', 'COLMAP 점', '네이버 메시 (COLMAP 자세)'), ('COLMAP_m', 'VGGT_m', 'COLMAP 점', 'VGGT 점'))):
        pts = [(f(r, a), f(r, b)) for r in rows if f(r, a) is not None and f(r, b) is not None]
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        ax.fill_between([-6, 6], [-7, 5], [-5, 7], color='#e4e3df', alpha=0.6, lw=0, label='±1 m 띠')
        ax.plot([-6, 6], [-6, 6], color=S.INK2, lw=1)
        ax.scatter(x, y, s=30, color=S.CAT[0], edgecolors=S.SURFACE, linewidths=1.2, zorder=3, label=f'벽 {len(x)}개')
        d = np.abs(x - y)
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal')
        ax.set_xlabel(f'{la}: 입면 자리 − 윤곽 (m)'); ax.set_ylabel(f'{lb} (m)')
        ax.set_title(f'{la} 대 {lb}: |차| 중앙값 {np.median(d):.2f} m, 1 m 안 {np.mean(d <= 1) * 100:.0f} %', fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=130); print('저장', out)
