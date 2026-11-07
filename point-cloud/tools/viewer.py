"""청림동 거리뷰 복원 3D 뷰어 (viser). 내부 전용 — 사진 색 점은 네이버 거리뷰 (© NAVER Corp., 비공식 경로) 에서 나왔다.
사용: venv/bin/python -I viewer.py [--host 0.0.0.0] [--port 8793]
자료: work/viewer/*.npz (viewer_prep.py). 좌표 = v5 지역 좌표 (x 동 · y 북 · z 위, m).
"""
import os, sys, time
import numpy as np
import viser

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'work', 'viewer')
host = sys.argv[sys.argv.index('--host') + 1] if '--host' in sys.argv else '0.0.0.0'
port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8793
RAMP = np.array([[0xcd, 0xe2, 0xfb], [0x86, 0xb6, 0xef], [0x39, 0x87, 0xe5], [0x1c, 0x5c, 0xab], [0x0d, 0x36, 0x6b]], np.float32)


def height_colors(z):
    lo, hi = np.percentile(z, 2), np.percentile(z, 99)
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2); f = (t - i)[:, None]
    return (RAMP[i] * (1 - f) + RAMP[i + 1] * f).astype(np.uint8)


vg = np.load(os.path.join(D, 'vggt.npz')); cm = np.load(os.path.join(D, 'colmap.npz')); me = np.load(os.path.join(D, 'mesh.npz'))
ln = np.load(os.path.join(D, 'lines.npz')); ca = np.load(os.path.join(D, 'cams.npz'))
vP_all, vC_all = vg['P'], vg['C']
# 하늘 점 [규칙]: 밝고 (최대 성분 > 150) 파란 쪽 (B > R + 15, B ≥ G) 이며 땅 (점 z 5 분위) 에서 20 m 넘게 위인 점 — VGGT 가 하늘 화소에 준 깊이
_z0 = np.percentile(vP_all[:, 2], 5)
_c = vC_all.astype(np.int16)
SKY = (_c.max(1) > 150) & (_c[:, 2] > _c[:, 0] + 15) & (_c[:, 2] >= _c[:, 1]) & (vP_all[:, 2] > _z0 + 20)
vH_all = height_colors(vP_all[:, 2])
vP, vC, vH = vP_all[~SKY], vC_all[~SKY], vH_all[~SKY]
server = viser.ViserServer(host=host, port=port, label='청림동 거리뷰 복원 (내부 전용)')
server.scene.set_up_direction('+z')
server.gui.configure_theme(dark_mode=False, show_logo=False, show_share_button=False)   # 공유 단추 끔: 외부 중계 공개 주소를 만들 수 있어서 (내부 전용)
sc = server.scene
h_vggt = sc.add_point_cloud('/vggt', vP, vC, point_size=0.12, precision='float32', point_shape='square')
h_colmap = sc.add_point_cloud('/colmap', cm['P'], cm['C'], point_size=0.18, precision='float32', visible=False)
h_mesh = sc.add_point_cloud('/mesh', me['P'], (235, 104, 52), point_size=0.1, precision='float32', visible=False)
h_tw = sc.add_line_segments('/outline_v5box', ln['towers'], (20, 20, 20), thickness=2.0, thickness_units='screen')
h_ctx = sc.add_line_segments('/outline_context', ln['context'], (160, 160, 155), thickness=1.0, thickness_units='screen', visible=False)
labels = [sc.add_label(f'/label/{k}', f'{k}동', position=tuple(float(v) for v in p), font_screen_scale=0.9) for k, p in zip(ln['labels'], ln['label_pos'])]
h_cv = sc.add_point_cloud('/cams_vggt', ca['vggt_C'], (235, 104, 52), point_size=0.5, point_shape='circle', precision='float32')
h_cc = sc.add_point_cloud('/cams_colmap', ca['colmap_C'], (27, 175, 122), point_size=0.5, point_shape='circle', precision='float32', visible=False)
# 파노라마 앞 방향 (VGGT 회전) 짧은 선
fw = np.einsum('nji,j->ni', ca['vggt_R'], np.array([0, 1.0, 0]))
h_fw = sc.add_line_segments('/cams_vggt_front', np.stack([ca['vggt_C'], ca['vggt_C'] + 4 * fw], 1).astype(np.float32), (235, 104, 52), thickness=1.5, thickness_units='screen')

gui = server.gui
gui.add_markdown('**청림동 현대아파트 둘레 — 네이버 거리뷰 복원 (내부 전용)**\n\n'
                 'VGGT-1B 피드포워드 (파노라마 306곳, 점 4,476만 개 중 300만 개) · COLMAP SfM (309곳) · 네이버 3D 메시.\n\n'
                 '좌표: v5 지역 좌표 (m). 검정 선 = V-World 동 윤곽을 v5 바닥 · v5 꼭대기 높이로 세운 상자.\n\n'
                 '사진 색은 네이버 지도 거리뷰 (© NAVER Corp.) 에서 왔다. 외부에 올리지 않는다.')
with gui.add_folder('보이기'):
    cb_v = gui.add_checkbox('VGGT 점', True); dd = gui.add_dropdown('VGGT 색', ['사진 색', '높이 색'], initial_value='사진 색')
    cb_sky = gui.add_checkbox('하늘 점 빼기', True, hint=f'밝은 하늘색 + 땅 20 m 위 [규칙], {int(SKY.sum()):,}개')
    cb_c = gui.add_checkbox('COLMAP 점 (사진 색)', False); cb_m = gui.add_checkbox('네이버 메시 점 (주황)', False)
    cb_o = gui.add_checkbox('동 윤곽 · v5 높이 상자', True); cb_l = gui.add_checkbox('동 번호', True); cb_x = gui.add_checkbox('주변 건물 윤곽 (땅 높이 0)', False)
    cb_cv = gui.add_checkbox('파노라마 자리 (VGGT, 주황 + 앞 방향)', True); cb_cc = gui.add_checkbox('파노라마 자리 (COLMAP, 청록)', False)
with gui.add_folder('점 크기'):
    sl_v = gui.add_slider('VGGT', 0.02, 0.6, 0.02, 0.12); sl_c = gui.add_slider('COLMAP', 0.05, 1.0, 0.05, 0.18)
    sl_m = gui.add_slider('메시', 0.02, 0.6, 0.02, 0.1)
with gui.add_folder('시점'):
    b_top = gui.add_button('위에서 (전체)'); b_obl = gui.add_button('비스듬히 (남서쪽에서)'); b_eye = gui.add_button('눈높이 (120 · 123동 사이 길, 북서쪽)')


def bind(cb, h):
    @cb.on_update
    def _(_):
        h.visible = cb.value


for cb, h in ((cb_v, h_vggt), (cb_c, h_colmap), (cb_m, h_mesh), (cb_o, h_tw), (cb_x, h_ctx), (cb_cv, h_cv), (cb_cc, h_cc)):
    bind(cb, h)


@cb_cv.on_update
def _(_):
    h_fw.visible = cb_cv.value


@cb_l.on_update
def _(_):
    for l in labels:
        l.visible = cb_l.value


def refresh_vggt():
    keep = ~SKY if cb_sky.value else np.ones(len(vP_all), bool)
    h_vggt.points = vP_all[keep]
    h_vggt.colors = (vC_all if dd.value == '사진 색' else vH_all)[keep]


@dd.on_update
def _(_):
    refresh_vggt()


@cb_sky.on_update
def _(_):
    refresh_vggt()


@sl_v.on_update
def _(_):
    h_vggt.point_size = sl_v.value


@sl_c.on_update
def _(_):
    h_colmap.point_size = sl_c.value


@sl_m.on_update
def _(_):
    h_mesh.point_size = sl_m.value


VIEWS = {'top': ((0.0, -40.0, 330.0), (0.0, -20.0, 0.0)), 'obl': ((-170.0, -250.0, 150.0), (0.0, -10.0, 0.0)),
         'eye': ((95.0, -45.0, 4.0), (60.0, 0.0, 15.0))}


def goto(client, k):
    pos, look = VIEWS[k]
    client.camera.position = pos; client.camera.look_at = look; client.camera.up_direction = (0.0, 0.0, 1.0)


for b, k in ((b_top, 'top'), (b_obl, 'obl'), (b_eye, 'eye')):
    def mk(k):
        def _(ev):
            goto(ev.client, k)
        return _
    b.on_click(mk(k))


@server.on_client_connect
def _(client):
    goto(client, 'obl')


print(f'뷰어 준비: http://{host}:{port}  (VGGT {len(vP):,} (하늘 점 {int(SKY.sum()):,} 뺌) · COLMAP {len(cm["P"]):,} · 메시 {len(me["P"]):,} 점)', flush=True)
while True:
    time.sleep(3600)
