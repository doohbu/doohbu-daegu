"""뷰어 화면 점검 (내부 확인용, 사진 색이 들어가므로 work/viewer/ 에만 둔다). 사용: ~/apt-recon/webcheck/bin/python viewer_shot.py <url> <out 접두>"""
import sys, time
from playwright.sync_api import sync_playwright
url, pre = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    br = p.chromium.launch(args=["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"])
    pg = br.new_page(viewport={'width': 1600, 'height': 900})
    logs = []
    pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}'[:200]))
    t = time.time(); pg.goto(url); pg.wait_for_timeout(30000)
    pg.screenshot(path=f'{pre}_oblique.png')
    for lab, nm in (('위에서 (전체)', 'top'), ('눈높이 (120 · 123동 사이 길, 북서쪽)', 'eye')):
        try:
            pg.get_by_text(lab).first.click(); pg.wait_for_timeout(6000); pg.screenshot(path=f'{pre}_{nm}.png')
        except Exception as e:
            logs.append(f'버튼 {lab}: {e}'[:200])
    try:
        pg.get_by_text('사진 색').first.click(); pg.wait_for_timeout(800); pg.get_by_text('높이 색').last.click(); pg.wait_for_timeout(5000)
        pg.get_by_text('위에서 (전체)').first.click(); pg.wait_for_timeout(5000); pg.screenshot(path=f'{pre}_top_height.png')
    except Exception as e:
        logs.append(f'색 바꾸기: {e}'[:200])
    print(f'{time.time() - t:.0f}초'); print('\n'.join(l for l in logs if 'error' in l.lower())[:2000])
    br.close()
