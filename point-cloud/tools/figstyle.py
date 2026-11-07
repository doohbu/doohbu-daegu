"""보고서 그림 공용 스타일 (dataviz 기준 팔레트: 범주 고정 순서, 연속 = 파랑 한 색조, 부호 = 파랑 ↔ 빨강 + 회색 가운데)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
for _f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e3df'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # 1 파랑 · 2 주황 · 3 청록 · 4 노랑 (이 순서로만)
NEUTRAL = '#9a9994'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_br', ['#1c5cab', '#86b6ef', '#f0efec', '#f0a09f', '#c03535'])


def setup():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 2,
        'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'axes.unicode_minus': False,
    })
