"""align_*.json 요약 한 줄씩 + 전체 중앙값."""
import json, sys
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); ch = d['chunks']
    g = [c['gps_res_m']['median'] for c in ch]; sp = [c['rig_center_spread_m']['median'] for c in ch]
    rr = [c['rig_rot_err_deg']['median'] for c in ch]; rm = [c['rig_rot_err_deg']['max'] for c in ch]
    lv = [c['leveling_deg']['median'] for c in ch]; dh = [c['heading_minus_naver_deg']['absmax'] for c in ch]
    good = sum(1 for c in ch if c['gps_res_m']['median'] < 2 and c['rig_rot_err_deg']['max'] < 15)
    print(f"{f}: 덩어리 {len(ch)} | GPS차 중앙 {np.median(g):.2f} m (덩어리별 {min(g):.1f}–{max(g):.1f}) | 중심퍼짐 {np.median(sp):.2f} m | "
          f"회전오차 중앙 {np.median(rr):.2f}° 최대 {max(rm):.0f}° | 수평 {np.median(lv):.2f}° | 좋은 덩어리 {good}/{len(ch)} | 덩어리사이 {d['between_chunks_center_diff_m']}")
