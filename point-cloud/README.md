# point-cloud — 네이버 거리뷰로 점구름 3D 복원 (도구 모음)

> **상태: 2026-11-07 스냅샷.** 작업 중인 코드다. 거리뷰 사진 · 복원 결과 · 모델 가중치는 이 저장소에 없다.

## 무엇인가

한 아파트 단지 둘레의 거리뷰 파노라마(큐브맵 6면 JPEG)로 3D 형상을 복원해 보는 실험 도구다.
피드포워드 복원 모델(VGGT)로 점구름을 만들고, 촬영 위치(GPS)로 덩어리를 맞춘다. COLMAP SfM 은 대조 기준으로만 돌린다.
복원한 벽 · 윤곽은 공개 지도 자료의 건물 윤곽과 견줘 본다.

## 이 폴더에 있는 것 · 없는 것

- **있는 것:** `tools/` 아래 파이썬 · 셸 도구 44개 (아래 표).
- **일부러 뺀 것:** 거리뷰 사진, 점구름 · 메시 · 그림 같은 복원 결과, 모델 가중치, 외부 코드(`third_party/`), 가상환경.
  - 거리뷰 사진과 거기서 나온 결과는 올리지 않는다(제3자 서비스의 이용 조건, 얼굴 · 번호판 개인정보 때문). 사진을 쓰려면 먼저 해당 서비스의 이용 조건을 확인할 것.
  - `.gitignore` 가 사진 · 이미지 · 점구름 · 가중치 파일이 실수로 올라가지 않게 막는다.

## 쓰는 것과 라이선스 (2026-11-07 조사 기준, 쓰기 전에 원문을 다시 확인할 것)

- **VGGT** (Meta · Oxford, CVPR 2025): 코드는 VGGT License, 가중치는 CC BY-NC 4.0 (비상업). 가중치는 이 저장소에 없다.
- **COLMAP / pycolmap**: BSD-3.
- `tools/colab/` 의 실험은 TRELLIS 계열 모델을 쓴다. 라이선스는 해당 프로젝트를 따른다.
- 이 저장소의 나머지(원작 SMASH-CAU)는 All rights reserved 이고, 이 폴더의 코드는 그와 무관한 별도 작업이다.

## 도구 (파일마다 머리말 첫 줄)

| 파일 | 설명 |
|---|---|
| `after_sfm.sh` | 전체 SfM 이 끝난 뒤 대조 (6절). 사용: bash after_sfm.sh |
| `cmp_strip.py` | 사진 띠와 모델 띠를 위아래로 붙인 내부 점검 그림 (front 방위 = heading + 보정 가정). 사진이 들어가므로 private/ 에만 둔다. |
| `colab/run_trellis.py` | (설명 없음) |
| `colab/run_trellis.sh` | (설명 없음) |
| `colab/trellis_setup.sh` | Colab VM 에서 TRELLIS.2 환경 만들기 (공식 setup.sh 를 micromamba Python 3.10 · CUDA 12.4 환경 안에서). 기록: /content/setup.log |
| `compare_walls.py` | 두 facade_check 결과의 같은 벽 (동 + 변 가운데 자리) 의 입면 자리를 견준다. |
| `cube.py` | 큐브맵 면 ↔ 원근 카메라 · 지역 좌표 공용 함수 (네이버 거리뷰 큐브맵, face_check.py 로 확인한 표준 십자 배치). |
| `eval_sfm.py` | SfM 결과 점검: 등록 비율 · 재투영 오차 · 중력 (수평) · GPS 와 맞춤 · 방위 차 · 카메라 높이 · v5 모델 면까지 거리. |
| `facade_check.py` | 입면 면 자리 · 높이 점검 (변 단위). 입력: 맞춘 점 npz (P, C), vworld_outlines.json, v5 scene.json. |
| `face_check.py` | 큐브맵 6면의 이음매 점검 — 면 화각이 90° 인지, 위 · 아래 면이 어떤 방향으로 붙는지. |
| `fig_report.py` | 보고서 그림 (형상 · 수치만, 사진 없음). 사용: ~/apt-recon/venv/bin/python -I fig_report.py <그림 이름> [인자 …] |
| `figstyle.py` | 보고서 그림 공용 스타일 (dataviz 기준 팔레트: 범주 고정 순서, 연속 = 파랑 한 색조, 부호 = 파랑 ↔ 빨강 + 회색 가운데). |
| `floor_period.py` | 입면 점의 높이 분포 주기 (층높이) 를 자기상관으로 잰다. facade_check 와 같은 변 · 같은 벽 앞 땅 규칙. |
| `merge_sfm.py` | SfM 모델 여러 개 (조각) 를 조각마다 eval_sfm.py 로 맞춘 뒤 (중력 + GPS), 점 · 카메라를 하나로 모은다. |
| `mvs_fix_workspace.py` | mvs_test 작업 폴더 고치기: (1) 영상 링크 → 회색조 JPEG 사본, (2) patch-match.cfg · fusion.cfg 를 작업 폴더 영상끼리의 짝으로. |
| `mvs_run.py` | 고친 작업 폴더에서 PatchMatch → 합치기만. 사용: venv-mvs/bin/python -I mvs_run.py <작업 폴더> [--max-size 1600] |
| `mvs_test.py` | 시험 구역 밀집 복원 (COLMAP PatchMatch, CUDA). 자세 = 전체 SfM 모델 0 (정밀, 재투영 ~1 px). 사진은 링크만 (복사 안 함). |
| `naver_mesh_check.py` | 네이버 3D 메시 (panos/<id>/model.npz, 수집 세션이 받은 것) 의 축 규약을 정하고, 복원 점과의 거리를 잰다 (독립 형상 대조). |
| `naver_mesh_world.py` | 네이버 3D 메시 → 세계 좌표 점 (면 위 고르게 뽑음). 자세는 복원 (VGGT 또는 SfM) 의 파노라마 자세를 쓴다. |
| `outline_check.py` | 맞춘 SfM 점 ↔ V-World 동 윤곽 (수평 거리) · v5 높이 대조, 그림 (형상만: 점 · 윤곽 · 카메라 자리). |
| `pose_compare.py` | 두 복원의 파노라마 자세 견주기 (같은 v5 좌표에 맞춘 뒤). 사용: python pose_compare.py <A poses npz> <B poses npz> <out.json> |
| `range_ratio.py` | 같은 광선에서 복원 점 거리 ÷ 네이버 메시 거리 (축척 대조). 자세 = poses.npz 의 파노라마 자세, 메시 규약 H2. |
| `range_ratio_sfm.py` | COLMAP 점 거리 ÷ 네이버 메시 거리 (같은 광선). 그 파노라마 면이 본 점 (트랙) 만 쓴다. |
| `remap.py` | 이미 맞춘 database 로 지도 만들기만 다시 (사전값 · 설정 시험용). |
| `render_check.py` | v5 모델을 파노라마 자리에서 360° 로 광선 추적해, 사진과 같은 등장방형 띠로 만든다 (방향 규약 · 자리 점검). |
| `reproj_check.py` | VGGT 자세의 재투영 오차: COLMAP 모델의 점 트랙 (옆면 관측) 을 VGGT 세계 자세 (views_world.npz) 로 다시 삼각측량해 2048 px 면에서 잰다. |
| `run_guarded.sh` | 명령을 nice 10 으로 돌리고 온도 감시를 붙인다. 사용: bash run_guarded.sh <기록 접두> <명령 ...> |
| `run_variants.sh` | 같은 33곳 (work/proto_ids.txt) 으로 VGGT 입력 구성 4가지를 다시 돌려 견준다. |
| `scene_mesh.py` | v5 장면 (meshes.npz + scene.json 물체 행렬) → 세계 좌표 삼각형 하나로 합친 npz. 원본은 읽기만 한다. |
| `sfm_rig.py` | 네이버 거리뷰 큐브맵 → COLMAP SfM (pycolmap 4.2.1, CPU). 한 파노라마 = rig 한 틀 (면 5장, 중심 같음, 상대 회전 고정). |
| `sky_align.py` | 사진 하늘 마스크 ↔ 모델 하늘 (광선이 아무것도 안 맞음) 을 방위 보정값마다 견줘, front 면 방위 보정값을 찾는다. |
| `summ_align.py` | align_*.json 요약 한 줄씩 + 전체 중앙값. |
| `thermal_guard.sh` | CPU 온도 기록 · 과열 멈춤: 5 초마다 x86_pkg_temp 를 적고, 90 °C 이상이면 대상 PID 를 SIGSTOP, 75 °C 아래로 식으면 SIGCONT. |
| `tower_outlines.py` | v5 동 메시를 수평으로 잘라 윤곽 선분을 얻는다 (v5 의 동 벽 = V-World dt_d198 · LT_C_SPBD 윤곽을 세운 것이라, 자른 선 = 그 윤곽 + 모델이 붙인 돌출). |
| `vggt_align.py` | VGGT 덩어리 결과 → v5 지역 좌표로 맞춤 · 점검 · 합친 점구름. |
| `vggt_chunks.py` | 네이버 거리뷰 큐브 면 → VGGT-1B 덩어리 추론 (피드포워드). 덩어리 = 이웃 파노라마 K 곳 × 면 F 장. |
| `vggt_graph.py` | VGGT 덩어리들을 한 좌표계로: 덩어리마다 닮음 변환 (축척 · 회전 · 이동) 을 GPS · 중력 · 땅 높이 · 덩어리 사이 겹침으로 함께 맞춘다. |
| `viewer.py` | 청림동 거리뷰 복원 3D 뷰어 (viser). 내부 전용 — 사진 색 점은 네이버 거리뷰 (© NAVER Corp., 비공식 경로) 에서 나왔다. |
| `viewer_prep.py` | 3D 뷰어용 자료 (내부 전용, 이 컴퓨터 안에서만 본다). 기존 결과 파일은 읽기만 하고 work/viewer/ 에 새로 쓴다. |
| `viewer_shot.py` | 뷰어 화면 점검 (내부 확인용, 사진 색이 들어가므로 work/viewer/ 에만 둔다). 사용: ~/apt-recon/webcheck/bin/python viewer_shot.py <url> <out 접두> |
| `views.py` | 큐브맵 6면 → 임의 방위 · 고개 · 화각의 원근 뷰 (쌍선형). 뷰 틀 = 파노라마 틀 (front 방위 0) 기준. |
| `views_gpu.py` | views.render 의 GPU 판 (torch grid_sample). 같은 규약 (views.view_R · cube.face_R), ss 배 크게 그린 뒤 평균으로 줄인다. |
| `vworld_outlines.py` | v5 빌더와 같은 규칙으로 V-World 동 윤곽을 캐시 파일에서 읽는다 (네트워크 안 씀, 원본 읽기만). |
| `wall_table.py` | 벽마다 세 근거의 입면 자리 (V-World 윤곽 기준) 를 한 표로. 사용: python wall_table.py <vggt.json> <colmap.json> <mesh.json> <out.csv> |

## 실행 환경

- 파이썬 가상환경은 저장소에 없다. 도구가 가져오는 외부 모듈: `numpy` · `scipy` · `PIL` · `matplotlib` · `open3d` · `pycolmap` · `torch` · `vggt` · `trimesh` · `viser` · `playwright` (`tools/colab/` 은 `trellis2` · `o_voxel` · `safetensors` 도). 버전은 고정하지 않았다.
- GPU 를 쓰는 도구(`vggt_chunks.py` 등)는 시작 전에 로컬 ComfyUI 큐(127.0.0.1:8189)가 비었는지 본다. 다른 환경에서는 이 검사를 바꿔야 한다.
- 입력 사진 · 파노라마 목록은 따로 준비해야 한다. 도구마다 머리말에 입력 형식이 적혀 있다.
- 일부 도구(`after_sfm.sh` · `run_variants.sh` · `viewer_prep.py` 등)는 작업 환경의 경로(`~/lightlab/runs/...` · `~/apt-recon/venv`)를 그대로 쓴다. 다른 환경에서는 경로를 바꿔야 한다.

## 알려진 점 (2026-11-07 시험 기준, 재검증 전)

- 큐브맵 6면을 그대로 넣으면 한 파노라마의 면들이 수 m 흩어진다. 큐브맵에서 겹치는 원근 뷰 8–12장을 다시 그려 넣으면 GPS 와의 차가 2.6 m 에서 0.9 m 로 줄었다.
- 덩어리는 이웃 4–5곳(약 30 m)이 좋았고, 64장으로 키우면 나빠졌다.
- COLMAP global mapper 는 등록 비율이 100 % 여도 GPS 와 25 m 어긋난 틀린 모델을 낼 수 있다. 등록 비율만 믿지 말고 늘 GPS 와 견준다.
- 높이는 거리 사진으로 재지 못했다(대장 높이보다 4–19 m 낮게 나옴).
