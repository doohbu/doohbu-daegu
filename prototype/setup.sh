#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo '=== HANA LIGHT SIMULATOR V1 / Codespaces setup ==='
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo 'Node.js/npm이 필요합니다. GitHub Codespaces 기본 이미지에서 실행해 주세요.' >&2
  exit 1
fi
npm install --no-audit --no-fund
mkdir -p site/vendor
cp node_modules/three/build/three.module.js site/vendor/three.module.js
cp node_modules/three/build/three.core.js site/vendor/three.core.js
cp node_modules/three/examples/jsm/controls/OrbitControls.js site/vendor/OrbitControls.js
echo 'Three.js 파일을 자체 웹폴더에 복사했습니다. 브라우저는 CDN을 사용하지 않습니다.'
echo '=== 대구역 실제 OSM 데이터 가져오기 ==='
if python3 scripts/fetch_daegu_osm.py; then
  echo '실제 OSM 데이터 준비 완료.'
else
  echo 'OSM 서버 접속 실패: 데모 파일이 없으면 가상 건물 데이터를 생성합니다.' >&2
  if [ ! -f site/data/daegu-1km.json ]; then python3 scripts/build_demo.py; fi
fi
echo '=== 준비 완료. 기존 8000 포트 서버가 켜져 있다면 테스트 브라우저 새로고침 ==='
echo '서버가 꺼졌다면: python3 -m http.server 8000 --bind 0.0.0.0 -d site'
