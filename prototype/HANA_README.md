# HANA LIGHT SIMULATOR — Private V1

This is an independent demo project, not a modified SMASH CAU source tree.
It uses OSM buildings/roads around Daegu Station, and a local copy of Three.js installed with npm.

## Setup in your private Codespace
1. In Terminal create a private development branch: `git switch -c lighting-dev` (or `git switch lighting-dev`).
2. Copy the unzipped project content into the repo root; preserve your own LICENSE if any.
3. Run `bash setup.sh`. This installs Three.js and fetches OSM building/road outlines.
4. Run `python3 -m http.server 8000 --bind 0.0.0.0 -d site` if server isn't running already.
5. In Codespaces PORTS, ensure port 8000 is Private, then Open in Browser.
6. When ready: `git add . && git commit -m "Add private 3D lighting prototype" && git push -u origin lighting-dev`.

## Controls
- Left-click a building surface: begin/end lighting placement.
- Line: 2 points on same building, shows 3D length.
- Indirect: 2 points on same building, third point for light direction.
- Floodlight: 1 fixture point and 1 beam target point.
- Drag/orbit, right-click/pan, scroll or pinch zoom.
- JSON save/load happens in the browser; no server endpoint.

## Security
- Codespaces is cloud-hosted, **not an offline private LAN**.
- Codespaces private port means authenticated-only, NOT zero external processing.
- Do not enable public ports or GitHub Pages. Keep repo Private.
- Running `setup.sh` downloads `three` from npm and **public OSM data** from Overpass into the Codespace. Browser does not contact them at runtime.
- Do not upload proprietary or customer data if your policy disallows cloud services.

## Limitations
- Buildings are OSM **footprints** extruded to available/estimated heights: not actual facade photos/windows.
- Terrain is flat in V1; no DEM elevation or building foundation elevations.
- All length values are model-based estimates, not construction survey lengths.
- Beam effects are illustrative, not photometric or illuminance calculations.
- Some OSM multipolygon relation outlines/courtyards are not included in this V1 parser.
- 1km means 1km x 1km centered on Korail Daegu Station (35.8759306,128.5960528).
- Data acquisition is manual; public Overpass API availability varies.
- If acquisition fails, a prominently labeled **fictional demo** is generated; do not mistake it for Daegu.

## Third-party licenses
- Three.js: MIT license; https://github.com/mrdoob/three.js
- OSM: © OpenStreetMap contributors, ODbL; https://www.openstreetmap.org/copyright

This project's custom code was prepared independently for the user and does not embed SMASH CAU game assets or scripts.
