import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { snapStripEnd, washDirection, floodDirection, surfacePoint, distance } from './placement.js';

const $ = (id) => document.getElementById(id);
const canvasHost = $('viewport');
const WELCOME = '건물의 시작 지점을 클릭하세요.';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07121d);
scene.fog = new THREE.FogExp2(0x07121d, 0.00065);
const camera = new THREE.PerspectiveCamera(55, 1, 0.25, 12000);
camera.position.set(360, 390, 530);
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false, powerPreference:'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.5;
canvasHost.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.maxPolarAngle = Math.PI / 2.02;
controls.minDistance = 6;
controls.maxDistance = 2500;
controls.target.set(0, 24, 0);
controls.update();
const ambient = new THREE.HemisphereLight(0xa4cfff, 0x203444, 2.0);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffedd3, 2.0);
sun.position.set(-250, 450, 180);
scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(3000,3000), new THREE.MeshStandardMaterial({color:0x203c34,roughness:1}));
ground.rotation.x=-Math.PI/2;
ground.position.y=-0.15;
scene.add(ground);
const grid = new THREE.GridHelper(1000, 40, 0x41656e, 0x27464b);
grid.position.y=-0.11;
scene.add(grid);
let buildingMesh=null;
let triangleEnds=[];
let buildingRecords=[];
let installed=[];
let selectedId=null;
let pending=[];
let mode='line';
let night=true;
let dataLoaded=false;
let counter=0;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const markerGroup = new THREE.Group();
const lightGroup = new THREE.Group();
scene.add(markerGroup,lightGroup);
const tempV=new THREE.Vector3();
const labels = document.createElement('div');
labels.style.cssText='position:absolute;inset:0;pointer-events:none;overflow:hidden;';
canvasHost.appendChild(labels);

function setStatus(message){$('status').textContent=message;}
function banner(message,kind=''){$('datasetBanner').textContent=message;$('datasetBanner').className='dataset-banner '+kind;}
function currentColor(){return $('color').value;}
function currentIntensity(){return Number($('intensity').value);}
function currentAngle(){return Number($('angle').value);}
function currentElevation(){return Number($('elevation').value);}
function currentOffset(){return Number($('surfaceOffset').value)/100;}
function currentSnap(){return $('snapAxis').checked;}
function worldNormal(n){return new THREE.Vector3(...n).normalize();}
function offsetPoint(point, normal, meters){return v3(surfacePoint(array3(point),array3(normal),meters));}
function entryNormal(entry){
  if(Array.isArray(entry.normal)&&entry.normal.length===3) return worldNormal(entry.normal);
  // For V1 JSON imports without a saved surface normal, infer the nearest facade.
  const record=buildingRecords.find(b=>String(b.id)===String(entry.buildingId));
  if(!record||!Array.isArray(record.points)||record.points.length<2)return new THREE.Vector3(0,0,1);
  const at=entry.points[0];
  if(Number(record.height)&&at[1]>Number(record.height)-.3)return new THREE.Vector3(0,1,0);
  let best=null,bestDist=Infinity;
  for(let i=0;i<record.points.length;i++){
    const a=record.points[i],b=record.points[(i+1)%record.points.length];
    const ax=a[0],az=-a[1],bx=b[0],bz=-b[1],vx=bx-ax,vz=bz-az;
    const t=THREE.MathUtils.clamp(((at[0]-ax)*vx+(at[2]-az)*vz)/(vx*vx+vz*vz||1),0,1);
    const d=(at[0]-ax-t*vx)**2+(at[2]-az-t*vz)**2;
    if(d<bestDist){bestDist=d;best=new THREE.Vector3(-vz,0,vx).normalize();}
  }
  if(!best)return new THREE.Vector3(0,0,1);
  const cent=record.points.reduce((a,p)=>(a.x+=p[0],a.z-=p[1],a),{x:0,z:0});
  cent.x/=record.points.length;cent.z/=record.points.length;
  if(best.dot(new THREE.Vector3(at[0]-cent.x,0,at[2]-cent.z))<0)best.negate();
  return best;
}
function v3(p){return new THREE.Vector3(p[0],p[1],p[2]);}
function array3(v){return [Number(v.x.toFixed(3)),Number(v.y.toFixed(3)),Number(v.z.toFixed(3))];}
function disposeObject(node){node.traverse(obj=>{if(obj.geometry)obj.geometry.dispose();if(obj.material){if(Array.isArray(obj.material))obj.material.forEach(m=>m.dispose());else obj.material.dispose();}});}
function resetGroup(group){while(group.children.length){const obj=group.children[0];group.remove(obj);disposeObject(obj);}}
function colorForBuilding(tags,id){const color = new THREE.Color(); const t=(tags.building || '').toLowerCase();
  if(t==='apartments'||t==='residential')color.set('#a0aebb');
  else if(t==='commercial'||t==='office')color.set('#91b0bb');
  else if(t==='industrial'||t==='warehouse')color.set('#9a9eaa');
  else color.set('#b5bdba');
  const variation=((id*2654435761>>>0)%17-8)/100;
  return color.multiplyScalar(1+variation);
}
function makeBuildings(data){
  if(buildingMesh){scene.remove(buildingMesh);disposeObject(buildingMesh);buildingMesh=null;}
  buildingRecords=[];triangleEnds=[];
  const parts=[];let count=0;
  for(const b of (data.buildings||[])){
    if(!b.points || b.points.length<3)continue;
    const clean=b.points.filter(p=>Array.isArray(p)&&p.length>=2&&p.every(Number.isFinite));
    if(clean.length<3)continue;
    if(clean.length>3&&Math.hypot(clean[0][0]-clean[clean.length-1][0],clean[0][1]-clean[clean.length-1][1])<0.01)clean.pop();
    if(clean.length<3)continue;
    const shape=new THREE.Shape();shape.moveTo(clean[0][0],clean[0][1]);
    for(let i=1;i<clean.length;i++)shape.lineTo(clean[i][0],clean[i][1]);
    shape.closePath();
    let g;
    try{g=new THREE.ExtrudeGeometry(shape,{depth:Math.max(2,Math.min(300,Number(b.height)||12)),bevelEnabled:false,steps:1,curveSegments:1});}
    catch(err){console.warn('Building geometry skipped',b.id,err);continue;}
    g.rotateX(-Math.PI/2);
    if(g.index){const plain=g.toNonIndexed();g.dispose();g=plain;}
    const pa=g.getAttribute('position'),na=g.getAttribute('normal');
    if(!pa||!na||pa.count%3!==0){g.dispose();continue;}
    const bc=colorForBuilding(b.tags||{},Number(b.id)||count);
    const colors=new Float32Array(pa.count*3);
    for(let i=0;i<pa.count;i++){
      const roof=na.getY(i)>0.5;
      const shift=roof?0.68:1.0;
      colors[i*3]=bc.r*shift;colors[i*3+1]=bc.g*shift;colors[i*3+2]=bc.b*shift;
    }
    parts.push({p:pa.array,n:na.array,c:colors});
    count+=pa.count/3;
    triangleEnds.push(count);
    buildingRecords.push(b);
    g.dispose();
  }
  if(!parts.length){banner('건물 데이터를 찾지 못했습니다. 데이터 파일을 확인하세요.','error');return;}
  const total=parts.reduce((v,p)=>v+p.p.length,0);
  const pos=new Float32Array(total),norm=new Float32Array(total),col=new Float32Array(total);
  let offset=0;
  for(const p of parts){pos.set(p.p,offset);norm.set(p.n,offset);col.set(p.c,offset);offset+=p.p.length;}
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('normal',new THREE.BufferAttribute(norm,3));
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  g.computeBoundingSphere();
  buildingMesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.86,metalness:.03,side:THREE.DoubleSide}));
  buildingMesh.name='OSM_BUILDINGS';
  scene.add(buildingMesh);
}
function makeRoads(data){
  let points=[];
  const widthOf={motorway:16,trunk:14,primary:11,secondary:9,tertiary:8,residential:5,service:3,footway:1.6,path:1.2,pedestrian:3};
  for(const road of data.roads||[]){const arr=road.points||[];let w=widthOf[road.type]||4;
    for(let i=1;i<arr.length;i++){
      const a=arr[i-1],b=arr[i];if(!a||!b)continue;
      const dx=b[0]-a[0],dz=-(b[1]-a[1]),len=Math.hypot(dx,dz);if(len<.05)continue;
      const ox=dz/len*w/2,oz=-dx/len*w/2;
      const az=-a[1],bz=-b[1],y=-.07;
      points.push(a[0]+ox,y,az+oz,a[0]-ox,y,az-oz,b[0]+ox,y,bz+oz,
                  b[0]+ox,y,bz+oz,a[0]-ox,y,az-oz,b[0]-ox,y,bz-oz);
    }
  }
  if(!points.length)return;
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(points,3));g.computeVertexNormals();
  const obj=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0x4c6266,side:THREE.DoubleSide}));
  obj.name='OSM_ROADS';scene.add(obj);
}
function whichBuilding(triangle){let lo=0,hi=triangleEnds.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(triangle<triangleEnds[mid])hi=mid;else lo=mid+1;}return lo;}
function pick(evt, allowSky=false){
  const rect=renderer.domElement.getBoundingClientRect();
  pointer.x=((evt.clientX-rect.left)/rect.width)*2-1;
  pointer.y=-((evt.clientY-rect.top)/rect.height)*2+1;
  raycaster.setFromCamera(pointer,camera);
  const hits=raycaster.intersectObjects(buildingMesh?[buildingMesh,ground]:[ground],false);
  if(!hits.length){
    if(!allowSky||!pending.length)return null;
    // An empty-sky click becomes a free 3D direction point at the source's depth.
    const forward=camera.getWorldDirection(new THREE.Vector3());
    const aimPlane=new THREE.Plane().setFromNormalAndCoplanarPoint(forward,pending[0].point);
    const target=new THREE.Vector3();
    if(!raycaster.ray.intersectPlane(aimPlane,target))raycaster.ray.at(40,target);
    return {point:target,normal:new THREE.Vector3(0,1,0),buildingId:null,isBuilding:false,isSky:true};
  }
  const h=hits[0];
  const isBuilding=h.object===buildingMesh;
  let normal=new THREE.Vector3(0,1,0);
  let buildingId=null;
  if(isBuilding){
    buildingId=buildingRecords[whichBuilding(h.faceIndex)]?.id;
    if(h.face?.normal){
      normal.copy(h.face.normal).transformDirection(h.object.matrixWorld).normalize();
      // Three.js double-sided faces can report inward normals: point out to the viewer.
      if(normal.dot(raycaster.ray.direction)>0)normal.negate();
      if(Math.abs(normal.y)>.75)normal.set(0,1,0);
      else {normal.y=0;normal.normalize();}
    }
  }
  return {point:h.point.clone(),normal,buildingId,isBuilding,isSky:false};
}
function makeCylinder(a,b,r,color,opacity=1){const d=new THREE.Vector3().subVectors(b,a),len=d.length();
  if(len<0.001)return null;
  const geometry=new THREE.CylinderGeometry(r,r,len,8,1,false);
  const material=opacity>=1?new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:3,roughness:.3}):new THREE.MeshBasicMaterial({color,transparent:true,opacity,depthWrite:false,blending:THREE.AdditiveBlending});
  const mesh=new THREE.Mesh(geometry,material);
  mesh.position.copy(a).add(b).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());
  return mesh;
}
function newMarker(point,normal=new THREE.Vector3(0,1,0)){const m=new THREE.Mesh(new THREE.SphereGeometry(.42,12,8),new THREE.MeshBasicMaterial({color:0xffdd80,depthTest:false}));m.position.copy(point).addScaledVector(normal,.25);markerGroup.add(m);}
function addSpot(parent,source,target,color,intensity,angle){
  const a=source.clone(),b=target.clone(),diff=b.clone().sub(a),d=diff.length();
  if(d<0.5)return;
  // 시각화용 원뿔 (실제 광학 시뮬레이션 아님)
  const coneRadius=Math.tan(THREE.MathUtils.degToRad(angle/2))*d;
  const beam=new THREE.Mesh(new THREE.ConeGeometry(coneRadius,d,20,1,true),new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.065,depthWrite:false,side:THREE.DoubleSide}));
  beam.position.copy(a).add(b).multiplyScalar(.5);
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),a.clone().sub(b).normalize());
  parent.add(beam);
  // 동적 광원은 성능을 위해 최대 12개만 사용
  if(scene.userData.spotCount<12){
    const spot=new THREE.SpotLight(color,Math.max(.1,intensity)*15,Math.min(100,Math.max(20,d*1.4)),THREE.MathUtils.degToRad(angle/2),.6,1.5);
    spot.position.copy(a);spot.target.position.copy(b);
    parent.add(spot,spot.target);scene.userData.spotCount++;
  }
}
function addIndirectWash(parent,a,b,normal,target,color,intensity){
  const n=worldNormal(array3(normal));
  const direction=v3(washDirection(array3(a),array3(b),array3(target),array3(n)));
  // Continuous full-length facade wash, fading away from the installed strip.
  const reach=THREE.MathUtils.clamp(5+intensity*.85,6,13);
  const farA=a.clone().addScaledVector(direction,reach);
  const farB=b.clone().addScaledVector(direction,reach);
  for(const [i,gain] of [0.35,1.0].entries()){
    const depth= i===0?.035:.055;
    const pts=[a.clone().addScaledVector(n,depth),b.clone().addScaledVector(n,depth),farB.clone().addScaledVector(n,depth),farA.clone().addScaledVector(n,depth)];
    const geo=new THREE.BufferGeometry();
    geo.setAttribute('position',new THREE.Float32BufferAttribute(pts.flatMap(p=>[p.x,p.y,p.z]),3));
    geo.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,1,0,1,1,0,1],2));
    geo.setIndex([0,1,2,0,2,3]);geo.computeVertexNormals();
    const mat=new THREE.ShaderMaterial({
      uniforms:{uColor:{value:color},uStrength:{value:(.23+intensity*.012)*gain}},
      vertexShader:`varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader:`varying vec2 vUv; uniform vec3 uColor; uniform float uStrength;void main(){float fade=pow(1.0-vUv.y,1.9);float xedge=smoothstep(0.0,0.025,vUv.x)*(1.0-smoothstep(0.975,1.0,vUv.x));gl_FragColor=vec4(uColor,uStrength*fade*xedge);}`,
      transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,side:THREE.DoubleSide
    });
    parent.add(new THREE.Mesh(geo,mat));
  }
  // Real local light sources spread over the ENTIRE bar (bounded for performance).
  const span=a.distanceTo(b);
  const requested=Math.max(2,Math.min(8,Math.ceil(span/3)+1));
  for(let i=0;i<requested&&scene.userData.localCount<18;i++){
    const pos=a.clone().lerp(b,(i+.5)/requested).addScaledVector(n,.18);
    const lamp=new THREE.PointLight(color,intensity*2.5,7,2);
    lamp.position.copy(pos);parent.add(lamp);scene.userData.localCount++;
  }
}
function createVisual(entry){
  const parent=new THREE.Group();parent.userData.entryId=entry.id;
  const c=new THREE.Color(entry.color||'#64dfff');
  const p=entry.points.map(v3);
  const n=entryNormal(entry);
  const gap=THREE.MathUtils.clamp(entry.offset??.18,.08,.5);
  let labelAt;
  if(entry.type==='line'||entry.type==='indirect'){
    const a=offsetPoint(p[0],n,gap),b=offsetPoint(p[1],n,gap);
    // Fixture radius is in real-world meters (V1 had a 0.5 m radius).
    const r=entry.type==='line'?.045:.075;
    const tube=makeCylinder(a,b,r,c);if(tube)parent.add(tube);
    const halo=makeCylinder(a,b,r*2.7,c,.18);if(halo)parent.add(halo);
    if(entry.type==='indirect'&&p[2])addIndirectWash(parent,a,b,n,p[2],c,entry.intensity);
    labelAt=a.clone().add(b).multiplyScalar(.5);
  }else if(entry.type==='flood'){
    const src=offsetPoint(p[0],n,gap+.16);
    const elevation=Number.isFinite(entry.elevation)?entry.elevation:40;
    const dir=v3(floodDirection(array3(src),array3(p[1]),array3(n),elevation));
    const dst=src.clone().addScaledVector(dir,34);
    const fixture=new THREE.Mesh(new THREE.SphereGeometry(.5,12,10),new THREE.MeshStandardMaterial({color:0x242f37,emissive:c,emissiveIntensity:2}));
    fixture.position.copy(src);parent.add(fixture);
    addSpot(parent,src,dst,c,entry.intensity,entry.angle||45);
    labelAt=src;
  }
  lightGroup.add(parent);
  const label=document.createElement('div');
  label.className='length-label';
  label.textContent=(entry.type==='flood'?'투광기':entry.type==='line'?'라인 ':'간접 ')+(entry.length!=null?entry.length.toFixed(2)+' m':'');
  label.style.cssText='position:absolute;display:none;color:#e6faff;background:#082438dc;border:1px solid #4487ad;border-radius:6px;padding:3px 7px;font:11px Arial;white-space:nowrap';
  labels.appendChild(label);
  return {group:parent,label,at:labelAt};
}
function disposeEntry(e){if(!e.visual)return;lightGroup.remove(e.visual.group);disposeObject(e.visual.group);e.visual.label.remove();e.visual=null;}
function rebuildEntries(){
  for(const e of installed)disposeEntry(e);
  scene.userData.spotCount=0;scene.userData.localCount=0;
  for(const e of installed)e.visual=createVisual(e);
  refreshList();
}
function refreshList(){
  let lin=0,ind=0,fl=0;
  const list=$('installList');list.replaceChildren();
  for(const e of installed){if(e.type==='line')lin+=e.length||0;else if(e.type==='indirect')ind+=e.length||0;else fl++;
    const btn=document.createElement('button');btn.type='button';btn.className='entry'+(selectedId===e.id?' selected':'');
    const names={line:'라인조명',indirect:'간접조명',flood:'투광기'};
    btn.textContent=`${names[e.type]} #${e.id} · ${e.length!=null?e.length.toFixed(2)+' m':'1대'}`;
    btn.onclick=()=>{
      selectedId=e.id;refreshList();
      $('color').value=e.color;$('intensity').value=String(e.intensity);
      $('angle').value=String(e.angle);$('elevation').value=String(e.elevation??40);
      $('surfaceOffset').value=String(Math.round((e.offset??.18)*100));
      $('intensityVal').textContent=String(e.intensity);
      $('angleVal').textContent=String(e.angle)+'°';
      $('elevationVal').textContent=String(e.elevation??40)+'°';
      $('surfaceOffsetVal').textContent=String(Math.round((e.offset??.18)*100))+' cm';
      setStatus('선택한 조명의 색상·밝기·돌출을 변경할 수 있습니다. 투광기는 상향각·조사각도 바로 바뀝니다.');
    };
    list.appendChild(btn);
  }
  if(!installed.length){const t=document.createElement('span');t.className='muted smaller';t.textContent='아직 설치한 조명이 없습니다.';list.appendChild(t);}
  $('lineTotal').textContent=lin.toFixed(2)+' m';
  $('indirectTotal').textContent=ind.toFixed(2)+' m';
  $('floodCount').textContent=fl+' 대';
  $('itemCount').textContent=String(installed.length);
}
function guidance(){
  const txt={line:['건물에서 라인조명 시작점을 클릭하세요.','같은 외벽의 끝점을 클릭하세요. 수직·수평은 자동 정렬됩니다.'],indirect:['건물에서 간접조명 시작점을 클릭하세요.','같은 건물에서 끝점을 클릭하세요.','간접조명이 퍼질 방향을 클릭하세요. 벽면·지면·하늘 모두 가능합니다.'],flood:['건물에서 투광기 설치점을 클릭하세요.','수평 방향을 클릭하세요. 하늘을 클릭해도 되며 상향각은 슬라이더로 조절합니다.']};
  setStatus(txt[mode][pending.length]||WELCOME);
  const descriptions={line:'건물의 두 지점을 클릭해 길이를 측정합니다.',indirect:'두 지점으로 바를 설치하고 세 번째 클릭으로 방향을 지정합니다.',flood:'설치점을 클릭하고 두 번째 클릭으로 조사 방향을 지정합니다.'};
  $('guide').textContent=descriptions[mode];
}
function clearPending(){pending=[];resetGroup(markerGroup);guidance();}
function handleSelection(hit){
  if(!dataLoaded){setStatus('도시 데이터가 아직 준비되지 않았습니다.');return;}
  const required=mode==='indirect'?3:2;
  const placingStrip=mode==='line'||mode==='indirect';
  const chooseBuilding=placingStrip?pending.length<2:pending.length===0;
  if(chooseBuilding&&!hit.isBuilding){setStatus('설치 위치는 건물 벽면이나 지붕에서 선택해 주세요.');return;}
  if(placingStrip&&pending.length===1){
    if(String(hit.buildingId)!==String(pending[0].buildingId)){
      setStatus('시작점과 끝점은 같은 건물에 지정해 주세요.');return;
    }
    const normal=pending[0].normal;
    // Reject other faces and parallel facade planes; cross-corner lines would cut through a building.
    if(normal.dot(hit.normal)<.94 || Math.abs(hit.point.clone().sub(pending[0].point).dot(normal))>.4){
      setStatus('라인이 벽 안으로 들어가지 않도록 같은 평면의 외벽(또는 지붕)을 선택해 주세요.');return;
    }
    hit={...hit,point:v3(snapStripEnd(array3(pending[0].point),array3(hit.point),array3(normal),currentSnap())),normal:normal.clone()};
  }
  pending.push(hit);
  newMarker(hit.point,hit.normal);
  if(pending.length<required){guidance();return;}
  const pts=pending.map(e=>array3(e.point));
  const length=mode==='flood'?null:distance(pts[0],pts[1]);
  if((length!==null&&length<.2)||(mode==='flood'&&distance(pts[0],pts[1])<.5)){
    setStatus('지점이 너무 가까워요. 다른 위치를 선택해 주세요.');pending.pop();resetGroup(markerGroup);for(const p of pending)newMarker(p.point,p.normal);return;
  }
  const entry={id:++counter,type:mode,points:pts,normal:array3(pending[0].normal),offset:currentOffset(),elevation:currentElevation(),snap:currentSnap(),color:currentColor(),intensity:currentIntensity(),angle:currentAngle(),length,buildingId:pending[0].buildingId};
  installed.push(entry);selectedId=entry.id;
  clearPending();rebuildEntries();
  setStatus(mode==='flood'?`투광기 설치 완료 · 상향각 ${entry.elevation}°`:`${mode==='line'?'라인조명':'간접조명'} 설치 완료 · 모델 기준 ${length.toFixed(2)} m`);
}
function safeEntry(candidate){
  if(!candidate || !['line','indirect','flood'].includes(candidate.type))return null;
  const count=candidate.type==='indirect'?3:2;
  if(!Array.isArray(candidate.points)||candidate.points.length!==count)return null;
  if(!candidate.points.every(p=>Array.isArray(p)&&p.length===3&&p.every(v=>typeof v==='number'&&Number.isFinite(v)&&Math.abs(v)<50000)))return null;
  const length=candidate.type==='flood'?null:v3(candidate.points[0]).distanceTo(v3(candidate.points[1]));
  return {id:++counter,type:candidate.type,points:candidate.points,
    normal:Array.isArray(candidate.normal)&&candidate.normal.length===3&&candidate.normal.every(Number.isFinite)?candidate.normal:null,
    offset:THREE.MathUtils.clamp(Number(candidate.offset)||.18,.08,.5),
    elevation:THREE.MathUtils.clamp(Number.isFinite(candidate.elevation)?candidate.elevation:40,-25,88),
    snap:!!candidate.snap,
    color:/^#[a-f0-9]{6}$/i.test(candidate.color)?candidate.color:'#64dfff',
    intensity:Math.max(1,Math.min(10,Number(candidate.intensity)||7)),
    angle:Math.max(15,Math.min(100,Number(candidate.angle)||45)),length,buildingId:candidate.buildingId||null};
}
for(const btn of document.querySelectorAll('[data-tool]'))btn.onclick=()=>{
  mode=btn.dataset.tool;
  selectedId=null;refreshList();
  for(const el of document.querySelectorAll('[data-tool]'))el.classList.toggle('active',el===btn);
  clearPending();
};
function updateSelected(prop,value,requiredType=null){
  const e=installed.find(item=>item.id===selectedId);
  if(!e || (requiredType && e.type!==requiredType))return;
  e[prop]=value;rebuildEntries();
}
$('color').oninput=()=>updateSelected('color',currentColor());
$('intensity').oninput=()=>{$('intensityVal').textContent=$('intensity').value;updateSelected('intensity',currentIntensity());};
$('angle').oninput=()=>{$('angleVal').textContent=$('angle').value+'°';updateSelected('angle',currentAngle(),'flood');};
$('elevation').oninput=()=>{$('elevationVal').textContent=$('elevation').value+'°';updateSelected('elevation',currentElevation(),'flood');};
$('surfaceOffset').oninput=()=>{$('surfaceOffsetVal').textContent=$('surfaceOffset').value+' cm';updateSelected('offset',currentOffset());};
$('btnUndo').onclick=()=>{if(pending.length){pending.pop();resetGroup(markerGroup);for(const p of pending)newMarker(p.point,p.normal);guidance();return;}const e=installed.pop();if(!e){setStatus('되돌릴 내용이 없어요.');return;}disposeEntry(e);selectedId=null;rebuildEntries();setStatus('마지막 설치를 취소했습니다.');};
$('btnDelete').onclick=()=>{const i=installed.findIndex(e=>e.id===selectedId);if(i<0){setStatus('삭제할 조명을 설치 목록에서 선택해 주세요.');return;}const [e]=installed.splice(i,1);disposeEntry(e);selectedId=null;rebuildEntries();setStatus('선택한 조명을 삭제했습니다.');};
$('btnClear').onclick=()=>{if(!installed.length&&!pending.length)return;if(!confirm('설치한 조명을 모두 삭제할까요?'))return;for(const e of installed)disposeEntry(e);installed=[];selectedId=null;clearPending();rebuildEntries();};
$('btnExport').onclick=()=>{
  const payload={version:2,project:'HANA LIGHT SIMULATOR',dataSource:window.__hanaDataSource||'unknown',savedAt:new Date().toISOString(),measurements:'3D model approximation; verify on site',entries:installed.map(({visual,...data})=>data)};
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='hana-light-plan.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  setStatus('JSON 파일을 기기에 저장했습니다.');
};
$('btnImport').onclick=()=>$('fileImport').click();
$('fileImport').onchange=async e=>{const file=e.target.files?.[0];if(!file)return;
  try{const json=JSON.parse(await file.text());if(!Array.isArray(json.entries))throw new Error('entries 형식이 아닙니다.');
    if(json.entries.length>1000)throw new Error('조명은 최대 1,000개까지 불러옵니다.');
    const parsed=json.entries.map(safeEntry);if(parsed.some(x=>!x))throw new Error('일부 조명 데이터의 형식이 잘못되었습니다.');
    for(const a of installed)disposeEntry(a);installed=parsed;selectedId=null;clearPending();rebuildEntries();setStatus(`${parsed.length}개 조명을 불러왔습니다.`);
  }catch(err){setStatus('불러오기 실패: '+err.message);}finally{e.target.value='';}
};
function setNight(value){night=value;scene.background.set(value?0x07121d:0xc9e3f3);scene.fog.color.set(value?0x07121d:0xc9e3f3);ambient.intensity=value?1.4:2.6;sun.intensity=value?0.35:2.3;renderer.toneMappingExposure=value?1.65:1.15;grid.visible=value; $('btnDay').classList.toggle('active',!value);$('btnNight').classList.toggle('active',value);}
$('btnDay').onclick=()=>setNight(false);$('btnNight').onclick=()=>setNight(true);
$('btnHome').onclick=()=>{camera.position.set(360,390,530);controls.target.set(0,24,0);controls.update();};
window.addEventListener('keydown',e=>{if(e.key==='Escape')clearPending();else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$('btnUndo').click();}});
let pointerDown=null;
renderer.domElement.addEventListener('pointerdown',e=>{if(e.button!==0)return;pointerDown={x:e.clientX,y:e.clientY,time:Date.now()};});
renderer.domElement.addEventListener('pointerup',e=>{
  if(!pointerDown)return;
  const moved=Math.hypot(e.clientX-pointerDown.x,e.clientY-pointerDown.y);
  pointerDown=null;if(moved>7)return;
  const aimClick=(mode==='indirect'&&pending.length===2)||(mode==='flood'&&pending.length===1);
  const hit=pick(e,aimClick);
  if(hit)handleSelection(hit);
});
renderer.domElement.addEventListener('pointercancel',()=>{pointerDown=null;});
function resize(){const r=canvasHost.getBoundingClientRect();const w=Math.max(1,r.width),h=Math.max(1,r.height);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false);}
window.addEventListener('resize',resize);
resize();
function loop(){requestAnimationFrame(loop);controls.update();
  const rect=canvasHost.getBoundingClientRect();
  for(const e of installed){if(!e.visual)continue;tempV.copy(e.visual.at).project(camera);
    const visible=tempV.z<1&&Math.abs(tempV.x)<1.15&&Math.abs(tempV.y)<1.15;
    e.visual.label.style.display=visible?'block':'none';
    if(visible){e.visual.label.style.left=((tempV.x*.5+.5)*rect.width)+'px';e.visual.label.style.top=((-tempV.y*.5+.5)*rect.height)+'px';}
  }
  renderer.render(scene,camera);
}
loop();
async function init(){
  try{
    const response=await fetch('./data/daegu-1km.json',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const d=await response.json();
    if(!d.buildings?.length)throw new Error('건물 데이터가 없습니다.');
    makeBuildings(d);makeRoads(d);dataLoaded=!!buildingMesh;
    window.__hanaDataSource=d.meta?.source||'unknown';
    const real=d.meta?.is_real_osm===true;
    banner(real?`대구역 중심 1km × 1km · OSM 실제 건물 ${buildingRecords.length.toLocaleString()}동 · 지형은 평면(높이는 일부 추정)`:`가상 건물 데모 ${buildingRecords.length}동 · 실제 대구 데이터가 아닙니다. 먼저 OSM 수집 스크립트를 실행하세요.`,real?'':'demo');
    setStatus(real?'실제 건물 데이터 로딩 완료. 시작점을 클릭하세요.':'데모가 표시 중입니다. 실제 대구 데이터를 수집하면 자동으로 교체됩니다.');
  }catch(err){banner('도시 데이터 로딩 실패: '+err.message+' · scripts/fetch_daegu_osm.py 실행 필요','error');setStatus('데이터 파일을 확인해 주세요.');console.error(err);}
  setNight(true);guidance();
}
init();
