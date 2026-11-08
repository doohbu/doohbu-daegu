// HANA LIGHT SIMULATOR V2 - pure spatial placement math (meters; Three.js Y is up).
// No imports so these helpers can also be verified with Node independently.
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const mul=(v,k)=>[v[0]*k,v[1]*k,v[2]*k];
const mag=v=>Math.hypot(...v);
export function unit(v, fallback=[0,0,1]){
  const n=mag(v);return n>1e-6?mul(v,1/n):fallback.slice();
}
export function snapStripEnd(start,end,normal,snap=true){
  const n=unit(normal);
  const d=sub(end,start);
  if(Math.abs(n[1])>0.75){
    // Flat roof: stay on the roof plane, optionally align with X or Z axis.
    const result= snap
      ? (Math.abs(d[0])>=Math.abs(d[2])?[start[0]+d[0],start[1],start[2]]:[start[0],start[1],start[2]+d[2]])
      : [end[0],start[1],end[2]];
    return result;
  }
  // A wall: tangent is horizontal and lies in the clicked facade plane.
  const tangent=unit([-n[2],0,n[0]],[1,0,0]);
  const along=dot(d,tangent);
  const rise=d[1];
  if(snap){
    return Math.abs(rise)>=Math.abs(along)
      ? add(start,[0,rise,0])
      : add(start,mul(tangent,along));
  }
  return add(start,add(mul(tangent,along),[0,rise,0]));
}
export function washDirection(start,end,aim,normal){
  const n=unit(normal);
  const mid=mul(add(start,end),.5);
  let v=sub(aim,mid);
  v=sub(v,mul(n,dot(v,n))); // strip-to-target vector, projected onto facade plane
  if(mag(v)<.1){
    v=Math.abs(n[1])>.75?[1,0,0]:[0,1,0];
  }
  return unit(v);
}
export function floodDirection(source,aim,normal,elevationDegrees){
  let h=[aim[0]-source[0],0,aim[2]-source[2]];
  if(mag(h)<.5)h=[normal[0],0,normal[2]];
  const forward=unit(h,[0,0,1]);
  const angle=Math.max(-25,Math.min(88,Number(elevationDegrees)||0))*Math.PI/180;
  return [forward[0]*Math.cos(angle),Math.sin(angle),forward[2]*Math.cos(angle)];
}
export function surfacePoint(point,normal,offsetMeters){
  return add(point,mul(unit(normal),Math.max(0.02,Math.min(1,offsetMeters))));
}
export function distance(a,b){return mag(sub(a,b));}
