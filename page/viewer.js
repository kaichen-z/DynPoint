// Minimal three.js point-cloud / scene-flow viewer for DynPoint exports (dynpoint/export_viewer.py).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

async function loadBin(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  let off = 4 + hlen;
  const take = (ctor, n) => { const a = new ctor(buf.slice(off, off + n * ctor.BYTES_PER_ELEMENT)); off += n * ctor.BYTES_PER_ELEMENT; return a; };
  const d = { header };
  d.xyz = take(Float32Array, header.n_points * 3);
  d.rgb = take(Uint8Array, header.n_points * 3);
  d.depthRgb = take(Uint8Array, header.n_points * 3);
  d.dyn = take(Uint8Array, header.n_points);
  d.flowXyz = take(Float32Array, header.n_flow * 3);
  d.flowVec = take(Float32Array, header.n_flow * 3);
  d.cam = take(Float32Array, 15);
  if (header.n_ref) { d.refXyz = take(Float32Array, header.n_ref * 3); d.refRgb = take(Uint8Array, header.n_ref * 3); }
  return d;
}

export function makeViewer(container, options = {}) {
  const width = container.clientWidth, height = options.height || Math.round(width * 0.6);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  renderer.setClearColor(0xffffff, 1);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const group = new THREE.Group();
  // OpenCV world (y down, z forward) -> three.js (y up, z toward viewer)
  group.rotation.x = Math.PI;
  scene.add(group);
  const state = { points: null, flow: null, ref: null, cam: null, data: null, mode: 'rgb', showFlow: true, showRef: false, flowScale: 1.0 };

  function clear() { for (const k of ['points', 'flow', 'ref', 'cam']) { if (state[k]) { group.remove(state[k]); state[k].geometry.dispose(); state[k] = null; } } }

  function build() {
    clear();
    const d = state.data; if (!d) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(d.xyz, 3));
    const col = state.mode === 'depth' ? d.depthRgb : d.rgb;
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
    state.points = new THREE.Points(geo, new THREE.PointsMaterial({ size: options.pointSize || 0.006, vertexColors: true, sizeAttenuation: true }));
    group.add(state.points);
    if (state.showFlow && d.header.n_flow) {
      const n = d.header.n_flow, pos = new Float32Array(n * 6), c = new Float32Array(n * 6);
      let mags = []; for (let i = 0; i < n; i++) mags.push(Math.hypot(d.flowVec[3 * i], d.flowVec[3 * i + 1], d.flowVec[3 * i + 2]));
      const sorted = [...mags].sort((a, b) => a - b), p95 = sorted[Math.floor(0.95 * (n - 1))] || 1e-6;
      const k = (0.08 / Math.max(p95, 1e-6)) * state.flowScale;     // 95th percentile -> 0.08 scene units
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < 3; j++) { pos[6 * i + j] = d.flowXyz[3 * i + j]; pos[6 * i + 3 + j] = d.flowXyz[3 * i + j] + d.flowVec[3 * i + j] * k; }
        c[6 * i] = 1; c[6 * i + 1] = 0.85; c[6 * i + 2] = 0.1; c[6 * i + 3] = 1; c[6 * i + 4] = 0.1; c[6 * i + 5] = 0.1;
      }
      const fg = new THREE.BufferGeometry();
      fg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      fg.setAttribute('color', new THREE.BufferAttribute(c, 3));
      state.flow = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ vertexColors: true }));
      group.add(state.flow);
    }
    if (state.showRef && d.refXyz) {
      const rg = new THREE.BufferGeometry();
      rg.setAttribute('position', new THREE.BufferAttribute(d.refXyz, 3));
      const rc = new Float32Array(d.refRgb.length);
      for (let i = 0; i < d.refRgb.length; i += 3) { rc[i] = 0.2 + 0.3 * d.refRgb[i] / 255; rc[i + 1] = 0.4 + 0.3 * d.refRgb[i + 1] / 255; rc[i + 2] = 1.0; }
      rg.setAttribute('color', new THREE.BufferAttribute(rc, 3));
      state.ref = new THREE.Points(rg, new THREE.PointsMaterial({ size: (options.pointSize || 0.006) * 1.2, vertexColors: true }));
      group.add(state.ref);
    }
    // camera frustum
    const cm = d.cam, o = [cm[0], cm[1], cm[2]], segs = [];
    for (let i = 1; i <= 4; i++) { const j = i % 4 + 1; segs.push(...o, cm[3 * i], cm[3 * i + 1], cm[3 * i + 2]); segs.push(cm[3 * i], cm[3 * i + 1], cm[3 * i + 2], cm[3 * j], cm[3 * j + 1], cm[3 * j + 2]); }
    const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs), 3));
    state.cam = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: 0x2a6f97 }));
    group.add(state.cam);
  }

  async function load(url) {
    state.data = await loadBin(url);
    build();
    // look from the camera position toward the scene centre
    const cm = state.data.cam;
    const xyz = state.data.xyz; let cx = 0, cy = 0, cz = 0; const n = xyz.length / 3;
    for (let i = 0; i < n; i++) { cx += xyz[3 * i]; cy += xyz[3 * i + 1]; cz += xyz[3 * i + 2]; }
    group.updateMatrixWorld(true);
    const target = new THREE.Vector3(cx / n, cy / n, cz / n).applyMatrix4(group.matrixWorld);   // cloud centroid
    const eye = new THREE.Vector3(cm[0], cm[1], cm[2]).applyMatrix4(group.matrixWorld);
    camera.position.copy(eye.clone().sub(target).multiplyScalar(1.6).add(target));
    camera.up.set(0, 1, 0);
    controls.target.copy(target); controls.update();
    return state.data.header;
  }
  function set(opts) { Object.assign(state, opts); build(); }
  function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
  animate();
  window.addEventListener('resize', () => { const w = container.clientWidth, h = options.height || Math.round(w * 0.6); renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); });
  return { load, set, state };
}
