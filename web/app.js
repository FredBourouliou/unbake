// Unbake editor: drop a GIF, get objects back, edit their motion, export.
const $ = (id) => document.getElementById(id);
const workspace = $('workspace');
const baked = $('baked'), bctx = baked.getContext('2d');
const unbaked = $('unbaked'), uctx = unbaked.getContext('2d');
const statusEl = $('status');
const HANDLE_R = 6;

const state = {
  scene: null,        // { width, height, frames, fps, background, objects: [...] }
  original: null,     // objects as reconstructed, for Reset
  frames: null,       // Uint8Array, frames × h × w × 3
  imageData: [],      // cached ImageData per frame
  frame: 0,
  playing: false,
  selected: null,
  undo: [],
  drag: null,
  hover: false,
};

// Worker --------------------------------------------------------------------------------
const worker = new Worker('worker.js', { type: 'module' });
let workerReady = false;
let pending = null;
worker.onmessage = ({ data }) => {
  if (data.type === 'status') setStatus(data.text);
  else if (data.type === 'ready') { workerReady = true; setStatus(pending ? 'Reconstructing…' : 'Ready. Drop a GIF or pick a sample.'); if (pending) { worker.postMessage(pending); pending = null; } }
  else if (data.type === 'scene') onScene(data.scene, new Uint8Array(data.frames));
  else if (data.type === 'exported') download(data);
  else if (data.type === 'error') { setStatus('Something went wrong: ' + data.text.split('\n').slice(-2).join(' ')); workspace.dataset.state = state.scene ? 'ready' : 'empty'; }
};

function setStatus(text) { statusEl.textContent = text; }

function reconstruct(bytes) {
  workspace.dataset.state = 'loading';
  const msg = { type: 'reconstruct', bytes };
  if (workerReady) { setStatus('Reconstructing…'); worker.postMessage(msg); }
  else { pending = msg; setStatus('Loading Python in your browser (first visit only)…'); }
}

// Input -----------------------------------------------------------------------------------
const drop = $('drop');
$('pick').onclick = () => $('file').click();
$('file').onchange = (e) => e.target.files[0] && readFile(e.target.files[0]);
['dragenter', 'dragover'].forEach(ev => document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
['dragleave', 'drop'].forEach(ev => document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
document.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) readFile(f); });
document.querySelectorAll('.sample').forEach(b => b.onclick = async () => {
  reconstruct(new Uint8Array(await (await fetch(b.dataset.src)).arrayBuffer()));
});
async function readFile(file) {
  if (!/gif$/i.test(file.type) && !/\.gif$/i.test(file.name)) { setStatus('Please drop a GIF.'); return; }
  reconstruct(new Uint8Array(await file.arrayBuffer()));
}

// Scene -----------------------------------------------------------------------------------
function onScene(scene, frames) {
  state.scene = scene;
  state.original = structuredClone(scene.objects);
  state.frames = frames;
  state.imageData = [];
  state.frame = 0;
  state.selected = null;
  state.undo = [];
  baked.width = scene.width; baked.height = scene.height;
  $('scrub').max = scene.frames - 1;
  workspace.dataset.state = 'ready';
  ['export-lottie', 'export-svg', 'new'].forEach(id => $(id).disabled = false);
  const keys = scene.objects.reduce((n, o) => n + o.pos.length + o.rot.length + o.scale.length, 0);
  setStatus(`${scene.objects.length} objects, ${keys} keyframes, reconstructed in ${scene.seconds} s on this machine.`);
  renderList();
  renderInspector();
  draw();
}

const scale = () => unbaked.width / state.scene.width;

function lerpKeys(keys, f, dims) {
  if (f <= keys[0][0]) return keys[0].slice(1);
  const last = keys[keys.length - 1];
  if (f >= last[0]) return last.slice(1);
  for (let i = 1; i < keys.length; i++) {
    if (f <= keys[i][0]) {
      const a = keys[i - 1], b = keys[i], t = (f - a[0]) / (b[0] - a[0]);
      return Array.from({ length: dims }, (_, d) => a[d + 1] + (b[d + 1] - a[d + 1]) * t);
    }
  }
}

function params(o, f) {
  if (o.hidden || f < o.span[0] || f > o.span[1]) return null;
  const [x, y] = lerpKeys(o.pos, f, 2);
  const [r] = lerpKeys(o.rot, f, 1);
  const [s] = lerpKeys(o.scale, f, 1);
  return { x, y, r, s };
}

function objectPath(o, p) {
  const S = scale(), c = Math.cos(p.r * Math.PI / 180), sn = Math.sin(p.r * Math.PI / 180);
  const path = new Path2D();
  for (const ring of [o.outer, ...(o.holes || [])]) {
    ring.forEach(([px, py], i) => {
      const X = (px * c - py * sn) * p.s + p.x, Y = (px * sn + py * c) * p.s + p.y;
      i ? path.lineTo(X * S, Y * S) : path.moveTo(X * S, Y * S);
    });
    path.closePath();
  }
  return path;
}

// Drawing ---------------------------------------------------------------------------------
function draw() {
  const sc = state.scene; if (!sc) return;
  const f = state.frame;
  // Baked: the input frame.
  if (!state.imageData[f]) {
    const { width: w, height: h } = sc, n = w * h, src = state.frames, off = f * n * 3;
    const img = new ImageData(w, h);
    for (let i = 0; i < n; i++) { img.data[i * 4] = src[off + i * 3]; img.data[i * 4 + 1] = src[off + i * 3 + 1]; img.data[i * 4 + 2] = src[off + i * 3 + 2]; img.data[i * 4 + 3] = 255; }
    state.imageData[f] = img;
  }
  bctx.putImageData(state.imageData[f], 0, 0);

  // Unbaked: objects re-rendered from the edited keyframes.
  uctx.fillStyle = sc.background;
  uctx.fillRect(0, 0, unbaked.width, unbaked.height);
  for (const o of sc.objects) {
    const p = params(o, f);
    if (!p) continue;
    uctx.fillStyle = o.color;
    uctx.fill(objectPath(o, p), 'evenodd');
  }
  const sel = sc.objects.find(o => o.id === state.selected);
  if (sel) drawSelection(sel, f);
  $('scrub').value = f;
  $('frame-out').textContent = `${String(f + 1).padStart(String(sc.frames).length, '0')} / ${sc.frames}`;
}

function drawSelection(o, f) {
  const S = scale(), accent = getComputedStyle(document.documentElement).getPropertyValue('--accent');
  uctx.save();
  // Motion path over the visible span.
  uctx.beginPath();
  for (let t = o.span[0]; t <= o.span[1]; t++) {
    const [x, y] = lerpKeys(o.pos, t, 2);
    t === o.span[0] ? uctx.moveTo(x * S, y * S) : uctx.lineTo(x * S, y * S);
  }
  uctx.strokeStyle = accent; uctx.globalAlpha = 0.7; uctx.lineWidth = 1.5; uctx.setLineDash([4, 4]); uctx.stroke();
  uctx.setLineDash([]); uctx.globalAlpha = 1;
  // Outline of the object on this frame.
  const p = params(o, f);
  if (p) { uctx.lineWidth = 2; uctx.stroke(objectPath(o, p)); }
  // Keyframe handles.
  for (const [kf, x, y] of o.pos) {
    uctx.beginPath(); uctx.arc(x * S, y * S, HANDLE_R, 0, Math.PI * 2);
    uctx.fillStyle = kf === f ? accent : '#1d1b19'; uctx.fill();
    uctx.strokeStyle = accent; uctx.lineWidth = 2; uctx.stroke();
  }
  uctx.restore();
}

// Objects list and inspector --------------------------------------------------------------
function renderList() {
  const ul = $('objects'); ul.replaceChildren();
  [...state.scene.objects].reverse().forEach((o) => {  // top-most first, as in design tools
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.className = 'obj' + (o.hidden ? ' is-hidden' : '');
    b.setAttribute('aria-pressed', String(o.id === state.selected));
    const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = o.color;
    const name = document.createElement('span'); name.textContent = `Object ${o.id + 1}`;
    const meta = document.createElement('span'); meta.className = 'mono'; meta.textContent = `${o.pos.length} keys`;
    b.append(sw, name, meta);
    b.onclick = () => select(o.id === state.selected ? null : o.id);
    li.append(b); ul.append(li);
  });
}

function renderInspector() {
  const o = selectedObject();
  $('inspector').hidden = !o;
  if (!o) return;
  $('color').value = o.color;
  $('toggle-visible').textContent = o.hidden ? 'Show' : 'Hide';
  const facts = [['frames', `${o.span[0] + 1}–${o.span[1] + 1}`], ['position keys', o.pos.length],
                 ['rotation keys', o.rot.length], ['scale keys', o.scale.length], ['colour', o.color]];
  $('facts').replaceChildren();
  for (const [k, v] of facts) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    $('facts').append(dt, dd);
  }
}

const selectedObject = () => state.scene?.objects.find(o => o.id === state.selected);
function select(id) { state.selected = id; renderList(); renderInspector(); draw(); }

function remember() {
  state.undo.push(structuredClone(state.scene.objects));
  if (state.undo.length > 100) state.undo.shift();
  $('undo').disabled = false;
}
function undo() {
  if (!state.undo.length) return;
  state.scene.objects = state.undo.pop();
  $('undo').disabled = !state.undo.length;
  renderList(); renderInspector(); draw();
}

$('color').addEventListener('pointerdown', () => remember());
$('color').addEventListener('keydown', () => remember(), { once: false });
$('color').oninput = (e) => { const o = selectedObject(); if (o) { o.color = e.target.value; renderList(); draw(); } };
$('color').onchange = () => renderInspector();
$('toggle-visible').onclick = () => { const o = selectedObject(); if (!o) return; remember(); o.hidden = !o.hidden; renderList(); renderInspector(); draw(); };
$('reset').onclick = () => {
  const o = selectedObject(); if (!o) return; remember();
  const i = state.scene.objects.indexOf(o);
  state.scene.objects[i] = structuredClone(state.original.find(x => x.id === o.id));
  renderList(); renderInspector(); draw();
};
$('undo').onclick = undo;
$('new').onclick = () => { stop(); state.scene = null; workspace.dataset.state = 'empty'; $('objects').replaceChildren(); $('inspector').hidden = true; ['export-lottie', 'export-svg', 'undo', 'new'].forEach(id => $(id).disabled = true); setStatus(workerReady ? 'Ready. Drop a GIF or pick a sample.' : ''); };

// Canvas interaction ----------------------------------------------------------------------
function canvasPoint(e) {
  const r = unbaked.getBoundingClientRect();
  return { x: (e.clientX - r.left) * unbaked.width / r.width, y: (e.clientY - r.top) * unbaked.height / r.height };
}

function hit(pt) {
  const S = scale(), sel = selectedObject();
  if (sel) {
    const k = sel.pos.findIndex(([, x, y]) => Math.hypot(x * S - pt.x, y * S - pt.y) <= HANDLE_R + 3);
    if (k >= 0) return { kind: 'key', obj: sel, k };
  }
  const objs = [...state.scene.objects].reverse();
  for (const o of objs) {
    const p = params(o, state.frame);
    if (p && uctx.isPointInPath(objectPath(o, p), pt.x, pt.y, 'evenodd')) return { kind: 'body', obj: o };
  }
  return null;
}

unbaked.addEventListener('pointerdown', (e) => {
  if (!state.scene) return;
  const pt = canvasPoint(e), h = hit(pt);
  if (!h) { select(null); return; }
  if (h.obj.id !== state.selected) select(h.obj.id);
  remember();
  state.drag = { ...h, start: pt, keys: structuredClone(h.obj.pos) };
  unbaked.setPointerCapture(e.pointerId);
  unbaked.classList.add('is-dragging');
});
unbaked.addEventListener('pointermove', (e) => {
  if (!state.scene) return;
  const pt = canvasPoint(e);
  if (!state.drag) { unbaked.classList.toggle('is-hover', !!hit(pt)); return; }
  const S = scale(), dx = (pt.x - state.drag.start.x) / S, dy = (pt.y - state.drag.start.y) / S;
  const o = state.drag.obj;
  if (state.drag.kind === 'key') {
    const [f, x, y] = state.drag.keys[state.drag.k];
    o.pos[state.drag.k] = [f, +(x + dx).toFixed(2), +(y + dy).toFixed(2)];
  } else {
    o.pos = state.drag.keys.map(([f, x, y]) => [f, +(x + dx).toFixed(2), +(y + dy).toFixed(2)]);
  }
  draw();
});
unbaked.addEventListener('pointerup', () => {
  if (!state.drag) return;
  const moved = JSON.stringify(state.drag.keys) !== JSON.stringify(state.drag.obj.pos);
  if (!moved) state.undo.pop();
  $('undo').disabled = !state.undo.length;
  state.drag = null;
  unbaked.classList.remove('is-dragging');
});

// Playback -------------------------------------------------------------------------------
let raf = 0, last = 0;
function tick(t) {
  if (!state.playing) return;
  if (t - last >= 1000 / state.scene.fps) { state.frame = (state.frame + 1) % state.scene.frames; last = t; draw(); }
  raf = requestAnimationFrame(tick);
}
function play() { if (!state.scene) return; state.playing = true; $('play-icon').setAttribute('d', 'M4 2.5h3v11H4zM9 2.5h3v11H9z'); $('play').setAttribute('aria-label', 'Pause'); raf = requestAnimationFrame(tick); }
function stop() { state.playing = false; cancelAnimationFrame(raf); $('play-icon').setAttribute('d', 'M4 2.5v11l9-5.5z'); $('play').setAttribute('aria-label', 'Play'); }
$('play').onclick = () => state.playing ? stop() : play();
$('scrub').oninput = (e) => { stop(); state.frame = +e.target.value; draw(); };

document.addEventListener('keydown', (e) => {
  if (!state.scene || e.target.matches('input[type=color]')) return;
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); undo(); }
  else if (e.key === ' ' && !e.target.matches('button')) { e.preventDefault(); state.playing ? stop() : play(); }
  else if (e.key === 'ArrowRight') { stop(); state.frame = Math.min(state.scene.frames - 1, state.frame + 1); draw(); }
  else if (e.key === 'ArrowLeft') { stop(); state.frame = Math.max(0, state.frame - 1); draw(); }
  else if (e.key === 'Escape') select(null);
});

// Export ---------------------------------------------------------------------------------
$('export-lottie').onclick = () => worker.postMessage({ type: 'export', scene: state.scene, format: 'lottie' });
$('export-svg').onclick = () => worker.postMessage({ type: 'export', scene: state.scene, format: 'svg' });
function download({ lottie, svg, format }) {
  const blob = format === 'svg' ? new Blob([svg], { type: 'image/svg+xml' }) : new Blob([lottie], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = format === 'svg' ? 'unbaked.svg' : 'unbaked.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  setStatus(`Exported ${a.download}.`);
}
