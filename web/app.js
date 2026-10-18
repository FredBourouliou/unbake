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
  key: null,          // selected timeline key: { row, frame }
  undo: [],
  drag: null,
  hover: false,
};

// Worker --------------------------------------------------------------------------------
// Bump on each release so browsers fetch the new worker and Python files instead of cached ones.
const VERSION = '0.4.0';
const worker = new Worker(`worker.js?v=${VERSION}`, { type: 'module' });
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
  state.key = null;
  state.undo = [];
  baked.width = scene.width; baked.height = scene.height;
  $('scrub').max = scene.frames - 1;
  workspace.dataset.state = 'ready';
  ['export-lottie', 'export-svg', 'new'].forEach(id => $(id).disabled = false);
  const keys = scene.objects.reduce((n, o) => n + keyCount(o), 0);
  setStatus(`${scene.objects.length} objects, ${keys} keyframes, reconstructed in ${scene.seconds} s on this machine.`);
  renderList();
  renderInspector();
  draw();
}

const scale = () => unbaked.width / state.scene.width;

// Eased keyframes: [frame, value, [x1, y1, x2, y2]] where the control points describe the
// cubic-Bézier easing to the next key (as in CSS, After Effects and Lottie).
function bezierPoint(c, s) {
  const m = 1 - s;
  return [3 * m * m * s * c[0] + 3 * m * s * s * c[2] + s ** 3, 3 * m * m * s * c[1] + 3 * m * s * s * c[3] + s ** 3];
}
function solveS(c, u) {  // x(s) = u; x is monotonic because control points stay in the unit square
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) { const mid = (lo + hi) / 2; if (bezierPoint(c, mid)[0] < u) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
function sampleKeys(keys, f) {
  if (keys.length === 1 || f <= keys[0][0]) return keys[0][1];
  const last = keys[keys.length - 1];
  if (f >= last[0]) return last[1];
  let i = 1; while (keys[i][0] < f) i++;
  const [fa, va, c] = keys[i - 1], [fb, vb] = keys[i];
  const u = (f - fa) / (fb - fa);
  return va + (vb - va) * bezierPoint(c, solveS(c, u))[1];
}
// Insert a key at frame f without changing the motion: split the easing curve there (de Casteljau).
function ensureKey(keys, f) {
  const at = keys.findIndex(k => k[0] === f);
  if (at >= 0) return at;
  let i = keys.findIndex(k => k[0] > f);
  if (i <= 0) { keys.splice(i < 0 ? keys.length : 0, 0, [f, sampleKeys(keys, f), [1 / 3, 1 / 3, 2 / 3, 2 / 3]]); return keys.findIndex(k => k[0] === f); }
  const [fa, va, c] = keys[i - 1], vb = keys[i][1];
  const s = solveS(c, (f - fa) / (keys[i][0] - fa));
  const lerp = (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  const P0 = [0, 0], P1 = [c[0], c[1]], P2 = [c[2], c[3]], P3 = [1, 1];
  const A = lerp(P0, P1, s), B = lerp(P1, P2, s), C = lerp(P2, P3, s), D = lerp(A, B, s), E = lerp(B, C, s), M = lerp(D, E, s);
  const norm = (p, o, w) => [(p[0] - o[0]) / (w[0] || 1), (p[1] - o[1]) / (w[1] || 1)];
  const clamp = (v) => Math.min(1, Math.max(0, v));
  const left = [...norm(A, P0, M), ...norm(D, P0, M)].map(clamp);
  const right = [...norm(E, M, [1 - M[0], 1 - M[1]]), ...norm(C, M, [1 - M[0], 1 - M[1]])].map(clamp);
  keys[i - 1] = [fa, va, left];
  keys.splice(i, 0, [f, va + (vb - va) * M[1], right]);
  return i;
}
const keyCount = (o) => o.kind === 'stroke'
  ? o.start.length + (o.closed ? o.len.length : o.end.length)
  : o.x.length + o.y.length + o.rot.length + o.scale.length;
const positionFrames = (o) => [...new Set([...o.x.map(k => k[0]), ...o.y.map(k => k[0])])].sort((a, b) => a - b);

function trimAt(o, f) {
  if (o.hidden || f < o.span[0] || f > o.span[1]) return null;
  const start = sampleKeys(o.start, f);
  const len = o.closed ? sampleKeys(o.len, f) : Math.max(0, sampleKeys(o.end, f) - start);
  return { start, len };
}

// Pieces of a polyline between arc-length fractions; a closed path may wrap.
function subPath(points, closed, start, len) {
  const pts = closed ? [...points, points[0]] : points;
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const total = cum[cum.length - 1];
  const at = (s) => {
    let i = 1; while (i < cum.length - 1 && cum[i] < s) i++;
    const t = (s - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]);
    return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t];
  };
  const cut = (a, b) => [at(a * total), ...pts.filter((_, i) => cum[i] > a * total && cum[i] < b * total), at(b * total)];
  if (len >= 0.999) return [closed ? [...points, points[0]] : points];
  if (len <= 1e-4) return [];
  if (!closed) return [cut(Math.max(0, start), Math.min(1, start + len))];
  const s0 = ((start % 1) + 1) % 1, e0 = s0 + len;
  return e0 <= 1 ? [cut(s0, e0)] : [cut(s0, 1), cut(0, e0 - 1)];
}

function strokePath(points, S) {
  const path = new Path2D();
  points.forEach(([x, y], i) => i ? path.lineTo(x * S, y * S) : path.moveTo(x * S, y * S));
  return path;
}

function drawStroke(o, f, ctx) {
  const tr = trimAt(o, f);
  if (!tr) return;
  const S = scale();
  ctx.save();
  ctx.strokeStyle = o.color; ctx.lineWidth = o.width * S; ctx.lineCap = o.cap === 'round' ? 'round' : 'butt'; ctx.lineJoin = 'round';
  for (const piece of subPath(o.line, o.closed, tr.start, tr.len)) ctx.stroke(strokePath(piece, S));
  ctx.restore();
}

function params(o, f) {
  if (o.hidden || f < o.span[0] || f > o.span[1]) return null;
  const x = sampleKeys(o.x, f), y = sampleKeys(o.y, f);
  const r = sampleKeys(o.rot, f), s = sampleKeys(o.scale, f);
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

// A linear gradient lives in the object's local frame: map its two end points like the outline.
function gradientFill(o, p) {
  const S = scale(), c = Math.cos(p.r * Math.PI / 180), sn = Math.sin(p.r * Math.PI / 180);
  const at = ([px, py]) => [((px * c - py * sn) * p.s + p.x) * S, ((px * sn + py * c) * p.s + p.y) * S];
  const [x0, y0] = at(o.gradient.p0), [x1, y1] = at(o.gradient.p1);
  const g = uctx.createLinearGradient(x0, y0, x1, y1);
  g.addColorStop(0, o.gradient.colors[0]); g.addColorStop(1, o.gradient.colors[1]);
  return g;
}
const swatchFill = (o) => o.gradient ? `linear-gradient(135deg, ${o.gradient.colors[0]}, ${o.gradient.colors[1]})` : o.color;

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
    if (o.kind === 'stroke') { drawStroke(o, f, uctx); continue; }
    const p = params(o, f);
    if (!p) continue;
    uctx.fillStyle = o.gradient ? gradientFill(o, p) : o.color;
    uctx.fill(objectPath(o, p), 'evenodd');
  }
  const sel = sc.objects.find(o => o.id === state.selected);
  if (sel) drawSelection(sel, f);
  updateTimeline();
  $('scrub').value = f;
  $('frame-out').textContent = `${String(f + 1).padStart(String(sc.frames).length, '0')} / ${sc.frames}`;
}

function drawSelection(o, f) {
  const S = scale(), accent = getComputedStyle(document.documentElement).getPropertyValue('--accent');
  if (o.kind === 'stroke') {
    uctx.save();
    uctx.strokeStyle = accent; uctx.globalAlpha = 0.8; uctx.lineWidth = 1.5; uctx.setLineDash([4, 4]);
    uctx.stroke(strokePath(o.closed ? [...o.line, o.line[0]] : o.line, S));
    uctx.restore();
    return;
  }
  uctx.save();
  // Motion path over the visible span.
  uctx.beginPath();
  for (let t = o.span[0]; t <= o.span[1]; t++) {
    const x = sampleKeys(o.x, t), y = sampleKeys(o.y, t);
    t === o.span[0] ? uctx.moveTo(x * S, y * S) : uctx.lineTo(x * S, y * S);
  }
  uctx.strokeStyle = accent; uctx.globalAlpha = 0.7; uctx.lineWidth = 1.5; uctx.setLineDash([4, 4]); uctx.stroke();
  uctx.setLineDash([]); uctx.globalAlpha = 1;
  // Outline of the object on this frame.
  const p = params(o, f);
  if (p) { uctx.lineWidth = 2; uctx.stroke(objectPath(o, p)); }
  // Keyframe handles.
  for (const kf of positionFrames(o)) {
    const x = sampleKeys(o.x, kf), y = sampleKeys(o.y, kf);
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
    const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = swatchFill(o);
    const name = document.createElement('span'); name.textContent = `${o.kind === 'stroke' ? 'Stroke' : 'Object'} ${o.id + 1}`;
    const meta = document.createElement('span'); meta.className = 'mono';
    meta.textContent = `${keyCount(o)} keys`;
    b.append(sw, name, meta);
    b.onclick = () => select(o.id === state.selected ? null : o.id);
    li.append(b); ul.append(li);
  });
}

function renderInspector() {
  const o = selectedObject();
  $('inspector').hidden = !o;
  renderTimeline();
  if (!o) return;
  $('color').value = o.gradient ? o.gradient.colors[0] : o.color;
  $('color-label').textContent = o.gradient ? 'Start colour' : 'Colour';
  $('color2-field').hidden = !o.gradient;
  if (o.gradient) $('color2').value = o.gradient.colors[1];
  $('toggle-visible').textContent = o.hidden ? 'Show' : 'Hide';
  $('width-field').hidden = o.kind !== 'stroke';
  $('hint').textContent = o.kind === 'stroke'
    ? 'An animated stroke: it draws itself along this path. Drag it to move it, change its colour or width.'
    : 'Drag a handle to move a keyframe. Drag the object to move it, with its whole path.';
  $('hint').textContent += ' On the timeline, drag a key to retime it, double-click to add one, Delete to remove it.';
  if (o.kind === 'stroke') { $('width').value = o.width; $('width-out').textContent = o.width.toFixed(1); }
  const facts = o.kind === 'stroke'
    ? [['frames', `${o.span[0] + 1}–${o.span[1] + 1}`], ['path', o.closed ? 'closed loop' : 'open'],
       ['caps', o.cap], ['start keys', o.start.length], [o.closed ? 'length keys' : 'end keys', (o.closed ? o.len : o.end).length], ['colour', o.color]]
    : [['frames', `${o.span[0] + 1}–${o.span[1] + 1}`], ['position keys', positionFrames(o).length],
       ['rotation keys', o.rot.length], ['scale keys', o.scale.length],
       ...(o.gradient ? [['fill', 'gradient']] : [['colour', o.color]])];
  $('facts').replaceChildren();
  for (const [k, v] of facts) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    $('facts').append(dt, dd);
  }
}

const selectedObject = () => state.scene?.objects.find(o => o.id === state.selected);
function select(id) { state.selected = id; state.key = null; renderList(); renderInspector(); draw(); }

function remember() {
  state.undo.push(structuredClone(state.scene.objects));
  if (state.undo.length > 100) state.undo.shift();
  $('undo').disabled = false;
}
function undo() {
  if (!state.undo.length) return;
  state.scene.objects = state.undo.pop();
  state.key = null;
  $('undo').disabled = !state.undo.length;
  renderList(); renderInspector(); draw();
}

$('color').addEventListener('pointerdown', () => remember());
$('color').addEventListener('keydown', () => remember(), { once: false });
$('color').oninput = (e) => {
  const o = selectedObject(); if (!o) return;
  if (o.gradient) o.gradient.colors[0] = e.target.value; else o.color = e.target.value;
  renderList(); draw();
};
$('color2').addEventListener('pointerdown', () => remember());
$('color2').addEventListener('keydown', () => remember());
$('color2').oninput = (e) => { const o = selectedObject(); if (o?.gradient) { o.gradient.colors[1] = e.target.value; renderList(); draw(); } };
$('color').onchange = () => renderInspector();
$('width').addEventListener('pointerdown', () => remember());
$('width').oninput = (e) => { const o = selectedObject(); if (o && o.kind === 'stroke') { o.width = +e.target.value; $('width-out').textContent = o.width.toFixed(1); draw(); } };
$('toggle-visible').onclick = () => { const o = selectedObject(); if (!o) return; remember(); o.hidden = !o.hidden; renderList(); renderInspector(); draw(); };
$('reset').onclick = () => {
  const o = selectedObject(); if (!o) return; remember();
  const i = state.scene.objects.indexOf(o);
  state.scene.objects[i] = structuredClone(state.original.find(x => x.id === o.id));
  renderList(); renderInspector(); draw();
};
$('undo').onclick = undo;
$('new').onclick = () => { stop(); state.scene = null; workspace.dataset.state = 'empty'; $('objects').replaceChildren(); $('inspector').hidden = true; ['export-lottie', 'export-svg', 'undo', 'new'].forEach(id => $(id).disabled = true); setStatus(workerReady ? 'Ready. Drop a GIF or pick a sample.' : ''); };

// Timeline --------------------------------------------------------------------------------
// One row per channel of the selected object; each key is a diamond that can be retimed.
// Position joins the x and y channels: a key there moves both, as in After Effects.
function timelineRows(o) {
  if (o.kind === 'stroke') return [{ name: 'Start', chans: [o.start] }, { name: o.closed ? 'Length' : 'End', chans: [o.closed ? o.len : o.end] }];
  return [{ name: 'Position', chans: [o.x, o.y] }, { name: 'Rotation', chans: [o.rot] }, { name: 'Scale', chans: [o.scale] }];
}
const rowFrames = (row) => [...new Set(row.chans.flatMap(c => c.map(k => k[0])))].sort((a, b) => a - b);
const at = (f) => String(state.scene.frames > 1 ? f / (state.scene.frames - 1) : 0);

function renderTimeline() {
  const tl = $('timeline'), o = selectedObject();
  tl.hidden = !o;
  tl.replaceChildren();
  if (!o) return;
  timelineRows(o).forEach((row, r) => {
    const frames = rowFrames(row);
    const el = document.createElement('div'); el.className = 'tl-row';
    const label = document.createElement('span'); label.className = 'tl-label'; label.textContent = row.name;
    const track = document.createElement('div'); track.className = 'tl-track';
    track.title = 'Double-click to add a key';
    const span = document.createElement('span'); span.className = 'tl-span';
    span.style.setProperty('--at', at(o.span[0])); span.style.setProperty('--len', at(o.span[1] - o.span[0]));
    const head = document.createElement('span'); head.className = 'tl-head';
    track.append(span, head);
    for (const f of frames) {
      const k = document.createElement('button'); k.className = 'tl-key';
      k.dataset.row = r; k.dataset.frame = f;
      k.style.setProperty('--at', at(f));
      k.setAttribute('aria-label', `${row.name} key, frame ${f + 1}`);
      k.addEventListener('pointerdown', (e) => keyDown(e, r, f));
      track.append(k);
    }
    track.addEventListener('dblclick', (e) => { if (!e.target.classList.contains('tl-key')) addKey(r, frameAt(track, e.clientX)); });
    const count = document.createElement('span'); count.className = 'tl-count mono'; count.textContent = frames.length;
    el.append(label, track, count);
    tl.append(el);
  });
  updateTimeline();
}

// Cheap per-frame refresh: playhead, current and selected keys (no rebuild, so drags survive).
function updateTimeline() {
  const tl = $('timeline'); if (tl.hidden) return;
  tl.querySelectorAll('.tl-head').forEach(h => h.style.setProperty('--at', at(state.frame)));
  tl.querySelectorAll('.tl-key').forEach(k => {
    const f = +k.dataset.frame;
    k.classList.toggle('is-now', f === state.frame);
    k.setAttribute('aria-pressed', String(state.key?.row === +k.dataset.row && state.key?.frame === f));
  });
}

function frameAt(track, clientX) {
  const r = track.getBoundingClientRect(), pad = 8;
  const u = (clientX - r.left - pad) / Math.max(1, r.width - 2 * pad);
  return Math.round(Math.min(1, Math.max(0, u)) * (state.scene.frames - 1));
}

function keyDown(e, r, f) {
  e.preventDefault(); e.stopPropagation();
  stop();
  const o = selectedObject(), row = timelineRows(o)[r], frames = rowFrames(row), i = frames.indexOf(f);
  // A key cannot pass its neighbours, so the order of keys (and their easings) never changes.
  const lo = i > 0 ? frames[i - 1] + 1 : 0, hi = i < frames.length - 1 ? frames[i + 1] - 1 : state.scene.frames - 1;
  state.key = { row: r, frame: f };
  state.frame = f;
  remember();
  const key = e.currentTarget, track = key.parentElement;
  key.setPointerCapture(e.pointerId);
  $('timeline').classList.add('is-dragging');
  let now = f;
  const move = (ev) => {
    const g = Math.min(hi, Math.max(lo, frameAt(track, ev.clientX)));
    if (g === now) return;
    for (const c of row.chans) { const k = c.find(k => k[0] === now); if (k) k[0] = g; }
    now = g;
    key.dataset.frame = g; key.style.setProperty('--at', at(g));
    key.setAttribute('aria-label', `${row.name} key, frame ${g + 1}`);
    state.key.frame = g; state.frame = g;
    draw();
  };
  const up = () => {
    key.removeEventListener('pointermove', move);
    key.removeEventListener('pointerup', up);
    key.removeEventListener('pointercancel', up);
    $('timeline').classList.remove('is-dragging');
    if (now === f) { state.undo.pop(); $('undo').disabled = !state.undo.length; }
    renderInspector();
    key.isConnected || $('timeline').querySelector(`.tl-key[data-row="${r}"][data-frame="${now}"]`)?.focus();
  };
  key.addEventListener('pointermove', move);
  key.addEventListener('pointerup', up);
  key.addEventListener('pointercancel', up);
  draw();
}

// Add a key without changing the motion: the easing curve is split there.
function addKey(r, f) {
  const o = selectedObject(), row = timelineRows(o)[r];
  stop();
  if (row.chans.every(c => c.some(k => k[0] === f))) { state.key = { row: r, frame: f }; state.frame = f; draw(); return; }
  remember();
  for (const c of row.chans) ensureKey(c, f);
  state.key = { row: r, frame: f }; state.frame = f;
  renderList(); renderInspector(); draw();
}

// Remove the selected key; the previous key keeps its easing, now running to the next one.
function deleteKey() {
  const o = selectedObject(); if (!o || !state.key) return;
  const row = timelineRows(o)[state.key.row], f = state.key.frame;
  const removable = row.chans.filter(c => c.length > 1 && c.some(k => k[0] === f));
  if (!removable.length) { setStatus('A channel keeps at least one key.'); return; }
  remember();
  for (const c of removable) c.splice(c.findIndex(k => k[0] === f), 1);
  state.key = null;
  renderList(); renderInspector(); draw();
}

// Canvas interaction ----------------------------------------------------------------------
function canvasPoint(e) {
  const r = unbaked.getBoundingClientRect();
  return { x: (e.clientX - r.left) * unbaked.width / r.width, y: (e.clientY - r.top) * unbaked.height / r.height };
}

function hit(pt) {
  const S = scale(), sel = selectedObject();
  if (sel && sel.kind !== 'stroke') {
    // The handle of the current frame sits under the object itself: grabbing there moves the object.
    const f = positionFrames(sel).find(kf => kf !== state.frame &&
      Math.hypot(sampleKeys(sel.x, kf) * S - pt.x, sampleKeys(sel.y, kf) * S - pt.y) <= HANDLE_R + 3);
    if (f !== undefined) return { kind: 'key', obj: sel, frame: f };
  }
  const objs = [...state.scene.objects].reverse();
  for (const o of objs) {
    if (o.kind === 'stroke') {
      if (!trimAt(o, state.frame)) continue;
      uctx.save(); uctx.lineWidth = Math.max(o.width * S, 12);
      const onIt = uctx.isPointInStroke(strokePath(o.closed ? [...o.line, o.line[0]] : o.line, S), pt.x, pt.y);
      uctx.restore();
      if (onIt) return { kind: 'body', obj: o };
      continue;
    }
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
  if (h.kind === 'key') { ensureKey(h.obj.x, h.frame); ensureKey(h.obj.y, h.frame); }
  const snapshot = h.obj.kind === 'stroke' ? h.obj.line : { x: h.obj.x, y: h.obj.y };
  state.drag = { ...h, start: pt, keys: structuredClone(snapshot) };
  unbaked.setPointerCapture(e.pointerId);
  unbaked.classList.add('is-dragging');
});
unbaked.addEventListener('pointermove', (e) => {
  if (!state.scene) return;
  const pt = canvasPoint(e);
  if (!state.drag) { unbaked.classList.toggle('is-hover', !!hit(pt)); return; }
  const S = scale(), dx = (pt.x - state.drag.start.x) / S, dy = (pt.y - state.drag.start.y) / S;
  const o = state.drag.obj;
  if (o.kind === 'stroke') {
    o.line = state.drag.keys.map(([x, y]) => [+(x + dx).toFixed(2), +(y + dy).toFixed(2)]);
  } else if (state.drag.kind === 'key') {
    const f = state.drag.frame;
    o.x = state.drag.keys.x.map(k => k[0] === f ? [f, +(k[1] + dx).toFixed(2), k[2]] : k);
    o.y = state.drag.keys.y.map(k => k[0] === f ? [f, +(k[1] + dy).toFixed(2), k[2]] : k);
  } else {
    o.x = state.drag.keys.x.map(([f, v, c]) => [f, +(v + dx).toFixed(2), c]);
    o.y = state.drag.keys.y.map(([f, v, c]) => [f, +(v + dy).toFixed(2), c]);
  }
  draw();
});
unbaked.addEventListener('pointerup', () => {
  if (!state.drag) return;
  const o = state.drag.obj;
  const moved = JSON.stringify(state.drag.keys) !== JSON.stringify(o.kind === 'stroke' ? o.line : { x: o.x, y: o.y });
  if (!moved) state.undo.pop();
  else { renderList(); renderInspector(); }  // a handle drag may have added keys
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
  else if ((e.key === 'Delete' || e.key === 'Backspace') && state.key) { e.preventDefault(); deleteKey(); }
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
