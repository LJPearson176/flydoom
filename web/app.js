const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let playing = true;
let flyEye = false;

function setPlaying(next) {
  playing = next;
  $('#runState').textContent = playing ? 'LIVE' : 'PAUSED';
  $('#pauseButton').textContent = playing ? 'Ⅱ' : '▶';
  $('#playButton').textContent = playing ? 'Ⅱ' : '▶';
  document.body.classList.toggle('is-paused', !playing);
}

$('#pauseButton').addEventListener('click', () => setPlaying(!playing));
$('#playButton').addEventListener('click', () => setPlaying(!playing));

$$('.mode-tab').forEach((button) => button.addEventListener('click', (e) => {
  if (button.tagName.toLowerCase() === 'a') return;
  $$('.mode-tab').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
  const label = button.textContent.trim();
  const showBenchmark = ['Compare', 'Experiment'].includes(label);
  const isAnatomy = label.startsWith('Anatomy');
  const isBrain3D = label.startsWith('Brain');

  document.body.classList.toggle('benchmark-active', showBenchmark);
  document.body.classList.toggle('anatomy-active', isAnatomy);
  document.body.classList.toggle('brain-3d-active', isBrain3D);
}));

$('#cameraButton').addEventListener('click', () => {
  flyEye = !flyEye;
  $('#cameraButton').innerHTML = flyEye ? 'SWITCH TO OBSERVER CAMERA <span>↗</span>' : 'SWITCH TO FLY-EYE <span>↗</span>';
  $('.doom-view').classList.toggle('fly-eye-active', flyEye);
  $('.doom-label').textContent = flyEye ? 'FLY-EYE / MOTION ENERGY' : 'GZDOOM / E1M1';
});

$$('.seg').forEach((button) => button.addEventListener('click', () => {
  $$('.seg').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
}));

const traceButton = $('#traceButton');
if (traceButton) traceButton.addEventListener('click', () => {
  const nodes = $$('.trace-node');
  nodes.forEach((node, index) => setTimeout(() => node.classList.add('trace-flash'), index * 140));
  setTimeout(() => nodes.forEach((node) => node.classList.remove('trace-flash')), nodes.length * 140 + 500);
});

let t = 56;
const progress = $('#timelineProgress');
const handle = $('#timelineHandle');
function animate() {
  if (playing) {
    t += 0.035;
    if (t > 86) t = 18;
    progress.style.width = `${t}%`;
    handle.style.left = `${t}%`;
  }
  requestAnimationFrame(animate);
}
animate();

// --- LIVE NATIVE GZDOOM TELEMETRY ---
// The static file view remains fully usable; this activates when served by
// observatory.live_server.py on the same origin.
const liveFrame = $('#liveFrame');
let lastLiveFrame = 0;
let lastObsHealth = 100;
let hudEnabled = true;

const toggleHudButton = $('#toggleHudButton');
if (toggleHudButton) {
  toggleHudButton.addEventListener('click', () => {
    hudEnabled = !hudEnabled;
    const overlay = $('#neuralHudOverlay');
    if (overlay) overlay.classList.toggle('is-hidden', !hudEnabled);
    toggleHudButton.innerHTML = hudEnabled ? '2D HUD: <span style="color:var(--cyan);">ON</span>' : '2D HUD: <span style="color:var(--muted);">OFF</span>';
  });
}

// --- IN-SITU ISOMETRIC 3D BRAIN HOLOGRAM OVERLAY ---
let holoEnabled = true;
const toggle3dHoloButton = $('#toggle3dHoloButton');
const holoCanvas = $('#doom3dHologram');

if (toggle3dHoloButton && holoCanvas) {
  toggle3dHoloButton.addEventListener('click', () => {
    holoEnabled = !holoEnabled;
    holoCanvas.classList.toggle('is-hidden', !holoEnabled);
    const textSpan = $('#holoStateText');
    if (textSpan) textSpan.textContent = holoEnabled ? 'ON' : 'OFF';
    toggle3dHoloButton.style.borderColor = holoEnabled ? '#00e5ff' : 'var(--line)';
    toggle3dHoloButton.style.color = holoEnabled ? '#00e5ff' : 'var(--muted)';
  });

  holoCanvas.addEventListener('dblclick', () => {
    holoCanvas.classList.toggle('is-full');
  });
}

const holoState = {
  yaw: 0.22,
  pitch: 0.20,
  dist: 680.0,
  target: [0.0, -70.0, 0.0],
  isDragging: false,
  lastX: 0,
  lastY: 0,
  asym: 0.0,
  action: 'FORWARD',
  isFire: false,
  isDoor: false,
  isDamage: false,
  pulsePhase: 0.0,
};

let cnsModel3D = null;
let cnsMeshData = null;

Promise.all([
  fetch('./fly_3d_cns_model.json').then(r => r.json()),
  fetch('./authentic_fly_cns_mesh.json').then(r => r.json()),
]).then(([model, mesh]) => {
  cnsModel3D = model;
  cnsMeshData = mesh;
  initHologramRenderer();
}).catch(err => console.warn('Could not load 3D CNS model for hologram:', err));

function initHologramRenderer() {
  if (!holoCanvas) return;
  const ctx = holoCanvas.getContext('2d');

  function resizeHolo() {
    const rect = holoCanvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    holoCanvas.width = rect.width * window.devicePixelRatio;
    holoCanvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
  resizeHolo();
  window.addEventListener('resize', resizeHolo);

  holoCanvas.addEventListener('mousedown', e => {
    holoState.isDragging = true;
    holoState.lastX = e.clientX;
    holoState.lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => { holoState.isDragging = false; });
  window.addEventListener('mousemove', e => {
    if (!holoState.isDragging) return;
    const dx = e.clientX - holoState.lastX;
    const dy = e.clientY - holoState.lastY;
    holoState.lastX = e.clientX;
    holoState.lastY = e.clientY;
    holoState.yaw += dx * 0.01;
    holoState.pitch = Math.max(-1.4, Math.min(1.4, holoState.pitch + dy * 0.01));
  });

  function projectHolo(x, y, z, w, h) {
    const tx = x - holoState.target[0];
    const ty = y - holoState.target[1];
    const tz = z - holoState.target[2];
    const cosY = Math.cos(holoState.yaw), sinY = Math.sin(holoState.yaw);
    const x1 = tx * cosY - tz * sinY;
    const z1 = tx * sinY + tz * cosY;
    const cosP = Math.cos(holoState.pitch), sinP = Math.sin(holoState.pitch);
    const y2 = ty * cosP - z1 * sinP;
    const z2 = ty * sinP + z1 * cosP;
    const eyeZ = z2 + holoState.dist;
    if (eyeZ <= 10.0) return null;
    const scale = 580.0 / eyeZ;
    return { x: w / 2 + x1 * scale, y: h / 2 - y2 * scale, scale: scale };
  }

  function renderHoloFrame() {
    requestAnimationFrame(renderHoloFrame);
    if (!holoEnabled || !cnsModel3D || holoCanvas.classList.contains('is-hidden')) return;

    const w = holoCanvas.width / window.devicePixelRatio;
    const h = holoCanvas.height / window.devicePixelRatio;
    ctx.clearRect(0, 0, w, h);

    holoState.pulsePhase += 0.04;

    // 1. Exoskeleton Cuticle Wireframe
    const lines = cnsModel3D.exoskeleton.lines;
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      const p1 = projectHolo(l[0], l[1], l[2], w, h);
      const p2 = projectHolo(l[3], l[4], l[5], w, h);
      if (!p1 || !p2) continue;

      let stroke = 'rgba(0, 229, 255, 0.28)';
      if (l[6] === 'eye') stroke = 'rgba(140, 90, 255, 0.45)';
      else if (l[6] === 'wing') stroke = 'rgba(100, 200, 255, 0.3)';

      ctx.strokeStyle = stroke;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // 2. Sampled Brain Mesh Surface
    if (cnsMeshData && cnsMeshData.brain_surface_edges) {
      ctx.strokeStyle = 'rgba(30, 100, 160, 0.18)';
      ctx.lineWidth = 0.6;
      const bEdges = cnsMeshData.brain_surface_edges;
      for (let i = 0; i < bEdges.length; i += 4) {
        const e = bEdges[i];
        const p1 = projectHolo(e[0], e[1], e[2], w, h);
        const p2 = projectHolo(e[3], e[4], e[5], w, h);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }

function blendRgb(hex, targetR, targetG, targetB, t) {
  t = Math.max(0, Math.min(1, t));
  const num = parseInt(hex.replace('#', ''), 16) || 0;
  const r1 = (num >> 16) & 255;
  const g1 = (num >> 8) & 255;
  const b1 = num & 255;
  const r = Math.round(r1 * (1 - t) + targetR * t);
  const g = Math.round(g1 * (1 - t) + targetG * t);
  const b = Math.round(b1 * (1 - t) + targetB * t);
  return `rgb(${r},${g},${b})`;
}

    // 3. Dense FlyWire Fibers with Dynamic Activity Illumination
    const fibers = cnsModel3D.connectome.fibers;
    const asym = holoState.asym;
    const isFire = holoState.isFire;
    const isDoor = holoState.isDoor;

    for (let i = 0; i < fibers.length; i += 2) {
      const f = fibers[i];
      const pts = f.points;
      if (pts.length < 2) continue;

      let color = f.color;
      let alpha = 0.55;
      let lw = 0.9;

      const isRight = f.neuropil.endsWith('_R') || pts[0][0] > 0;
      const isLeft = f.neuropil.endsWith('_L') || pts[0][0] < 0;
      const drive = isRight ? Math.max(0, asym) : (isLeft ? Math.max(0, -asym) : 0);

      if (isFire && (f.neuropil.startsWith('LO_') || f.neuropil.startsWith('VNC_') || f.neuropil === 'DN_TRUNK')) {
        color = '#ff3300';
        alpha = 1.0;
        lw = 1.6;
      } else if (isDoor && (f.neuropil === 'SEZ' || f.neuropil === 'VNC_T1')) {
        color = '#00ffaa';
        alpha = 1.0;
        lw = 1.6;
      } else if (drive > 0.04) {
        const blend = Math.min(1.0, (drive - 0.04) / 0.45);
        color = blendRgb(f.color, 255, 160, 0, blend);
        alpha = 0.55 + 0.4 * blend;
        lw = 0.9 + 0.7 * blend;
      }

      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let started = false;
      for (let j = 0; j < pts.length; j++) {
        const p = projectHolo(pts[j][0], pts[j][1], pts[j][2], w, h);
        if (!p) { started = false; continue; }
        if (!started) { ctx.moveTo(p.x, p.y); started = true; }
        else { ctx.lineTo(p.x, p.y); }
      }
      if (started) ctx.stroke();
      ctx.globalAlpha = 1.0;

      // Action Potential Pulse
      if (i % 16 === 0) {
        const t = (holoState.pulsePhase + i * 0.06) % 1.0;
        const idx = Math.min(pts.length - 2, Math.floor(t * (pts.length - 1)));
        const frac = (t * (pts.length - 1)) - idx;
        const pA = pts[idx], pB = pts[idx + 1];
        const px = pA[0] + (pB[0] - pA[0]) * frac;
        const py = pA[1] + (pB[1] - pA[1]) * frac;
        const pz = pA[2] + (pB[2] - pA[2]) * frac;
        const pScreen = projectHolo(px, py, pz, w, h);
        if (pScreen) {
          ctx.fillStyle = (lw > 1.2) ? '#ffea00' : '#00ffff';
          ctx.beginPath();
          ctx.arc(pScreen.x, pScreen.y, 1.3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // 4. High-Fidelity Retinotopic Columns, SWC Neurons & Chemical Synapses
    if (cnsModel3D.high_fidelity_pathways) {
      const hfp = cnsModel3D.high_fidelity_pathways;

      // SWC Morphology Skeletons
      if (hfp.neurons) {
        for (let i = 0; i < hfp.neurons.length; i++) {
          const nrn = hfp.neurons[i];
          const nodeMap = new Map();
          for (let k = 0; k < nrn.nodes.length; k++) {
            nodeMap.set(nrn.nodes[k][0], nrn.nodes[k]);
          }

          let stroke = nrn.color_hex;
          let lw = 1.6;
          const nrnDrive = (nrn.cell_type === 'T4a' || nrn.cell_type === 'Mi1' || nrn.cell_type === 'Tm3')
            ? Math.max(0, asym)
            : ((nrn.cell_type === 'Mi4' || nrn.cell_type === 'Mi9') ? Math.max(0, -asym) : 0);

          if (isFire && nrn.cell_type === 'DNpe017') {
            stroke = '#ff1744';
            lw = 2.8;
          } else if (nrnDrive > 0.04) {
            const blend = Math.min(1.0, (nrnDrive - 0.04) / 0.45);
            stroke = (nrn.cell_type === 'Mi4' || nrn.cell_type === 'Mi9')
              ? blendRgb(nrn.color_hex, 244, 63, 94, blend)
              : blendRgb(nrn.color_hex, 255, 234, 0, blend);
            lw = 1.6 + 0.8 * blend;
          }

          ctx.strokeStyle = stroke;
          ctx.lineWidth = lw;
          ctx.globalAlpha = 0.95;

          for (let k = 0; k < nrn.nodes.length; k++) {
            const nd = nrn.nodes[k];
            const pId = nd[6];
            if (pId === -1) {
              const pS = projectHolo(nd[2], nd[3], nd[4], w, h);
              if (pS) {
                ctx.fillStyle = stroke;
                ctx.beginPath();
                ctx.arc(pS.x, pS.y, Math.max(2.5, nd[5] * 2.2 * pS.scale), 0, 2 * Math.PI);
                ctx.fill();
              }
              continue;
            }
            const pNd = nodeMap.get(pId);
            if (!pNd) continue;
            const p1 = projectHolo(nd[2], nd[3], nd[4], w, h);
            const p2 = projectHolo(pNd[2], pNd[3], pNd[4], w, h);
            if (!p1 || !p2) continue;

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }

      // Chemical Synapses Point Cloud
      if (hfp.synapses) {
        for (let i = 0; i < hfp.synapses.length; i += 2) {
          const syn = hfp.synapses[i];
          const p = projectHolo(syn.pos[0], syn.pos[1], syn.pos[2], w, h);
          if (!p) continue;

          let col = syn.color || '#00e5ff';
          let r = Math.max(1.8, (1.6 + syn.weight * 1.4) * p.scale);
          if (isFire && syn.pre_type === 'DNpe017') {
            col = '#ff0033';
            r *= 1.8;
          } else if (asym > 0.04 && syn.pre_type === 'Mi1') {
            const blend = Math.min(1.0, (asym - 0.04) / 0.45);
            col = blendRgb('#00e5ff', 255, 221, 0, blend);
            r *= (1.0 + 0.5 * blend);
          }

          ctx.fillStyle = col;
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1.0;
    }

    // Hologram Header Badge
    ctx.fillStyle = 'rgba(0, 229, 255, 0.85)';
    ctx.font = '9px monospace';
    ctx.fillText('3D CONNECTOME TWIN', 8, 14);
  }

  requestAnimationFrame(renderHoloFrame);
}

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value;
}

const activationHistory = [];

function drawActivationTrace(neural) {
  const canvas = document.getElementById('activationTraceCanvas');
  if (!canvas) return;
  const sample = {
    lLeading: Number(neural.t4_l_leading_activation ?? 0),
    lCentral: Number(neural.t4_l_central_activation ?? 0),
    lTrailing: Number(neural.t4_l_trailing_activation ?? 0),
    lSoma: Number(neural.t4_l_soma_activation ?? 0),
    rLeading: Number(neural.t4_r_leading_activation ?? 0),
    rCentral: Number(neural.t4_r_central_activation ?? 0),
    rTrailing: Number(neural.t4_r_trailing_activation ?? 0),
    rSoma: Number(neural.t4_r_soma_activation ?? 0),
    lSpike: Number(neural.t4_l_spike ?? 0) > 0,
    rSpike: Number(neural.t4_r_spike ?? 0) > 0,
  };
  activationHistory.push(sample);
  if (activationHistory.length > 64) activationHistory.shift();

  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0a171c';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#19313a';
  ctx.lineWidth = 1;
  for (const fraction of [0.25, 0.5, 0.75]) {
    const y = height - fraction * height;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }

  const traces = [
    ['lLeading', '#58dfc2', false], ['lCentral', '#71aef5', false],
    ['lTrailing', '#f0a65a', false], ['lSoma', '#dce8e5', false],
    ['rLeading', '#58dfc2', true], ['rCentral', '#71aef5', true],
    ['rTrailing', '#f0a65a', true], ['rSoma', '#dce8e5', true],
  ];
  traces.forEach(([key, color, dashed]) => {
    ctx.strokeStyle = color;
    ctx.globalAlpha = dashed ? 0.48 : 0.92;
    ctx.setLineDash(dashed ? [3, 3] : []);
    ctx.lineWidth = key.endsWith('Soma') ? 1.8 : 1;
    ctx.beginPath();
    activationHistory.forEach((entry, index) => {
      const x = activationHistory.length <= 1 ? 0 : index * width / (activationHistory.length - 1);
      const value = Math.max(0, Math.min(1, entry[key]));
      const y = height - value * (height - 4) - 2;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  activationHistory.forEach((entry, index) => {
    if (!entry.lSpike && !entry.rSpike) return;
    const x = activationHistory.length <= 1 ? 0 : index * width / (activationHistory.length - 1);
    ctx.strokeStyle = '#f27663';
    ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function setActivationBar(id, value) {
  const node = document.getElementById(id);
  if (!node) return;
  const activation = Math.max(0, Math.min(1, Number(value ?? 0)));
  node.style.transform = `scaleX(${activation.toFixed(3)})`;
  node.setAttribute('title', `${(activation * 100).toFixed(1)}% activation`);
}

function updateActivationMap(neural) {
  const keys = [
    ['t4LLeading', 't4_l_leading_activation'],
    ['t4LCentral', 't4_l_central_activation'],
    ['t4LTrailing', 't4_l_trailing_activation'],
    ['t4LSoma', 't4_l_soma_activation'],
    ['t4RLeading', 't4_r_leading_activation'],
    ['t4RCentral', 't4_r_central_activation'],
    ['t4RTrailing', 't4_r_trailing_activation'],
    ['t4RSoma', 't4_r_soma_activation'],
  ];
  keys.forEach(([id, key]) => setActivationBar(id, neural[key]));
  const driveText = (side) => {
    const p = side === 'L' ? 't4_l_' : 't4_r_';
    const f = (key) => Number(neural[p + key] ?? 0).toFixed(2);
    return `${side} input/g · Mi1 ${f('mi1_input')}/${f('mi1_drive')} · Tm3 ${f('tm3_input')}/${f('tm3_drive')} · Mi4 ${f('mi4_input')}/${f('mi4_inhibition')} · Mi9 ${f('mi9_input')}/${f('mi9_inhibition')}`;
  };
  setText('#t4LDrives', driveText('L'));
  setText('#t4RDrives', driveText('R'));
  setText('#retinaDriveL', `L ${Number(neural.retina_left_drive ?? 0).toFixed(2)}`);
  setText('#retinaDriveC', `CENTER ${Number(neural.retina_center_drive ?? 0).toFixed(2)}`);
  setText('#retinaDriveR', `R ${Number(neural.retina_right_drive ?? 0).toFixed(2)}`);
  drawActivationTrace(neural);
}

function updateLiveView(data) {
  const live = Boolean(data.connected && data.source === 'native_gzdoom');
  setText('#environmentStatus', live ? 'GZDOOM.APP · LIVE CAPTURE' : 'GZDOOM.APP · NATIVE TARGET');
  if (!live) return;

  const neural = data.neural || {};
  const nativeState = data.native_game_state || {};
  const asym = Number(neural.norm_asymmetry || 0);
  const left = Math.max(0, Math.min(1, (1 - asym) / 2));
  const right = Math.max(0, Math.min(1, (1 + asym) / 2));
  const currentHealth = Math.round(Number(data.health ?? 100));

  setText('#healthValue', currentHealth);
  setText('#ammoValue', Math.round(Number(data.ammo ?? 0)));
  setText('#enemyValue', String(Math.round(Number(data.kills ?? 0))).padStart(2, '0'));
  setText('#leftMotor', left.toFixed(2));
  setText('#rightMotor', right.toFixed(2));
  setText('#actionOutput', data.action || 'NOOP');
  setText('#asymmetryValue', `Δ motor asymmetry ${asym >= 0 ? '+' : ''}${asym.toFixed(3)}`);

  const bars = $$('.motor-bar i');
  if (bars[0]) bars[0].style.width = `${left * 100}%`;
  if (bars[1]) bars[1].style.width = `${right * 100}%`;

  // Update 3D Brain Hologram Live State
  holoState.asym = asym;
  holoState.action = data.action || 'NOOP';
  holoState.isFire = (data.action === 'FIRE');
  holoState.isDoor = (data.action === 'USE');
  holoState.isDamage = (currentHealth < 40);
  updateActivationMap(neural);

  // Population Optic Flow, Central Complex & Mushroom Body Updates
  const flowDiv = Number(neural.flow_divergence ?? 0);
  const flowCurl = Number(neural.flow_curl ?? 0);
  const ebHead = Number(neural.eb_heading_deg ?? 0);
  const ebCoh = Number(neural.eb_bump_coherence ?? 0);
  const mbVal = Number(neural.mb_valence ?? 0);
  const mbDa = Number(neural.mb_ppl1_da ?? 0);

  setText('#flowDivVal', `${flowDiv >= 0 ? '+' : ''}${flowDiv.toFixed(2)}`);
  setText('#flowCurlVal', `${flowCurl >= 0 ? '+' : ''}${flowCurl.toFixed(2)}`);
  setText('#ebHeadingVal', `${ebHead >= 0 ? '+' : ''}${ebHead.toFixed(1)}° (R=${ebCoh.toFixed(2)})`);
  setText('#mbValenceVal', `${mbVal >= 0 ? '+' : ''}${mbVal.toFixed(2)} (PPL1 DA ${mbDa.toFixed(2)})`);

  const divBar = $('#flowDivBar');
  if (divBar) divBar.style.width = `${Math.max(0, Math.min(100, 50 + flowDiv * 50))}%`;
  const curlBar = $('#flowCurlBar');
  if (curlBar) curlBar.style.width = `${Math.max(0, Math.min(100, 50 + flowCurl * 50))}%`;

  // --- IN-SITU NEURAL HUD UPDATES ---
  if (hudEnabled) {
    const t4L = Number(neural.t4_l_v ?? -65.0);
    const t4R = Number(neural.t4_r_v ?? -65.0);
    setText('#hudOpticValL', `${t4L.toFixed(1)}mV`);
    setText('#hudOpticValR', `${t4R.toFixed(1)}mV`);

    // Optic lobe glow intensity (depolarization increases brightness)
    const glowL = $('#hudOpticGlowL');
    const glowR = $('#hudOpticGlowR');
    if (glowL) glowL.setAttribute('opacity', Math.max(0.2, Math.min(0.95, (t4L + 65.0) / 20.0 + 0.3)).toFixed(2));
    if (glowR) glowR.setAttribute('opacity', Math.max(0.2, Math.min(0.95, (t4R + 65.0) / 20.0 + 0.3)).toFixed(2));

    // CX Central Complex heading vs target angle
    const targetDeg = Number(neural.target_angle_deg ?? NaN);
    const agentDeg = Number(nativeState.angle_deg ?? 0.0);
    const pointer = $('#hudTargetPointer');
    const lockRing = $('#hudLockRing');
    const lockText = $('#hudLockText');

    if (!isNaN(targetDeg) && pointer) {
      const relDiff = ((targetDeg - agentDeg + 180) % 360) - 180;
      pointer.setAttribute('transform', `rotate(${relDiff.toFixed(1)})`);

      const isLocked = Math.abs(relDiff) <= 18.0;
      if (lockRing) lockRing.classList.toggle('locked', isLocked);
      if (lockText) {
        if (Number(neural.door_candidate) > 0) {
          lockText.textContent = 'DOOR ALIGN';
        } else if (Number(neural.combat_active) > 0) {
          lockText.textContent = isLocked ? 'TARGET LOCK' : 'COMBAT SCAN';
        } else {
          lockText.textContent = 'CX HEADING';
        }
      }
    }

    // Motor tracks & asymmetry bar
    const asymIndicator = $('#hudAsymIndicator');
    if (asymIndicator) asymIndicator.setAttribute('x', (198 + asym * 45).toFixed(1));
    const dnL = $('#hudDnLeft');
    const dnR = $('#hudDnRight');
    if (dnL) dnL.setAttribute('opacity', (0.3 + left * 0.7).toFixed(2));
    if (dnR) dnR.setAttribute('opacity', (0.3 + right * 0.7).toFixed(2));

    // Action Badge
    const action = data.action || 'NOOP';
    setText('#hudActionName', action);
    const badge = $('#hudActionBadge');
    if (badge) {
      badge.classList.toggle('action-fire', action === 'FIRE');
      badge.classList.toggle('action-use', action === 'USE');
    }
    let circuitDesc = 'LPTC BILATERAL BALANCE';
    if (action === 'FIRE') circuitDesc = 'TARGET ACQUIRED · STRIKE COMMAND';
    else if (action === 'USE') circuitDesc = 'MECHANOSENSORY PROBOSCIS INTERACT';
    else if (action === 'TURN_LEFT' || action === 'TURN_RIGHT') circuitDesc = 'DESCENDING STEERING (DNpe017)';
    else if (Number(neural.health_priority_active) > 0) circuitDesc = 'NOCICEPTIVE EVASION CIRCUIT';
    setText('#hudActionCircuit', circuitDesc);

    // Nociceptive damage vignette flash
    if (currentHealth < lastObsHealth) {
      const vignette = $('#damageVignette');
      if (vignette) {
        vignette.classList.add('damage-flash');
        setTimeout(() => vignette.classList.remove('damage-flash'), 500);
      }
    }
    lastObsHealth = currentHealth;
  }

  if (liveFrame && data.updated_at !== lastLiveFrame) {
    lastLiveFrame = data.updated_at;
    liveFrame.src = `/api/frame?ts=${encodeURIComponent(lastLiveFrame)}`;
    liveFrame.classList.add('is-live');
  }
}

async function pollLiveTelemetry() {
  if (location.protocol === 'file:') return;
  try {
    const response = await fetch('/api/telemetry', { cache: 'no-store' });
    if (response.ok) updateLiveView(await response.json());
  } catch (error) {
    // Static/demo mode should remain quiet when the live server is absent.
  }
}

pollLiveTelemetry();
setInterval(pollLiveTelemetry, 250);

// --- RECORDED RUN REPLAY SUPPORT ---
const runSelect = $('#runSelect');
let replayTicks = [];
let replayIndex = 0;
let replayTimer = null;

async function loadRunsList() {
  if (!runSelect || location.protocol === 'file:') return;
  try {
    const res = await fetch('/api/runs');
    if (res.ok) {
      const runs = await res.json();
      runSelect.innerHTML = '<option value="live">🔴 LIVE GZDOOM</option>' +
        runs.map(r => `<option value="${r.id}">${r.name} (${r.steps} steps, ${r.kills} kills, ${r.damage} dmg)</option>`).join('');
    }
  } catch (err) {}
}

if (runSelect) {
  runSelect.addEventListener('change', async () => {
    const val = runSelect.value;
    if (val === 'live') {
      if (replayTimer) clearInterval(replayTimer);
      replayTicks = [];
      setText('#runState', 'LIVE');
      return;
    }
    try {
      const res = await fetch(`/api/episode?id=${encodeURIComponent(val)}`);
      if (res.ok) {
        const ep = await res.json();
        replayTicks = ep.trajectory || [];
        replayIndex = 0;
        setText('#runState', 'REPLAY');
        if (replayTimer) clearInterval(replayTimer);
        replayTimer = setInterval(() => {
          if (!playing || replayTicks.length === 0) return;
          const tick = replayTicks[replayIndex];
          updateLiveView({
            connected: true,
            source: 'native_gzdoom',
            action: tick.action,
            step: tick.step,
            health: tick.health,
            ammo: tick.ammo,
            kills: tick.kills,
            damage_dealt: tick.damage_dealt,
            neural: {
              norm_asymmetry: tick.norm_asymmetry,
              t4_l_v: tick.t4_l_v,
              t4_r_v: tick.t4_r_v,
              target_angle_deg: tick.target_angle_deg,
              door_candidate: tick.door_candidate,
              combat_active: tick.combat_active,
              health_priority_active: tick.health_priority_active,
              t4_l_leading_activation: tick.t4_l_leading_activation,
              t4_l_central_activation: tick.t4_l_central_activation,
              t4_l_trailing_activation: tick.t4_l_trailing_activation,
              t4_l_soma_activation: tick.t4_l_soma_activation,
              t4_l_mi1_drive: tick.t4_l_mi1_drive,
              t4_l_tm3_drive: tick.t4_l_tm3_drive,
              t4_l_mi4_inhibition: tick.t4_l_mi4_inhibition,
              t4_l_mi9_inhibition: tick.t4_l_mi9_inhibition,
              t4_r_leading_activation: tick.t4_r_leading_activation,
              t4_r_central_activation: tick.t4_r_central_activation,
              t4_r_trailing_activation: tick.t4_r_trailing_activation,
              t4_r_soma_activation: tick.t4_r_soma_activation,
              t4_r_mi1_drive: tick.t4_r_mi1_drive,
              t4_r_tm3_drive: tick.t4_r_tm3_drive,
              t4_r_mi4_inhibition: tick.t4_r_mi4_inhibition,
              t4_r_mi9_inhibition: tick.t4_r_mi9_inhibition,
              retina_left_drive: tick.retina_left_drive,
              retina_right_drive: tick.retina_right_drive,
              retina_center_drive: tick.retina_center_drive,
              t4_l_mi1_input: tick.t4_l_mi1_input,
              t4_l_tm3_input: tick.t4_l_tm3_input,
              t4_l_mi4_input: tick.t4_l_mi4_input,
              t4_l_mi9_input: tick.t4_l_mi9_input,
              t4_r_mi1_input: tick.t4_r_mi1_input,
              t4_r_tm3_input: tick.t4_r_tm3_input,
              t4_r_mi4_input: tick.t4_r_mi4_input,
              t4_r_mi9_input: tick.t4_r_mi9_input,
            },
            native_game_state: {
              angle_deg: tick.angle_deg,
              x: tick.x,
              y: tick.y,
              speed: tick.linear_velocity,
            },
          });
          replayIndex = (replayIndex + 1) % replayTicks.length;
        }, 120);
      }
    } catch (e) {
      console.warn('Error loading replay:', e);
    }
  });
}

loadRunsList();

// --- ANATOMY MODE & INTERACTIVE SYNAPSE/BRANCH LOGIC ---

let CURRENT_ANATOMY_MODE = 'synthetic';
let CURRENT_DATASET = null;
let SYNAPSE_DATA = [];

function loadDataset(url, mode) {
  fetch(url)
    .then(res => {
      if (!res.ok) throw new Error(`Failed to load ${url}`);
      return res.json();
    })
    .then(data => {
      CURRENT_ANATOMY_MODE = mode;
      CURRENT_DATASET = data;
      initAnatomy(data);
      updateBannerAndInspector(mode, data);
    })
    .catch(err => {
      console.warn(`Could not load ${url}:`, err);
    });
}

function updateBannerAndInspector(mode, data) {
  const isBio = (mode === 'malecns');
  const title = document.querySelector('#modelStatusTitle');
  const desc = document.querySelector('#modelStatusDesc');
  const statusDot = document.querySelector('#statusDot');
  const partnerBadge = document.querySelector('#partnerCountBadge');

  if (isBio) {
    if (title) title.textContent = 'MALECNS v1.0 — VERIFIED OFFLINE FIXTURE';
    if (desc) desc.textContent = `Offline fixture for Drosophila T4a Body ID ${data.target_cell_id}. ${data.total_synapses} synapses, 5 partner classes, prototype arbor in 8 nm EM coordinates. Source Mode: ${data.source_mode}.`;
    if (statusDot) {
      statusDot.style.background = 'var(--cyan)';
      statusDot.style.boxShadow = '0 0 10px var(--cyan)';
    }
    if (partnerBadge) partnerBadge.textContent = `${data.total_synapses} EM SYNAPSES`;
  } else {
    if (title) title.textContent = 'SYNTHETIC CANONICAL T4';
    if (desc) desc.textContent = 'Not derived from MaleCNS EM data. Used for hypothesis testing. 110 synthetic contacts, 4 cell types.';
    if (statusDot) {
      statusDot.style.background = 'var(--amber)';
      statusDot.style.boxShadow = '0 0 10px var(--amber)';
    }
    if (partnerBadge) partnerBadge.textContent = `${data.total_synapses || 110} CONTACTS`;
  }
}

function initAnatomy(data) {
  SYNAPSE_DATA = data.synapses || [];

  // Populate Retinotopic Grid
  const gridContainer = document.querySelector('#retinotopicGrid');
  if (gridContainer) {
    gridContainer.innerHTML = '';
    for (let c = 0; c < 16; c++) {
      const colDiv = document.createElement('div');
      colDiv.className = 'retino-col';
      colDiv.dataset.col = c;
      if (c === 6 || c === 8 || c === 10) {
        colDiv.style.borderBottom = '3px solid ' + (c === 6 ? '#d07af5' : (c === 8 ? '#58dfc2' : '#f0a65a'));
      }
      colDiv.addEventListener('click', () => selectRetinotopicColumn(c));
      gridContainer.appendChild(colDiv);
    }
  }

  // Populate Synapses on Arbor SVG
  const synGroup = document.querySelector('#synapsesGroup');
  if (synGroup) {
    synGroup.innerHTML = '';
    SYNAPSE_DATA.forEach(s => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', s.x);
      circle.setAttribute('cy', s.y);
      circle.setAttribute('r', '1.1');
      circle.setAttribute('fill', s.color);
      circle.setAttribute('stroke', '#081116');
      circle.setAttribute('stroke-width', '0.3');
      circle.setAttribute('class', 'synapse-node');
      circle.dataset.synId = s.id;
      circle.addEventListener('click', (e) => {
        e.stopPropagation();
        selectSynapse(s.id);
      });
      synGroup.appendChild(circle);
    });
  }

  // Add click listeners to dendritic branches
  document.querySelectorAll('.dendrite-branch').forEach(branch => {
    branch.addEventListener('click', (e) => {
      e.stopPropagation();
      const comp = branch.dataset.branch;
      selectBranch(comp);
    });
  });

  renderSynapseTable();

  if (SYNAPSE_DATA.length > 0) {
    selectSynapse(SYNAPSE_DATA[0].id);
  }
}

// Initial load: Synthetic canonical hypothesis dataset
loadDataset('./anatomy.json', 'synthetic');

// Populate Synapse Table
const tableBody = document.querySelector('#synapseTableBody');
function renderSynapseTable(filterType = 'all') {
  if (!tableBody) return;
  tableBody.innerHTML = '';
  const filtered = filterType === 'all' ? SYNAPSE_DATA : SYNAPSE_DATA.filter(s => s.type === filterType);
  filtered.forEach(s => {
    const tr = document.createElement('tr');
    tr.dataset.synId = s.id;
    tr.innerHTML = '<td>#' + s.id + '</td><td style="color:' + s.color + '">' + s.type + '</td><td>' + s.comp + '</td><td>' + s.x + '</td><td>' + s.nt + '</td>';
    tr.addEventListener('click', () => selectSynapse(s.id));
    tableBody.appendChild(tr);
  });
}



// Partner Filters
document.querySelectorAll('.partner-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.partner-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    const target = pill.dataset.partner;
    renderSynapseTable(target);
    document.querySelectorAll('.synapse-node').forEach(node => {
      const syn = SYNAPSE_DATA.find(s => s.id == node.dataset.synId);
      node.style.opacity = (target === 'all' || (syn && syn.type === target)) ? '1.0' : '0.15';
    });
  });
});

// Interactive Element Selection (Tri-Partite Layering)
function selectSynapse(id) {
  const syn = SYNAPSE_DATA.find(s => s.id === id);
  if (!syn) return;

  document.querySelectorAll('.synapse-node').forEach(n => n.classList.toggle('highlighted', n.dataset.synId == id));
  document.querySelectorAll('#synapseTableBody tr').forEach(r => r.classList.toggle('selected', r.dataset.synId == id));
  document.querySelectorAll('.dendrite-branch').forEach(b => b.classList.toggle('highlighted', b.dataset.branch === syn.comp));
  document.querySelectorAll('.retino-col').forEach(c => c.classList.toggle('highlighted', c.dataset.col == syn.col));

  const titleElem = document.querySelector('#inspectorTitle');
  if (titleElem) {
    const tierName = (CURRENT_ANATOMY_MODE === 'malecns') ? 'BIOLOGICAL EM' : 'SYNTHETIC';
    titleElem.innerHTML = `SELECTED ELEMENT: <strong>${syn.type} ${tierName} SYNAPSE #${syn.id}</strong>`;
  }

  const detailsElem = document.querySelector('#inspectorDetails');
  if (detailsElem) {
    const rawVoxelStr = syn.raw_voxel ? `[${syn.raw_voxel.join(', ')}] (8 nm voxels)` : `(${syn.x}, ${syn.y}, ${syn.z}) μm`;
    const distStr = syn.dist_to_soma_um ? `<b>${syn.dist_to_soma_um} μm from soma</b>` : `<b>${syn.comp}</b>`;

    detailsElem.innerHTML =
      `<span>PRE-TYPE: <b style="color:${syn.color}">${syn.type} (Pre Body #${syn.preId})</b></span>` +
      `<span>COMPARTMENT: <b>${syn.comp.replace('_', ' ').toUpperCase()}</b></span>` +
      `<span>RAW EM COORD: <b>${rawVoxelStr}</b></span>` +
      `<span>SOMA DISTANCE: ${distStr}</span>` +
      `<span>TRANSMITTER: <b class="orange-text">${syn.nt_full || syn.nt}</b></span>` +
      `<span>RECEPTIVE FIELD: <b>Column ${syn.col} (Ommatidium #${400 + syn.col * 8})</b></span>`;
  }

  const provTier = document.querySelector('#provTierBadge');
  const provSource = document.querySelector('#provSource');
  const provRationale = document.querySelector('#provRationale');

  if (CURRENT_ANATOMY_MODE === 'malecns') {
    if (provTier) {
      provTier.textContent = 'SOURCE_DERIVED_FIXTURE';
      provTier.className = 'provenance-tier-tag biological';
    }
    if (provSource) provSource.textContent = `MaleCNS_v1.0_EM_Fixture (Body ID ${CURRENT_DATASET?.target_cell_id || 5813072001})`;
    if (provRationale) provRationale.textContent = `Offline source-derived fixture from Janelia MaleCNS v1.0 specifications (prototype arbor in 8 nm coordinates). Nearest-skeleton projection: ${syn.dist_to_soma_um || 0} μm from root soma.`;
  } else {
    if (provTier) {
      provTier.textContent = 'COMPUTATIONAL_HYPOTHESIS';
      provTier.className = 'provenance-tier-tag hypothesis';
    }
    if (provSource) provSource.textContent = 'Synthetic_Canonical_T4a_Model';
    if (provRationale) provRationale.textContent = 'Parameterized hypothesis testing of ' + syn.comp + ' inputs based on Takemura et al. (2017) and Borst & Haag (2020).';
  }
}

function selectBranch(comp) {
  document.querySelectorAll('.dendrite-branch').forEach(b => b.classList.toggle('highlighted', b.dataset.branch === comp));
  const mappedSyns = SYNAPSE_DATA.filter(s => s.comp === comp || (comp === 'leading' && s.comp === 'leading_tip') || (comp === 'leading' && s.comp === 'distal_tip'));
  document.querySelectorAll('.synapse-node').forEach(n => {
    const isMapped = mappedSyns.some(s => s.id == n.dataset.synId);
    n.classList.toggle('highlighted', isMapped);
  });
  if (mappedSyns.length > 0) {
    selectSynapse(mappedSyns[0].id);
  }
}

function selectRetinotopicColumn(colIdx) {
  document.querySelectorAll('.retino-col').forEach(c => c.classList.toggle('highlighted', c.dataset.col == colIdx));
  const mappedSyns = SYNAPSE_DATA.filter(s => s.col === colIdx);
  document.querySelectorAll('.synapse-node').forEach(n => {
    const isMapped = mappedSyns.some(s => s.id == n.dataset.synId);
    n.classList.toggle('highlighted', isMapped);
  });
  if (mappedSyns.length > 0) {
    selectSynapse(mappedSyns[0].id);
  }
}

// Model Evaluator Switcher
const MODEL_DATA = {
  model_a: { dsi: '0.2919', mi: '0.4545', v: '0.1865', status: 'FAIL', curve: '[57, 39, 75, 89, 104, 89, 74, 53]' },
  model_b: { dsi: '0.1099', mi: '0.5303', v: '0.2292', status: 'FAIL (TONIC)', curve: '[45, 31, 81, 91, 100, 91, 101, 44]' },
  model_c: { dsi: '0.3620', mi: '0.5102', v: '0.2321', status: 'PASSING Δ', curve: '[52, 36, 81, 104, 111, 104, 101, 50]' },
  model_d: { dsi: '0.5902', mi: '1.0000', v: '0.4513', status: 'PASS (HYPOTHESIS GATE)', curve: '[25, 19, 0, 84, 97, 84, 0, 33]' },
};

document.querySelectorAll('.model-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.model-tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const m = MODEL_DATA[btn.dataset.model];
    if (m) {
      document.querySelector('#activeDsi').textContent = m.dsi;
      document.querySelector('#activeDsi').nextElementSibling.textContent = m.status;
      document.querySelector('#activeMi').textContent = m.mi;
      document.querySelector('#activeVector').textContent = m.v;
      document.querySelector('#activeCurve').textContent = m.curve;
    }
  });
});

// Model Status Banner Switching Logic (Synthetic 1C-Hyp vs MaleCNS 1C-A)
const btnSyn = document.querySelector('#statusSyntheticBtn');
const btnReal = document.querySelector('#statusRealBtn');

if (btnSyn) {
  btnSyn.addEventListener('click', () => {
    btnSyn.classList.add('active');
    if (btnReal) btnReal.classList.remove('active');
    loadDataset('./anatomy.json', 'synthetic');
  });
}

if (btnReal) {
  btnReal.addEventListener('click', () => {
    btnReal.classList.add('active');
    if (btnSyn) btnSyn.classList.remove('active');
    loadDataset('./malecns_anatomy.json', 'malecns');
  });
}
