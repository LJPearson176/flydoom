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

$$('.mode-tab').forEach((button) => button.addEventListener('click', () => {
  $$('.mode-tab').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
}));

$('#cameraButton').addEventListener('click', () => {
  flyEye = !flyEye;
  $('#cameraButton').innerHTML = flyEye ? 'SWITCH TO OBSERVER CAMERA <span>↗</span>' : 'SWITCH TO FLY-EYE <span>↗</span>';
  $('.doom-view').classList.toggle('fly-eye-active', flyEye);
  $('.doom-label').textContent = flyEye ? 'FLY-EYE / MOTION ENERGY' : 'DOOM / E1M1';
});

$$('.seg').forEach((button) => button.addEventListener('click', () => {
  $$('.seg').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
}));

$('#traceButton').addEventListener('click', () => {
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

// --- ANATOMY MODE & INTERACTIVE SYNAPSE/BRANCH LOGIC ---

const partners = [
  { type: 'Mi4', count: 20, comp: 'leading', nt: 'GABA', color: '#d07af5', baseId: 2101, mx: -4.2, my: 0.0, mz: 1.5, tau: '40.0 ms', col: 6 },
  { type: 'Mi9', count: 15, comp: 'leading_tip', nt: 'Glu', color: '#f5d76e', baseId: 2201, mx: -7.2, my: -4.5, mz: 1.2, tau: '40.0 ms', col: 6 },
  { type: 'Mi1', count: 40, comp: 'central', nt: 'ACh', color: '#58dfc2', baseId: 3001, mx: 0.1, my: -5.5, mz: 0.0, tau: '15.0 ms', col: 8 },
  { type: 'Tm3', count: 35, comp: 'trailing', nt: 'ACh', color: '#f0a65a', baseId: 4001, mx: 5.2, my: -3.8, mz: -0.8, tau: '60.0 ms (delayed 20ms)', col: 10 },
];

const SYNAPSE_DATA = [];
let synId = 1;
partners.forEach(p => {
  for (let i = 0; i < p.count; i++) {
    const jitterX = (Math.random() - 0.5) * 1.8;
    const jitterY = (Math.random() - 0.5) * 1.8;
    const jitterZ = (Math.random() - 0.5) * 0.8;
    SYNAPSE_DATA.push({
      id: synId++,
      preId: p.baseId + (i % 3),
      type: p.type,
      comp: p.comp,
      nt: p.nt,
      color: p.color,
      x: +(p.mx + jitterX).toFixed(2),
      y: +(p.my + jitterY).toFixed(2),
      z: +(p.mz + jitterZ).toFixed(2),
      tau: p.tau,
      col: p.col
    });
  }
});

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
    circle.addEventListener('click', () => selectSynapse(s.id));
    synGroup.appendChild(circle);
  });
}

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
renderSynapseTable();

// Mode Switcher (Observe vs Anatomy)
document.querySelectorAll('.mode-tab').forEach((tab) => {
  tab.addEventListener('click', (e) => {
    const isAnatomy = e.target.textContent.trim() === 'Anatomy';
    document.body.classList.toggle('anatomy-active', isAnatomy);
  });
});

// Partner Filters
document.querySelectorAll('.partner-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.partner-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    const target = pill.dataset.partner;
    renderSynapseTable(target);
    document.querySelectorAll('.synapse-node').forEach(node => {
      const syn = SYNAPSE_DATA.find(s => s.id == node.dataset.synId);
      node.style.opacity = (target === 'all' || syn.type === target) ? '1.0' : '0.15';
    });
  });
});

// Interactive Element Selection
function selectSynapse(id) {
  const syn = SYNAPSE_DATA.find(s => s.id === id);
  if (!syn) return;

  document.querySelectorAll('.synapse-node').forEach(n => n.classList.toggle('highlighted', n.dataset.synId == id));
  document.querySelectorAll('#synapseTableBody tr').forEach(r => r.classList.toggle('selected', r.dataset.synId == id));
  document.querySelectorAll('.dendrite-branch').forEach(b => b.classList.toggle('highlighted', b.dataset.branch === syn.comp));
  document.querySelectorAll('.retino-col').forEach(c => c.classList.toggle('highlighted', c.dataset.col == syn.col));

  const titleElem = document.querySelector('#inspectorTitle');
  if (titleElem) {
    titleElem.innerHTML = 'SELECTED ELEMENT: <strong>' + syn.type + ' SYNAPSE #' + syn.id + '</strong>';
  }
  const detailsElem = document.querySelector('#inspectorDetails');
  if (detailsElem) {
    const ntDesc = syn.nt === 'ACh' ? 'Acetylcholine (Excitatory)' : (syn.nt === 'GABA' ? 'GABA (Shunting Inhibitory)' : 'Glutamate (Inhibitory GluCl)');
    detailsElem.innerHTML = '<span>PRE-TYPE: <b style="color:' + syn.color + '">' + syn.type + ' (#' + syn.preId + ')</b></span>' +
      '<span>COMPARTMENT: <b>' + syn.comp.replace('_', ' ').toUpperCase() + '</b></span>' +
      '<span>COORDINATES: <b>(' + syn.x + ', ' + syn.y + ', ' + syn.z + ') μm</b></span>' +
      '<span>TRANSMITTER: <b class="orange-text">' + ntDesc + '</b></span>' +
      '<span>DELAY / TAU: <b>' + syn.tau + '</b></span>' +
      '<span>RECEPTIVE FIELD: <b>Visual Column ' + syn.col + ' (Ommatidium #' + (400 + syn.col * 8) + ')</b></span>';
  }

  const provTier = document.querySelector('#provTierBadge');
  if (provTier) {
    provTier.textContent = 'COMPUTATIONAL_HYPOTHESIS';
    provTier.className = 'provenance-tier-tag hypothesis';
  }
  const provSource = document.querySelector('#provSource');
  if (provSource) provSource.textContent = 'Synthetic_Canonical_T4a_Model';
  const provRationale = document.querySelector('#provRationale');
  if (provRationale) provRationale.textContent = 'Parameterized hypothesis testing of ' + syn.comp + ' inputs based on Takemura et al. (2017) and Borst & Haag (2020).';
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
  model_d: { dsi: '0.5902', mi: '1.0000', v: '0.4513', status: 'PASS (BIOLOGICAL)', curve: '[25, 19, 0, 84, 97, 84, 0, 33]' },
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

// Model Status Banner Buttons
const btnSyn = document.querySelector('#statusSyntheticBtn');
const btnReal = document.querySelector('#statusRealBtn');
if (btnSyn) {
  btnSyn.addEventListener('click', () => {
    btnSyn.classList.add('active');
    if (btnReal) btnReal.classList.remove('active');
    const label = document.querySelector('#modelStatusBanner .model-status-badge strong');
    if (label) label.textContent = 'SYNTHETIC CANONICAL T4';
    const desc = document.querySelector('#modelStatusBanner .model-status-desc');
    if (desc) desc.textContent = 'Not derived from MaleCNS EM data. Used for hypothesis testing. 110 synthetic contacts, 4 cell types.';
  });
}
if (btnReal) {
  btnReal.addEventListener('click', () => {
    btnReal.classList.add('active');
    if (btnSyn) btnSyn.classList.remove('active');
    const label = document.querySelector('#modelStatusBanner .model-status-badge strong');
    if (label) label.textContent = 'MALECNS T4a RECONSTRUCTION (TARGET)';
    const desc = document.querySelector('#modelStatusBanner .model-status-desc');
    if (desc) desc.textContent = 'Awaiting Phase 1C-A EM pipeline extraction: Body ID query, raw 3D synapse coordinates, morphological SWC arbor.';
  });
}
