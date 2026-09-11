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

