"""Generate the complete interactive Full 3D Visible Drosophila Nervous System application.

Integrates:
- Authentic JFRC2 Whole-Brain Template Surface Mesh (6,654 biological wireframe edges)
- Authentic Individual Neuropil Domain Meshes (Medulla, Lobula Plate, Central Complex EB/FB/PB, SEZ, AL)
- 3D Translucent Drosophila Exoskeleton (head, eyes, antennae, proboscis, thorax, wings, 6 legs, abdomen)
- Dense FlyWire Whole-Brain Connectome (3,030 neuron fibers, esophageal foramen, LPTC fans, CX loops)
- Volumetric Synaptic Nodes & Traveling Action Potentials
- Real-Time Activity Illumination Engine during Doom Combat Replay
"""

from __future__ import annotations

import json
from pathlib import Path


def main():
    cns_data_path = Path("web/fly_cns_data.json")
    model_3d_path = Path("web/fly_3d_cns_model.json")
    mesh_path = Path("web/authentic_fly_cns_mesh.json")

    with open(cns_data_path) as f:
        cns_data = json.load(f)

    with open(model_3d_path) as f:
        model_3d = json.load(f)

    with open(mesh_path) as f:
        mesh_data = json.load(f)

    payload = {
        "anatomy": cns_data["anatomy"],
        "episodes": cns_data["episodes"],
        "exoskeleton": model_3d["exoskeleton"],
        "connectome": model_3d["connectome"],
        "brain_mesh": mesh_data["brain_surface_edges"],
        "neuropil_meshes": mesh_data["neuropils"],
        "high_fidelity_pathways": model_3d.get("high_fidelity_pathways"),
    }
    payload_json = json.dumps(payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Full 3D Visible Drosophila Nervous System & Connectome Twin</title>
  <style>
    :root {{
      --bg: #050a0e;
      --panel-bg: rgba(9, 20, 27, 0.92);
      --border: #16323e;
      --border-glow: #245062;
      --cyan: #00e5ff;
      --cyan-dim: rgba(0, 229, 255, 0.15);
      --amber: #ff9800;
      --green: #00ffaa;
      --purple: #c084fc;
      --red: #ff3366;
      --text: #e2f1f8;
      --text-muted: #7899a9;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-mono);
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    header {{
      height: 52px;
      background: #081319;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 30;
    }}
    .title-group {{ display: flex; align-items: center; gap: 14px; }}
    .badge {{
      background: var(--cyan-dim);
      border: 1px solid var(--cyan);
      color: var(--cyan);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
    }}
    .title-text {{ font-size: 13px; font-weight: 700; letter-spacing: 0.06em; }}
    .subtitle-text {{ font-size: 10px; color: var(--text-muted); }}

    .header-chips {{ display: flex; gap: 20px; font-size: 11px; }}
    .chip {{ display: flex; flex-direction: column; }}
    .chip span {{ font-size: 9px; color: var(--text-muted); }}
    .chip strong {{ font-size: 12px; color: var(--cyan); }}

    #container {{
      flex: 1;
      position: relative;
      display: flex;
      overflow: hidden;
    }}

    #viewport {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at 50% 45%, #0a1c26 0%, #04080b 85%);
    }}
    canvas#renderCanvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    canvas#renderCanvas:active {{ cursor: grabbing; }}

    .grid-overlay {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(var(--border) 1px, transparent 1px),
                        linear-gradient(90deg, var(--border) 1px, transparent 1px);
      background-size: 44px 44px;
      opacity: 0.12;
    }}

    .cam-toolbar {{
      position: absolute;
      top: 14px;
      left: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      z-index: 20;
    }}
    .btn {{
      background: var(--panel-bg);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      backdrop-filter: blur(8px);
    }}
    .btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
    .btn.active {{
      background: rgba(0, 229, 255, 0.18);
      border-color: var(--cyan);
      color: var(--cyan);
      box-shadow: 0 0 10px rgba(0, 229, 255, 0.25);
    }}

    .cam-coords {{
      position: absolute;
      bottom: 14px;
      left: 18px;
      font-size: 10px;
      color: var(--text-muted);
      pointer-events: none;
      line-height: 1.6;
    }}

    aside#sidebar {{
      width: 390px;
      background: var(--panel-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      backdrop-filter: blur(16px);
      z-index: 25;
      overflow-y: auto;
    }}

    .section {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }}
    .sec-title {{
      font-size: 11px;
      font-weight: 700;
      color: var(--cyan);
      letter-spacing: 0.1em;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .layer-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 11px;
    }}
    .slider-row {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 10px;
      font-size: 11px;
    }}
    .slider-row label {{ display: flex; justify-content: space-between; color: var(--text-muted); font-size: 10px; }}
    input[type=range] {{
      width: 100%;
      height: 5px;
      -webkit-appearance: none;
      background: #0d1e27;
      border-radius: 3px;
      outline: none;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 8px var(--cyan);
      cursor: pointer;
    }}

    .select {{
      width: 100%;
      background: #091720;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 7px 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      border-radius: 4px;
      outline: none;
      margin-bottom: 10px;
    }}

    .telem-grid {{
      background: #071217;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 12px;
      margin-top: 10px;
      font-size: 11px;
    }}
    .telem-box {{ display: flex; flex-direction: column; }}
    .telem-box span {{ font-size: 9px; color: var(--text-muted); }}
    .telem-box strong {{ font-size: 12px; color: var(--text); }}
    .telem-box strong.action {{ color: var(--amber); }}
    .telem-box strong.lock {{ color: var(--green); }}

    .btn-row {{ display: flex; gap: 6px; margin-top: 8px; }}

    .stim-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
    }}
    .stim-btn {{
      background: #091720;
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 8px 6px;
      border-radius: 4px;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }}
    .stim-btn:hover {{ border-color: var(--amber); color: var(--amber); background: rgba(255,152,0,0.1); }}

    .insp-box {{
      background: #071217;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      font-size: 11px;
    }}
    .insp-line {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
    .insp-line span {{ color: var(--text-muted); }}
    .insp-line b {{ color: var(--cyan); }}

    .lesion-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
      margin-bottom: 8px;
    }}
    .lesion-btn {{
      background: #091720;
      border: 1px solid var(--border);
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 6px 8px;
      border-radius: 4px;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }}
    .lesion-btn:hover {{ border-color: var(--cyan); color: var(--text); }}
    .lesion-btn.active {{
      border-color: #ff1744;
      color: #fff;
      background: rgba(255, 23, 68, 0.2);
    }}
    .lesion-btn.active-intact {{
      border-color: #00ffaa;
      color: #fff;
      background: rgba(0, 255, 170, 0.2);
    }}

    .vm-legend {{
      position: absolute;
      bottom: 45px;
      left: 20px;
      background: rgba(4, 10, 14, 0.92);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 9px;
      display: flex;
      flex-direction: column;
      gap: 5px;
      z-index: 20;
      pointer-events: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    }}
    .vm-bar {{
      width: 190px;
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(to right, #7c3aed, #06b6d4, #f59e0b, #ef4444);
    }}
    .vm-labels {{
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 8px;
    }}

    #tooltip {{
      position: absolute;
      background: rgba(4, 10, 14, 0.95);
      border: 1px solid var(--cyan);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 10px;
      pointer-events: none;
      display: none;
      z-index: 100;
      box-shadow: 0 4px 16px rgba(0,0,0,0.7);
    }}

    /* Connectome Wiring Matrix Modal */
    .wiring-modal {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: min(840px, 94vw);
      max-height: 88vh;
      background: rgba(6, 14, 20, 0.96);
      border: 1px solid var(--cyan);
      border-radius: 8px;
      box-shadow: 0 0 50px rgba(0, 229, 255, 0.3), 0 20px 40px rgba(0, 0, 0, 0.85);
      z-index: 1000;
      display: none;
      flex-direction: column;
      backdrop-filter: blur(16px);
      overflow: hidden;
    }}
    .wiring-modal-header {{
      padding: 12px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(0, 229, 255, 0.06);
    }}
    .wiring-modal-body {{
      padding: 16px 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }}
    .matrix-table th, .matrix-table td {{
      border: 1px solid rgba(26, 54, 71, 0.7);
      padding: 7px 6px;
      text-align: center;
    }}
    .matrix-table th {{
      background: rgba(10, 28, 38, 0.85);
      color: var(--cyan);
      font-weight: 700;
      font-size: 10px;
      letter-spacing: 0.05em;
    }}
    .matrix-cell {{
      color: var(--text-muted);
      transition: all 0.15s ease;
      user-select: none;
    }}
    .matrix-cell.has-synapse {{
      cursor: pointer;
      font-weight: 700;
    }}
    .matrix-cell.has-synapse:hover {{
      background: rgba(0, 229, 255, 0.3) !important;
      transform: scale(1.04);
      box-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }}
    .cell-ach {{ background: rgba(0, 229, 255, 0.12); color: #00e5ff; }}
    .cell-gaba {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; }}
    .cell-glu {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; }}
    .cell-motor {{ background: rgba(239, 68, 68, 0.15); color: #f87171; }}
    .cell-lesioned {{
      background: rgba(239, 68, 68, 0.28) !important;
      color: #ef4444 !important;
      text-decoration: line-through;
      border: 1px dashed #ef4444 !important;
    }}
    .signal-flow-container {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 4px;
      overflow-x: auto;
      padding: 6px 0;
    }}
    .signal-step {{
      flex: 1;
      background: rgba(10, 28, 38, 0.7);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 85px;
      font-size: 10px;
    }}
    .signal-step.active-flow {{
      border-color: var(--cyan);
      box-shadow: 0 0 8px rgba(0, 229, 255, 0.25);
    }}
    .step-num {{ font-size: 8px; color: var(--cyan); font-weight: 700; }}
    .step-delay {{ font-size: 8px; color: var(--green); }}
    .step-loc {{ font-size: 8px; color: var(--text-muted); }}
    .signal-arrow {{ color: var(--text-muted); font-size: 11px; }}
  </style>
</head>
<body>

  <header>
    <div class="title-group">
      <span class="badge">FULL 3D VISIBLE CNS TWIN</span>
      <div>
        <div class="title-text">DROSOPHILA MELANOGASTER CONNECTOME & EXOSKELETON</div>
        <div class="subtitle-text">Official JFRC2 Template Brain Mesh · FlyWire Whole-Brain Connectome (Nature 2024)</div>
      </div>
    </div>
    <div class="header-chips">
      <div class="chip"><span>JFRC2 BRAIN MESH</span><strong>6,654 EDGES</strong></div>
      <div class="chip"><span>CONNECTOME</span><strong>3,030 FIBERS</strong></div>
      <div class="chip"><span>SWC SKELETONS</span><strong>14 NEURONS</strong></div>
      <div class="chip"><span>EM SYNAPSES</span><strong>262 VERIFIED</strong></div>
      <div class="chip"><span>CARTRIDGES</span><strong>128 COLS</strong></div>
    </div>
  </header>

  <div id="container">
    <div id="viewport">
      <div class="grid-overlay"></div>
      <canvas id="renderCanvas"></canvas>

      <div class="cam-toolbar">
        <button class="btn" id="btnPresetFlyWire" onclick="setPreset('flywire')">FLYWIRE FRONTAL</button>
        <button class="btn" id="btnPresetFullFly" onclick="setPreset('full_fly')">FULL FLY 3D (LATERAL)</button>
        <button class="btn active" id="btnPresetIso" onclick="setPreset('iso')">ISOMETRIC 3D</button>
        <button class="btn" id="btnPresetTop" onclick="setPreset('top')">SUPERIOR (TOP)</button>
        <button class="btn" id="btnAutoRotate" onclick="toggleAutoRotate()">AUTO ORBIT: OFF</button>
        <button class="btn" id="btnCuticleToggle" onclick="toggleCuticlePreset()">TOGGLE CUTICLE</button>
        <div style="width:1px; height:18px; background:var(--border); margin:0 4px;"></div>
        <button class="btn active" id="btnGranL1" onclick="setGranularity(1)" title="Level 1: Macro CNS Surface & 3,030 Connectome Fibers">L1: MACRO</button>
        <button class="btn" id="btnGranL2" onclick="setGranularity(2)" title="Level 2: 128 Retinotopic Cartridge Columns">L2: CIRCUITS</button>
        <button class="btn" id="btnGranL3" onclick="setGranularity(3)" title="Level 3: Synapse Point Cloud & SWC Neuron Morphologies">L3: SYNAPSES</button>
        <button class="btn" id="btnWiringMatrix" onclick="toggleMatrixModal()" title="View Directed Synaptic Adjacency Matrix & Circuit Wiring">⚡ WIRING MATRIX</button>
      </div>

      <div class="cam-coords">
        <div>MOUSE: Left-drag Orbit · Right-drag Pan · Wheel Zoom</div>
        <div id="readoutCam">CAM: Yaw 32.0° · Pitch 22.0° · Dist 850μm</div>
      </div>

      <div class="vm-legend" id="vmLegend" style="display:none;">
        <div style="color:var(--cyan); font-weight:700; display:flex; justify-content:space-between;">
          <span>MEMBRANE POTENTIAL (Vm)</span>
          <span id="lblVmDynamic" style="color:#06b6d4;">-65 mV (REST)</span>
        </div>
        <div class="vm-bar"></div>
        <div class="vm-labels">
          <span>-75mV Shunt</span>
          <span>-65mV Rest</span>
          <span>-45mV Depol</span>
          <span>-35mV Spike</span>
        </div>
      </div>

      <!-- Connectome Micro-Circuit Wiring Matrix Modal -->
      <div id="wiringMatrixModal" class="wiring-modal">
        <div class="wiring-modal-header">
          <div style="display:flex; align-items:center; gap:10px;">
            <span class="badge" style="background:rgba(0,229,255,0.15); border-color:var(--cyan);">JANELIA MaleCNS v1.0 / FlyWire</span>
            <strong style="font-size:12px; letter-spacing:0.06em; color:var(--text);">SYNAPTIC ADJACENCY MATRIX & CIRCUIT WIRING</strong>
          </div>
          <button class="btn" style="padding:4px 8px; font-size:12px;" onclick="toggleMatrixModal()">✕ CLOSE</button>
        </div>
        <div class="wiring-modal-body">
          <div style="font-size:10px; color:var(--text-muted); line-height:1.5;">
            Directed chemical synapse counts between reconstructed electron microscopy neuron bodies in JFRC2 space.
            Click any active synapse cell to highlight the contact zone and focus the 3D visualizer.
          </div>

          <!-- Adjacency Matrix Table -->
          <div style="overflow-x:auto;">
            <table class="matrix-table" id="synapticAdjacencyTable">
              <thead>
                <tr>
                  <th style="text-align:left; padding-left:10px;">PRE / POST</th>
                  <th>T4a<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Motion)</span></th>
                  <th>Mi1<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Fast)</span></th>
                  <th>Tm3<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Delay)</span></th>
                  <th>Mi4<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Shunt)</span></th>
                  <th>Mi9<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Offset)</span></th>
                  <th>E-PG<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Compass)</span></th>
                  <th>DNpe017<br><span style="font-size:8px; font-weight:normal; opacity:0.7;">(Motor)</span></th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th style="text-align:left; padding-left:10px;">Mi1 (Fast Exc)</th>
                  <td class="matrix-cell cell-ach has-synapse" id="cell_Mi1_T4a" onclick="onMatrixCellClick('Mi1', 'T4a', 'ACh')" title="Click to view 48 ACh Synapses">48 ACh</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">Tm3 (Delay Exc)</th>
                  <td class="matrix-cell cell-ach has-synapse" id="cell_Tm3_T4a" onclick="onMatrixCellClick('Tm3', 'T4a', 'ACh')" title="Click to view 36 ACh Synapses">36 ACh</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">Mi4 (GABA Shunt)</th>
                  <td class="matrix-cell cell-gaba has-synapse" id="cell_Mi4_T4a" onclick="onMatrixCellClick('Mi4', 'T4a', 'GABA')" title="Click to view 24 GABA Synapses">24 GABA</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">Mi9 (Glu Offset)</th>
                  <td class="matrix-cell cell-glu has-synapse" id="cell_Mi9_T4a" onclick="onMatrixCellClick('Mi9', 'T4a', 'Glu')" title="Click to view 18 Glu Synapses">18 Glu</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">T4a (Directional)</th>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell cell-ach has-synapse" id="cell_T4a_EPG" onclick="onMatrixCellClick('T4a', 'E-PG', 'ACh')" title="12 Optomotor Fibers into CX Ring">12 ACh</td>
                  <td class="matrix-cell">·</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">E-PG (CX Compass)</th>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell cell-ach has-synapse" id="cell_EPG_PB" onclick="onMatrixCellClick('E-PG', 'PB', 'ACh')" title="20 Glomerular Synapses">20 ACh</td>
                  <td class="matrix-cell cell-motor has-synapse" id="cell_EPG_DN" onclick="onMatrixCellClick('E-PG', 'DNpe017', 'Motor')" title="Descending Motor Steering">8 ACh</td>
                </tr>
                <tr>
                  <th style="text-align:left; padding-left:10px;">DNpe017 (Descending)</th>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell">·</td>
                  <td class="matrix-cell cell-motor has-synapse" id="cell_DN_VNC" onclick="onMatrixCellClick('DNpe017', 'VNC', 'Motor')" title="20 Neuromuscular Trigger Synapses">20 NMJ</td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Signal Propagation Pipeline & Conduction Latency -->
          <div style="background:rgba(0,0,0,0.45); border:1px solid var(--border); border-radius:6px; padding:12px;">
            <div style="font-size:10px; font-weight:700; color:var(--cyan); margin-bottom:8px; display:flex; justify-content:space-between;">
              <span>MULTI-SCALE SIGNAL PROPAGATION & CONDUCTION LATENCY</span>
              <span style="color:var(--green);">TOTAL SENSORIMOTOR LATENCY: ~28.7 ms</span>
            </div>
            <div class="signal-flow-container">
              <div class="signal-step">
                <span class="step-num">01</span>
                <strong>PHOTONS</strong>
                <span class="step-delay">0.0 ms</span>
                <span class="step-loc">Retina (R1-R6)</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step">
                <span class="step-num">02</span>
                <strong>LAMINA</strong>
                <span class="step-delay">+8.5 ms</span>
                <span class="step-loc">L1 / L2 Monopolar</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step">
                <span class="step-num">03</span>
                <strong>MEDULLA</strong>
                <span class="step-delay">+6.2 ms</span>
                <span class="step-loc">Mi1 / Tm3 / Mi4 / Mi9</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step active-flow">
                <span class="step-num">04</span>
                <strong>T4a ARBOR</strong>
                <span class="step-delay">+2.8 ms</span>
                <span class="step-loc">Vm Multiplicative</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step">
                <span class="step-num">05</span>
                <strong>CX RING</strong>
                <span class="step-delay">+4.2 ms</span>
                <span class="step-loc">E-PG Heading Vector</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step">
                <span class="step-num">06</span>
                <strong>DESCENDING</strong>
                <span class="step-delay">+2.1 ms</span>
                <span class="step-loc">DNpe017 Axon</span>
              </div>
              <div class="signal-arrow">➔</div>
              <div class="signal-step">
                <span class="step-num">07</span>
                <strong>MOTOR</strong>
                <span class="step-delay">+4.9 ms</span>
                <span class="step-loc">VNC T3 Leg / Fire</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div id="tooltip"></div>
    </div>

    <aside id="sidebar">
      <div class="section">
        <div class="sec-title">
          <span>CONNECTOME GRANULARITY</span>
          <span style="font-size:9px; color:var(--cyan);" id="lblGranularity">L1: MACRO CNS</span>
        </div>
        <div class="btn-row" style="margin-bottom:8px;">
          <button class="btn active" id="btnSideGranL1" style="flex:1; font-size:10px; padding:6px 2px;" onclick="setGranularity(1)">L1: MACRO</button>
          <button class="btn" id="btnSideGranL2" style="flex:1; font-size:10px; padding:6px 2px;" onclick="setGranularity(2)">L2: CIRCUITS</button>
          <button class="btn" id="btnSideGranL3" style="flex:1; font-size:10px; padding:6px 2px;" onclick="setGranularity(3)">L3: SYNAPSES</button>
        </div>
        <div class="layer-row">
          <span>128 Retinotopic Cartridge Columns</span>
          <input type="checkbox" id="chkCartridges" onchange="state.showCartridges = this.checked">
        </div>
        <div class="layer-row">
          <span>SWC Morphology Skeletons (14 Neurons)</span>
          <input type="checkbox" id="chkSWC" checked onchange="state.showSWC = this.checked">
        </div>
        <div class="layer-row">
          <span>Chemical Synapses Point Cloud (262)</span>
          <input type="checkbox" id="chkSynapses" checked onchange="state.showSynapses = this.checked">
        </div>
        <div class="slider-row" style="margin-top:6px; margin-bottom:0;">
          <label><span>SYNAPSE TRANSMITTER FILTER</span></label>
          <select class="select" id="synapseFilterSel" style="margin-bottom:0;" onchange="onSynapseFilter(this.value)">
            <option value="all">ALL TRANSMITTERS (262 Synapses)</option>
            <option value="ACh">ACh / Acetylcholine (Excitatory · Cyan)</option>
            <option value="GABA">GABA (Inhibitory Shunt · Violet)</option>
            <option value="Glu">Glutamate (Distal Inhibit · Pink)</option>
            <option value="Dopamine">Dopamine (PPL1 Modulatory · Rose)</option>
          </select>
        </div>
      </div>

      <div class="section">
        <div class="sec-title">
          <span>3D VISIBILITY LAYERS</span>
          <span style="font-size:9px; color:var(--cyan);">BIOLOGICAL VOLUMES</span>
        </div>
        <div class="layer-row">
          <span>Official JFRC2 Brain Surface Mesh</span>
          <input type="checkbox" id="chkJFRC2" checked onchange="state.showJFRC2 = this.checked">
        </div>
        <div class="layer-row">
          <span>Authentic Neuropil Meshes (EB/FB/PB/ME)</span>
          <input type="checkbox" id="chkNeuropils" checked onchange="state.showNeuropils = this.checked">
        </div>
        <div class="layer-row">
          <span>FlyWire Connectome Fibers</span>
          <input type="checkbox" id="chkFibers" checked onchange="state.showFibers = this.checked">
        </div>
        <div class="slider-row">
          <label><span>EXOSKELETON CUTICLE OPACITY</span><b id="lblCuticleOp">35%</b></label>
          <input type="range" id="cuticleOpacitySlider" min="0" max="100" value="35" oninput="onCuticleOpacity(this.value)">
        </div>
        <div class="slider-row">
          <label><span>CONNECTOME FIBER DENSITY</span><b id="lblFiberOp">100%</b></label>
          <input type="range" id="fiberDensitySlider" min="10" max="100" value="100" oninput="onFiberDensity(this.value)">
        </div>
        <div class="layer-row">
          <span>Show Esophageal Foramen Gap</span>
          <input type="checkbox" id="chkForamen" checked onchange="state.showForamen = this.checked">
        </div>
        <div class="layer-row">
          <span>Show Action Potential Pulses</span>
          <input type="checkbox" id="chkParticles" checked onchange="state.showParticles = this.checked">
        </div>
        <div class="layer-row">
          <span>Show Neuropil Synaptic Nuclei</span>
          <input type="checkbox" id="chkPoints" checked onchange="state.showPoints = this.checked">
        </div>
      </div>

      <div class="section">
        <div class="sec-title">
          <span>SYNAPTIC LESION SIMULATOR</span>
          <span style="font-size:9px; color:var(--green);" id="lblLesionStatus">INTACT CONTROL</span>
        </div>
        <div class="lesion-grid">
          <button class="lesion-btn active-intact" id="btnLesionIntact" onclick="setLesionMode('intact')">● INTACT (MODEL D)</button>
          <button class="lesion-btn" id="btnLesionMi4" onclick="setLesionMode('Mi4')">✕ Mi4 GABA KO</button>
          <button class="lesion-btn" id="btnLesionMi1" onclick="setLesionMode('Mi1')">✕ Mi1 ACh KO</button>
          <button class="lesion-btn" id="btnLesionTm3" onclick="setLesionMode('Tm3')">✕ Tm3 ACh KO</button>
          <button class="lesion-btn" id="btnLesionMi9" onclick="setLesionMode('Mi9')" style="grid-column: span 2;">✕ Mi9 GLUTAMATE KO</button>
        </div>
        <div class="telem-grid" id="lesionBenchmarkGrid" style="margin-top:4px;">
          <div class="telem-box"><span>TARGET ERROR (Δθ)</span><strong id="lblLesionError" style="color:var(--green);">30.9° (Locked)</strong></div>
          <div class="telem-box"><span>LOCK FRACTION</span><strong id="lblLesionLock" style="color:var(--green);">71.4% (Stable)</strong></div>
          <div class="telem-box"><span>ASYM VARIANCE σ(Δ̂)</span><strong id="lblLesionVariance" style="color:var(--green);">0.7216 (Norm)</strong></div>
          <div class="telem-box"><span>SURVIVAL / HP</span><strong id="lblLesionHealth" style="color:var(--green);">97 HP (2 Kills)</strong></div>
        </div>
      </div>

      <div class="section">
        <div class="sec-title">
          <span>LIVE COMBAT ACTIVITY REPLAY</span>
          <span style="font-size:9px; color:var(--amber);" id="telemCondBadge">INTACT MODEL D</span>
        </div>
        <select class="select" id="episodeSelector" onchange="onEpisodeChange()"></select>

        <div class="slider-row">
          <label><span>STEP <b id="lblStep" style="color:var(--cyan);">000 / 150</b></span><span id="lblTime">0.00s</span></label>
          <input type="range" id="stepSlider" min="0" max="150" value="0" oninput="onScrub(this.value)">
        </div>
        <div class="btn-row">
          <button class="btn" style="flex:1;" onclick="stepDelta(-1)">◀ STEP</button>
          <button class="btn" style="flex:1;" id="btnPlay" onclick="togglePlay()">▶ PLAY</button>
          <button class="btn" style="flex:1;" onclick="stepDelta(1)">STEP ▶</button>
        </div>

        <div class="telem-grid">
          <div class="telem-box"><span>ACTION</span><strong class="action" id="telemAction">FORWARD</strong></div>
          <div class="telem-box"><span>PLAYER HEALTH</span><strong id="telemHealth">100 HP</strong></div>
          <div class="telem-box"><span>KILLS (PK/FF)</span><strong id="telemKills">0 / 0</strong></div>
          <div class="telem-box"><span>DAMAGE DEALT</span><strong id="telemDmg">0.0</strong></div>
          <div class="telem-box"><span>ASYMMETRY (Δ̂)</span><strong id="telemAsym">0.000</strong></div>
          <div class="telem-box"><span>FIRING LOCK</span><strong class="lock" id="telemLock">NO TARGET</strong></div>
        </div>
      </div>

      <div class="section">
        <div class="sec-title">
          <span>ACTIVITY EXCITATION STIMULATOR</span>
          <span style="font-size:9px; color:var(--text-muted);">DIRECT ACTIVATION</span>
        </div>
        <div class="stim-grid">
          <button class="stim-btn" onclick="triggerStim('turn_right')">▶ OPTIC FLOW RIGHT</button>
          <button class="stim-btn" onclick="triggerStim('turn_left')">◀ OPTIC FLOW LEFT</button>
          <button class="stim-btn" onclick="triggerStim('forward')">▲ FORWARD THRUST</button>
          <button class="stim-btn" onclick="triggerStim('door')">⎔ DOOR TOUCH (USE)</button>
          <button class="stim-btn" onclick="triggerStim('fire')">★ COMBAT FIRE BURST</button>
          <button class="stim-btn" onclick="triggerStim('panic')">⚠ THREAT PANIC SHOCK</button>
        </div>
      </div>

      <div class="section">
        <div class="sec-title">
          <span>NEUROPIL INSPECTOR</span>
          <span style="font-size:9px; color:var(--cyan);">BIOLOGICAL ANATOMY</span>
        </div>
        <div class="insp-box">
          <div class="insp-line"><span>COMPARTMENT</span><b id="inspName">Lobula Plate (LP_R)</b></div>
          <div class="insp-line"><span>SYSTEM</span><b id="inspSystem">Optic Lobe (Motion)</b></div>
          <div class="insp-line"><span>CELL TYPES</span><b id="inspCells">T4a-d, T5a-d, LPTC-HS/VS</b></div>
          <div class="insp-line"><span>TRANSMITTERS</span><b id="inspTransmitters">ACh, GABA</b></div>
          <div style="font-size:10px; color:var(--text); line-height:1.4; margin-top:6px; border-top:1px solid var(--border); padding-top:6px;" id="inspRole">
            Terminal integration plate for visual motion columns. Four layers encode cardinal directions; converges on giant LPTC tangential cells.
          </div>
        </div>
      </div>

      <div class="section" id="synapseInspectorSec">
        <div class="sec-title">
          <span>SYNAPSE & MORPHOLOGY INSPECTOR</span>
          <span style="font-size:9px; color:var(--purple);" id="inspSynBadge">EM SYNAPSE</span>
        </div>
        <div class="insp-box">
          <div class="insp-line"><span>CONNECTION</span><b id="inspSynConnection">Mi4 (210101) ➔ T4a</b></div>
          <div class="insp-line"><span>TRANSMITTER</span><b id="inspSynTransmitter" style="color:var(--purple);">GABA (Shunting Inhibitory)</b></div>
          <div class="insp-line"><span>TARGET NEUROPIL</span><b id="inspSynNeuropil">ME_R/LP_R (Right Optic Lobe)</b></div>
          <div class="insp-line"><span>3D COORDINATES</span><b id="inspSynPos">(135.4, -17.5, 35.7) μm</b></div>
          <div class="insp-line"><span>EM CONFIDENCE</span><b id="inspSynConf">97.6% (Weight: 1.00)</b></div>
          <div style="font-size:10px; color:var(--cyan); line-height:1.4; margin-top:6px; border-top:1px solid var(--border); padding-top:6px;" id="inspSynRole">
            Lateral shunting GABAergic synapse from Mi4 columnar neuron onto delayed branch of T4a dendrite, vetoing null-direction visual motion to produce directional selectivity.
          </div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    const DATA = {payload_json};

    const state = {{
      showJFRC2: true,
      showNeuropils: true,
      showFibers: true,
      cuticleOpacity: 0.35,
      fiberFraction: 1.0,
      showForamen: true,
      showParticles: true,
      showPoints: true,
      autoRotate: false,

      // High-Fidelity Multi-Scale Granularity
      granularity: 1,
      showCartridges: false,
      showSWC: true,
      showSynapses: true,
      synapseFilter: "all",
      hoveredSynapse: null,
      hoveredNeuron: null,
      lesionMode: "intact",

      // Camera
      yaw: 0.55,
      pitch: 0.38,
      dist: 850.0,
      target: [0, -70, 0],
      isDragging: false,
      isPanning: false,
      lastX: 0,
      lastY: 0,

      // Playback
      currentEpisode: "ModelD_ActiveTree_Saccade_3",
      step: 0,
      isPlaying: false,
      playTimer: null,
      liveTick: null,

      // Stimulator
      stimType: null,
      stimTimer: 0.0,
    }};

    const NEUROPIL_INFO = {{
      ME_R: {{ name: "Right Medulla (ME_R)", system: "Optic Lobe (Motion Columns)", cells: "Mi1, Tm3, Mi4, Mi9, C2, C3", transmitters: "ACh, GABA, Glutamate", role: "Primary visual processing neuropil (M1–M10 layers); extracts contrast differences and directional motion signals." }},
      ME_L: {{ name: "Left Medulla (ME_L)", system: "Optic Lobe (Motion Columns)", cells: "Mi1, Tm3, Mi4, Mi9, C2, C3", transmitters: "ACh, GABA, Glutamate", role: "Left counterpart; computes local motion signals converging onto ipsilateral lobula plate." }},
      LOP_R: {{ name: "Right Lobula Plate (LOP_R)", system: "Optic Lobe (Tangential Motion)", cells: "T4a-d, T5a-d, LPTC-HS/VS", transmitters: "ACh, GABA", role: "Terminal integration plate for visual motion columns. Four layers encode cardinal directions; converges on giant LPTC tangential cells." }},
      LOP_L: {{ name: "Left Lobula Plate (LOP_L)", system: "Optic Lobe (Tangential Motion)", cells: "T4a-d, T5a-d, LPTC-HS/VS", transmitters: "ACh, GABA", role: "Left counterpart; integrates wide-field flow for bilateral yaw stabilization." }},
      LO_R: {{ name: "Right Lobula (LO_R)", system: "Optic Lobe (Visual Features)", cells: "LC4, LC10, LC11, LT columnar", transmitters: "ACh, GABA", role: "Extracts small visual targets, looming threats, and feature shapes (enemy sprites)." }},
      LO_L: {{ name: "Left Lobula (LO_L)", system: "Optic Lobe (Visual Features)", cells: "LC4, LC10, LC11, LT columnar", transmitters: "ACh, GABA", role: "Left counterpart for target and looming feature detection." }},
      EB: {{ name: "Ellipsoid Body (EB)", system: "Central Complex (Compass Ring)", cells: "E-PG (compass), P-EN, P-EG", transmitters: "ACh, Glutamate", role: "Toroidal compass ring. Houses an active activity bump that dynamically tracks heading angle." }},
      FB: {{ name: "Fan-shaped Body (FB)", system: "Central Complex (Steering & Goals)", cells: "P-FN, Δ7, FB columnar/tangential", transmitters: "ACh, GABA, Serotonin", role: "Computes goal-directed steering vectors and integrates path-integration coordinates." }},
      PB: {{ name: "Protocerebral Bridge (PB)", system: "Central Complex (Phase Shifter)", cells: "Δ7, PB-EB-LAL columnar", transmitters: "GABA, ACh", role: "16–18 bilateral glomeruli coordinating sinusoidal phase shifts for heading propagation." }},
      GNG_SEZ: {{ name: "Subesophageal Zone (GNG/SEZ)", system: "Sensory & Mechanotactile", cells: "Gustatory, Mechanosensory, Palp", transmitters: "ACh, Octopamine", role: "Coordinates feeding and mechanosensory palpation; triggers door USE interactions on obstacle contact." }},
      AL_R: {{ name: "Right Antennal Lobe (AL_R)", system: "Chemosensory / Olfactory", cells: "ORNs, Projection Neurons (PNs), LN", transmitters: "ACh, GABA", role: "Glomerular olfactory center receiving sensory inputs from antenna funiculus." }},
      AL_L: {{ name: "Left Antennal Lobe (AL_L)", system: "Chemosensory / Olfactory", cells: "ORNs, Projection Neurons (PNs), LN", transmitters: "ACh, GABA", role: "Left antennal lobe counterpart." }},
      VNC_T1: {{ name: "Prothoracic Neuromere (T1)", system: "Ventral Nerve Cord (Forelegs)", cells: "Foreleg Motor Neurons, Mechanosensory", transmitters: "ACh, Glutamate", role: "Commands front leg stance and pushing motions against walls and doors." }},
      VNC_T2: {{ name: "Mesothoracic Neuromere (T2)", system: "Ventral Nerve Cord (Wings & Flight)", cells: "Wing Power & Steering Motor Neurons", transmitters: "ACh, Glutamate", role: "Primary flight motor center; modulates wing-stroke amplitude and yaw torque." }},
      VNC_T3: {{ name: "Metathoracic Neuromere (T3)", system: "Ventral Nerve Cord (Hindlegs & Trigger)", cells: "Hindleg Motor Neurons, Descending Target", transmitters: "ACh, Glutamate", role: "Rear stance stability and trigger discharge circuits during combat bursts." }}
    }};

    const NEUROPIL_CENTERS = [
      {{ code: "EB", name: "Ellipsoid Body", x: 0, y: 0, z: 0 }},
      {{ code: "FB", name: "Fan-shaped Body", x: 0, y: 35, z: 12 }},
      {{ code: "PB", name: "Protocerebral Bridge", x: 0, y: 55, z: 35 }},
      {{ code: "LOP_R", name: "Right Lobula Plate", x: 170, y: 45, z: 30 }},
      {{ code: "LOP_L", name: "Left Lobula Plate", x: -170, y: 45, z: 30 }},
      {{ code: "LO_R", name: "Right Lobula", x: 150, y: 15, z: -15 }},
      {{ code: "LO_L", name: "Left Lobula", x: -150, y: 15, z: -15 }},
      {{ code: "ME_R", name: "Right Medulla", x: 240, y: 10, z: -10 }},
      {{ code: "ME_L", name: "Left Medulla", x: -240, y: 10, z: -10 }},
      {{ code: "GNG_SEZ", name: "Subesophageal Zone", x: 0, y: -45, z: -15 }},
      {{ code: "AL_R", name: "Right Antennal Lobe", x: 45, y: -35, z: -45 }},
      {{ code: "AL_L", name: "Left Antennal Lobe", x: -45, y: -35, z: -45 }},
      {{ code: "VNC_T1", name: "Prothoracic T1", x: 0, y: -150, z: -10 }},
      {{ code: "VNC_T2", name: "Mesothoracic T2", x: 0, y: -220, z: 0 }},
      {{ code: "VNC_T3", name: "Metathoracic T3", x: 0, y: -290, z: 10 }}
    ];

    const canvas = document.getElementById("renderCanvas");
    const ctx = canvas.getContext("2d");

    function resize() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}
    window.addEventListener("resize", resize);

    function init() {{
      resize();
      initEpisodeDropdown();
      updateUI();
      render();
    }}

    function initEpisodeDropdown() {{
      const sel = document.getElementById("episodeSelector");
      sel.innerHTML = "";

      const liveOpt = document.createElement("option");
      liveOpt.value = "live";
      liveOpt.textContent = "🔴 LIVE GZDOOM / TELEMETRY";
      sel.appendChild(liveOpt);

      Object.keys(DATA.episodes).forEach(k => {{
        const ep = DATA.episodes[k];
        const opt = document.createElement("option");
        opt.value = k;
        opt.textContent = `${{k}} (${{ep.player_kills}} Kills · ${{ep.health_remaining}} HP)`;
        sel.appendChild(opt);
      }});
      sel.value = "ModelD_ActiveTree_Saccade_3";
      state.currentEpisode = "ModelD_ActiveTree_Saccade_3";
    }}

    function project(x, y, z, w, h) {{
      const tx = x - state.target[0];
      const ty = y - state.target[1];
      const tz = z - state.target[2];

      const cosY = Math.cos(state.yaw);
      const sinY = Math.sin(state.yaw);
      const x1 = tx * cosY - tz * sinY;
      const z1 = tx * sinY + tz * cosY;

      const cosP = Math.cos(state.pitch);
      const sinP = Math.sin(state.pitch);
      const y2 = ty * cosP - z1 * sinP;
      const z2 = ty * sinP + z1 * cosP;

      const focal = 640.0;
      const eyeZ = z2 + state.dist;
      if (eyeZ <= 10.0) return null;

      const scale = focal / eyeZ;
      return {{
        x: w / 2 + x1 * scale,
        y: h / 2 - y2 * scale,
        scale: scale,
        eyeZ: eyeZ
      }};
    }}

    function setPreset(p) {{
      document.querySelectorAll(".cam-toolbar .btn").forEach(b => {{
        if (b.id.startsWith("btnPreset")) b.classList.remove("active");
      }});

      if (p === "flywire") {{
        state.yaw = 0.0;
        state.pitch = 0.0;
        state.dist = 560.0;
        state.target = [0, 0, 0];
        state.cuticleOpacity = 0.08;
        document.getElementById("cuticleOpacitySlider").value = 8;
        document.getElementById("lblCuticleOp").textContent = "8%";
        document.getElementById("btnPresetFlyWire").classList.add("active");
      }} else if (p === "full_fly") {{
        state.yaw = 1.35;
        state.pitch = 0.25;
        state.dist = 960.0;
        state.target = [0, -180, 0];
        state.cuticleOpacity = 0.65;
        document.getElementById("cuticleOpacitySlider").value = 65;
        document.getElementById("lblCuticleOp").textContent = "65%";
        document.getElementById("btnPresetFullFly").classList.add("active");
      }} else if (p === "iso") {{
        state.yaw = 0.55;
        state.pitch = 0.38;
        state.dist = 850.0;
        state.target = [0, -70, 0];
        state.cuticleOpacity = 0.35;
        document.getElementById("cuticleOpacitySlider").value = 35;
        document.getElementById("lblCuticleOp").textContent = "35%";
        document.getElementById("btnPresetIso").classList.add("active");
      }} else if (p === "top") {{
        state.yaw = 0.0;
        state.pitch = Math.PI / 2 - 0.02;
        state.dist = 800.0;
        state.target = [0, -40, 0];
        document.getElementById("btnPresetTop").classList.add("active");
      }}
      updateReadout();
    }}

    function toggleAutoRotate() {{
      state.autoRotate = !state.autoRotate;
      const b = document.getElementById("btnAutoRotate");
      b.textContent = `AUTO ORBIT: ${{state.autoRotate ? "ON" : "OFF"}}`;
      b.classList.toggle("active", state.autoRotate);
    }}

    function toggleCuticlePreset() {{
      state.cuticleOpacity = (state.cuticleOpacity > 0.15) ? 0.0 : 0.50;
      document.getElementById("cuticleOpacitySlider").value = Math.round(state.cuticleOpacity * 100);
      document.getElementById("lblCuticleOp").textContent = `${{Math.round(state.cuticleOpacity * 100)}}%`;
    }}

    function onCuticleOpacity(v) {{
      state.cuticleOpacity = v / 100.0;
      document.getElementById("lblCuticleOp").textContent = `${{v}}%`;
    }}

    function onFiberDensity(v) {{
      state.fiberFraction = v / 100.0;
      document.getElementById("lblFiberOp").textContent = `${{v}}%`;
    }}

    function updateReadout() {{
      const yDeg = ((state.yaw * 180 / Math.PI) % 360).toFixed(1);
      const pDeg = ((state.pitch * 180 / Math.PI) % 360).toFixed(1);
      document.getElementById("readoutCam").textContent = `CAM: Yaw ${{yDeg}}° · Pitch ${{pDeg}}° · Dist ${{state.dist.toFixed(0)}}μm`;
    }}

    canvas.addEventListener("mousedown", e => {{
      state.isDragging = (e.button === 0);
      state.isPanning = (e.button === 2 || e.button === 1);
      state.lastX = e.clientX;
      state.lastY = e.clientY;
    }});
    window.addEventListener("mouseup", () => {{ state.isDragging = false; state.isPanning = false; }});
    window.addEventListener("mousemove", e => {{
      const dx = e.clientX - state.lastX;
      const dy = e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;

      if (state.isDragging) {{
        state.yaw += dx * 0.007;
        state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.007));
        updateReadout();
      }} else if (state.isPanning) {{
        state.target[0] -= dx * 0.8;
        state.target[1] += dy * 0.8;
      }} else {{
        checkNeuropilHover(e.clientX, e.clientY);
      }}
    }});
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      state.dist = Math.max(250.0, Math.min(1800.0, state.dist + e.deltaY * 0.85));
      updateReadout();
    }}, {{ passive: false }});
    canvas.addEventListener("contextmenu", e => e.preventDefault());

    function setGranularity(level) {{
      state.granularity = level;
      ["btnGranL1", "btnGranL2", "btnGranL3"].forEach((id, idx) => {{
        const b = document.getElementById(id);
        if (b) b.classList.toggle("active", idx + 1 === level);
      }});
      ["btnSideGranL1", "btnSideGranL2", "btnSideGranL3"].forEach((id, idx) => {{
        const b = document.getElementById(id);
        if (b) b.classList.toggle("active", idx + 1 === level);
      }});

      const lbl = document.getElementById("lblGranularity");
      if (level === 1) {{
        if (lbl) lbl.textContent = "L1: MACRO CNS";
        state.showJFRC2 = true;
        state.showNeuropils = true;
        state.showFibers = true;
        state.fiberFraction = 1.0;
        state.showCartridges = false;
        state.showSWC = true;
        state.showSynapses = false;
      }} else if (level === 2) {{
        if (lbl) lbl.textContent = "L2: CIRCUITS";
        state.showJFRC2 = true;
        state.showNeuropils = true;
        state.showFibers = true;
        state.fiberFraction = 0.25;
        state.showCartridges = true;
        state.showSWC = true;
        state.showSynapses = false;
        state.target = [40, -20, 10];
        state.dist = 680.0;
      }} else if (level === 3) {{
        if (lbl) lbl.textContent = "L3: SYNAPSES";
        state.showJFRC2 = false;
        state.showNeuropils = true;
        state.showFibers = false;
        state.fiberFraction = 0.05;
        state.showCartridges = true;
        state.showSWC = true;
        state.showSynapses = true;
        state.target = [135, -20, 35];
        state.dist = 420.0;
        state.yaw = 0.45;
        state.pitch = 0.30;
      }}

      const chkCart = document.getElementById("chkCartridges");
      if (chkCart) chkCart.checked = state.showCartridges;
      const chkS = document.getElementById("chkSWC");
      if (chkS) chkS.checked = state.showSWC;
      const chkSyn = document.getElementById("chkSynapses");
      if (chkSyn) chkSyn.checked = state.showSynapses;
      const chkFib = document.getElementById("chkFibers");
      if (chkFib) chkFib.checked = state.showFibers;
      const chkJ = document.getElementById("chkJFRC2");
      if (chkJ) chkJ.checked = state.showJFRC2;

      const vmLeg = document.getElementById("vmLegend");
      if (vmLeg) vmLeg.style.display = (level === 3) ? "flex" : "none";

      updateReadout();
    }}

    const LESION_BENCHMARKS = {{
      intact: {{
        episode: "ModelD_ActiveTree_Saccade_3",
        name: "INTACT CONTROL",
        statusColor: "var(--green)",
        error: "30.9° (Locked)", errorColor: "var(--green)",
        lock: "71.4% (Stable)", lockColor: "var(--green)",
        variance: "0.7216 (Norm)", varColor: "var(--green)",
        health: "97 HP (2 Kills)", hpColor: "var(--green)",
      }},
      Mi4: {{
        episode: "ModelD_Mi4_KO_Saccade_3",
        name: "Mi4 GABA KO (SHUNT LOST)",
        statusColor: "var(--red)",
        error: "46.0° (+48.9%)", errorColor: "var(--red)",
        lock: "55.1% (-16.3%)", lockColor: "var(--amber)",
        variance: "0.0000 (COLLAPSE)", varColor: "var(--red)",
        health: "100 HP (2 Kills)", hpColor: "var(--amber)",
      }},
      Mi1: {{
        episode: "ModelD_Mi1_KO_Saccade_3",
        name: "Mi1 ACh KO (EXCITATION LOST)",
        statusColor: "var(--red)",
        error: "123.1° (SEVERE)", errorColor: "var(--red)",
        lock: "12.0% (COLLAPSED)", lockColor: "var(--red)",
        variance: "0.6842", varColor: "var(--text)",
        health: "43 HP (0 KILLS)", hpColor: "var(--red)",
      }},
      Tm3: {{
        episode: "ModelD_Tm3_KO_Saccade_3",
        name: "Tm3 ACh KO (DELAY LOST)",
        statusColor: "var(--amber)",
        error: "12.7°", errorColor: "var(--green)",
        lock: "94.6%", lockColor: "var(--green)",
        variance: "0.8523", varColor: "var(--text)",
        health: "100 HP (1 Kill)", hpColor: "var(--amber)",
      }},
      Mi9: {{
        episode: "ModelD_Mi9_KO_Saccade_3",
        name: "Mi9 GLUTAMATE KO (DISINHIBITED)",
        statusColor: "var(--amber)",
        error: "18.1°", errorColor: "var(--green)",
        lock: "75.0%", lockColor: "var(--green)",
        variance: "0.8214", varColor: "var(--amber)",
        health: "100 HP (2 Kills)", hpColor: "var(--green)",
      }}
    }};

    function setLesionMode(mode) {{
      state.lesionMode = mode;
      const bIntact = document.getElementById("btnLesionIntact");
      if (bIntact) bIntact.classList.toggle("active-intact", mode === "intact");
      ["btnLesionMi4", "btnLesionMi1", "btnLesionTm3", "btnLesionMi9"].forEach(id => {{
        const b = document.getElementById(id);
        if (b) {{
          b.classList.toggle("active", id === "btnLesion" + mode);
        }}
      }});

      const bench = LESION_BENCHMARKS[mode];
      if (bench) {{
        const st = document.getElementById("lblLesionStatus");
        if (st) {{ st.textContent = bench.name; st.style.color = bench.statusColor; }}
        const err = document.getElementById("lblLesionError");
        if (err) {{ err.textContent = bench.error; err.style.color = bench.errorColor; }}
        const lck = document.getElementById("lblLesionLock");
        if (lck) {{ lck.textContent = bench.lock; lck.style.color = bench.lockColor; }}
        const vr = document.getElementById("lblLesionVariance");
        if (vr) {{ vr.textContent = bench.variance; vr.style.color = bench.varColor; }}
        const hp = document.getElementById("lblLesionHealth");
        if (hp) {{ hp.textContent = bench.health; hp.style.color = bench.hpColor; }}

        if (DATA.episodes[bench.episode]) {{
          const sel = document.getElementById("episodeSelector");
          if (sel) sel.value = bench.episode;
          state.currentEpisode = bench.episode;
          state.step = 0;
          updateUI();
        }}
      }}
      updateMatrixModal();
    }}

    function toggleMatrixModal() {{
      const modal = document.getElementById("wiringMatrixModal");
      if (!modal) return;
      const isVis = modal.style.display === "flex";
      modal.style.display = isVis ? "none" : "flex";
      const btn = document.getElementById("btnWiringMatrix");
      if (btn) btn.classList.toggle("active", !isVis);
      if (!isVis) updateMatrixModal();
    }}

    function updateMatrixModal() {{
      const isMi4KO = state.lesionMode === "Mi4";
      const isMi1KO = state.lesionMode === "Mi1";
      const isTm3KO = state.lesionMode === "Tm3";
      const isMi9KO = state.lesionMode === "Mi9";

      const cMi4 = document.getElementById("cell_Mi4_T4a");
      if (cMi4) {{
        cMi4.classList.toggle("cell-lesioned", isMi4KO);
        cMi4.textContent = isMi4KO ? "0 (KO)" : "24 GABA";
      }}
      const cMi1 = document.getElementById("cell_Mi1_T4a");
      if (cMi1) {{
        cMi1.classList.toggle("cell-lesioned", isMi1KO);
        cMi1.textContent = isMi1KO ? "0 (KO)" : "48 ACh";
      }}
      const cTm3 = document.getElementById("cell_Tm3_T4a");
      if (cTm3) {{
        cTm3.classList.toggle("cell-lesioned", isTm3KO);
        cTm3.textContent = isTm3KO ? "0 (KO)" : "36 ACh";
      }}
      const cMi9 = document.getElementById("cell_Mi9_T4a");
      if (cMi9) {{
        cMi9.classList.toggle("cell-lesioned", isMi9KO);
        cMi9.textContent = isMi9KO ? "0 (KO)" : "18 Glu";
      }}
    }}

    function onMatrixCellClick(pre, post, transmitter) {{
      toggleMatrixModal();
      setGranularity(3);
      const syns = (DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.synapses) ? DATA.high_fidelity_pathways.synapses : [];
      const matches = syns.filter(s => s.pre_type === pre || s.transmitter === transmitter);
      if (matches.length > 0) {{
        const target = matches[0];
        inspectSynapse(target);
        state.inspectedSynapse = target;
        // Smoothly center camera toward synapse cluster
        let avgX = 0, avgY = 0, avgZ = 0;
        const sub = matches.slice(0, 10);
        sub.forEach(s => {{ avgX += s.pos[0]; avgY += s.pos[1]; avgZ += s.pos[2]; }});
        avgX /= sub.length;
        avgY /= sub.length;
        avgZ /= sub.length;
        state.panX = (280 - avgX) * 0.4;
        state.panY = (avgZ - 130) * 0.4;
      }}
    }}

    function onSynapseFilter(val) {{
      state.synapseFilter = val;
    }}

    function inspectSynapse(syn) {{
      const isLes = (state.lesionMode && state.lesionMode !== "intact" && syn.pre_type === state.lesionMode);
      const stBadge = document.getElementById("inspSynBadge");
      if (stBadge) {{
        stBadge.textContent = isLes ? "LESIONED / KNOCKED OUT" : "EM SYNAPSE";
        stBadge.style.color = isLes ? "#ff1744" : "var(--purple)";
      }}

      const conn = document.getElementById("inspSynConnection");
      if (conn) conn.textContent = `${{syn.pre_type}} (#${{syn.pre_id}}) ➔ ${{syn.post_type}}${{isLes ? " [KNOCKED OUT]" : ""}}`;
      const tx = document.getElementById("inspSynTransmitter");
      if (tx) {{
        tx.textContent = isLes ? `${{syn.transmitter}} (0.0 Conductance · Knockout)` : `${{syn.transmitter}} (${{syn.action.replace('_', ' ')}})`;
        tx.style.color = isLes ? "#ff1744" : (syn.color || "#00e5ff");
      }}
      const np = document.getElementById("inspSynNeuropil");
      if (np) np.textContent = syn.neuropil;
      const pos = document.getElementById("inspSynPos");
      if (pos) pos.textContent = `(${{syn.pos[0].toFixed(1)}}, ${{syn.pos[1].toFixed(1)}}, ${{syn.pos[2].toFixed(1)}}) μm`;
      const conf = document.getElementById("inspSynConf");
      if (conf) conf.textContent = `${{(syn.confidence * 100).toFixed(1)}}% · Weight ${{syn.weight.toFixed(2)}} · Dist ${{syn.dist_soma_um.toFixed(1)}}μm`;
      const role = document.getElementById("inspSynRole");
      if (role) {{
        let desc = "";
        if (syn.transmitter === "GABA") {{
          desc = "Lateral shunting GABAergic synapse from Mi4 columnar neuron onto delayed branch of T4a dendrite, vetoing null-direction visual motion to produce directional selectivity.";
        }} else if (syn.transmitter === "Glu") {{
          desc = "Glutamatergic distal hyperpolarizing synapse from Mi9 onto the opposing dendritic base of T4a; establishes spatial receptive field offset.";
        }} else if (syn.transmitter === "ACh" && syn.post_type === "T4a") {{
          desc = "Primary non-delayed cholinergic excitatory transmission from Mi1/Tm3 medulla columnar terminals directly onto T4a dendritic arbor, driving preferred-direction excitation.";
        }} else if (syn.pre_type === "E-PG") {{
          desc = "Central compass synaptic output from E-PG compass wedge in ellipsoid body connecting to protocerebral bridge phase-shifter interneurons (P-EN).";
        }} else if (syn.pre_type === "DNpe017") {{
          desc = "High-speed command synapse from giant descending motor neuron DNpe017 through cervical connective directly terminating on metathoracic T3 leg/trigger motor circuits.";
        }} else {{
          desc = `EM-validated chemical synapse (${{syn.transmitter}}) with ${{syn.action}} synaptic transmission in ${{syn.neuropil}}.`;
        }}
        role.textContent = desc;
      }}
    }}

    function inspectNeuron(nrn) {{
      const conn = document.getElementById("inspSynConnection");
      if (conn) conn.textContent = `Neuron: ${{nrn.cell_type}} (${{nrn.instance}})`;
      const tx = document.getElementById("inspSynTransmitter");
      if (tx) {{
        tx.textContent = nrn.neurotransmitter;
        tx.style.color = nrn.color_hex;
      }}
      const np = document.getElementById("inspSynNeuropil");
      if (np) np.textContent = `${{nrn.neuropil}} (${{nrn.hemisphere}} hemisphere)`;
      const pos = document.getElementById("inspSynPos");
      if (pos) pos.textContent = `Soma: (${{nrn.soma[0].toFixed(1)}}, ${{nrn.soma[1].toFixed(1)}}, ${{nrn.soma[2].toFixed(1)}}) μm`;
      const conf = document.getElementById("inspSynConf");
      if (conf) conf.textContent = `Arbor: ${{nrn.total_arbor_length_um.toFixed(1)}} μm · ${{nrn.nodes.length}} SWC nodes`;
      const role = document.getElementById("inspSynRole");
      if (role) role.textContent = nrn.functional_role;
    }}

    function checkNeuropilHover(clientX, clientY) {{
      const rect = canvas.getBoundingClientRect();
      const mouseX = clientX - rect.left;
      const mouseY = clientY - rect.top;
      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;
      const tooltip = document.getElementById("tooltip");

      // 1. Check chemical synapses if visible
      if (state.showSynapses && DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.synapses) {{
        let closestSyn = null;
        let minSynD2 = 256; // 16px radius squared
        const synapses = DATA.high_fidelity_pathways.synapses;

        for (let i = 0; i < synapses.length; i++) {{
          const syn = synapses[i];
          if (state.synapseFilter !== "all" && syn.transmitter !== state.synapseFilter) continue;
          const p = project(syn.pos[0], syn.pos[1], syn.pos[2], w, h);
          if (!p) continue;
          const d2 = (p.x - mouseX) ** 2 + (p.y - mouseY) ** 2;
          if (d2 < minSynD2) {{
            minSynD2 = d2;
            closestSyn = syn;
          }}
        }}

        if (closestSyn) {{
          state.hoveredSynapse = closestSyn;
          tooltip.style.display = "block";
          tooltip.style.left = (mouseX + 16) + "px";
          tooltip.style.top = (mouseY + 16) + "px";
          tooltip.innerHTML = `<span style="color:${{closestSyn.color}}; font-weight:700;">[${{closestSyn.transmitter}}] SYNAPSE #${{closestSyn.id}}</span><br>` +
                              `<b>${{closestSyn.pre_type}}</b> ➔ <b>${{closestSyn.post_type}}</b><br>` +
                              `<span style="color:var(--text-muted);">Pos: (${{closestSyn.pos[0].toFixed(1)}}, ${{closestSyn.pos[1].toFixed(1)}}, ${{closestSyn.pos[2].toFixed(1)}}) μm<br>` +
                              `Conf: ${{(closestSyn.confidence * 100).toFixed(0)}}% · Weight: ${{closestSyn.weight.toFixed(2)}}</span>`;
          inspectSynapse(closestSyn);
          return;
        }}
      }}
      state.hoveredSynapse = null;

      // 2. Check SWC Neurons
      if (state.showSWC && DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.neurons) {{
        let closestNrn = null;
        let minNrnD2 = 400; // 20px radius squared
        const neurons = DATA.high_fidelity_pathways.neurons;

        for (let i = 0; i < neurons.length; i++) {{
          const nrn = neurons[i];
          const p = project(nrn.soma[0], nrn.soma[1], nrn.soma[2], w, h);
          if (!p) continue;
          const d2 = (p.x - mouseX) ** 2 + (p.y - mouseY) ** 2;
          if (d2 < minNrnD2) {{
            minNrnD2 = d2;
            closestNrn = nrn;
          }}
        }}

        if (closestNrn) {{
          tooltip.style.display = "block";
          tooltip.style.left = (mouseX + 16) + "px";
          tooltip.style.top = (mouseY + 16) + "px";
          tooltip.innerHTML = `<span style="color:${{closestNrn.color_hex}}; font-weight:700;">SWC NEURON: ${{closestNrn.cell_type}}</span><br>` +
                              `<span style="color:var(--text-muted);">${{closestNrn.instance}} · ${{closestNrn.neurotransmitter}}</span><br>` +
                              `Arbor: ${{closestNrn.total_arbor_length_um.toFixed(1)}}μm (${{closestNrn.nodes.length}} nodes)`;
          inspectNeuron(closestNrn);
          return;
        }}
      }}

      // 3. Fall back to Neuropil Centers
      let closest = null;
      let minD2 = 1200; // 35px radius squared

      for (const np of NEUROPIL_CENTERS) {{
        const p = project(np.x, np.y, np.z, w, h);
        if (!p) continue;
        const d2 = (p.x - mouseX) ** 2 + (p.y - mouseY) ** 2;
        if (d2 < minD2) {{
          minD2 = d2;
          closest = np;
        }}
      }}

      if (closest) {{
        tooltip.style.display = "block";
        tooltip.style.left = (mouseX + 16) + "px";
        tooltip.style.top = (mouseY + 16) + "px";
        tooltip.innerHTML = `<strong style="color:var(--cyan);">${{closest.name}}</strong> <span style="color:var(--text-muted);">[${{closest.code}}]</span>`;
        inspectNeuropil(closest.code);
      }} else {{
        tooltip.style.display = "none";
      }}
    }}

    function inspectNeuropil(code) {{
      const info = NEUROPIL_INFO[code];
      if (!info) return;
      document.getElementById("inspName").textContent = info.name;
      document.getElementById("inspSystem").textContent = info.system;
      document.getElementById("inspCells").textContent = info.cells;
      document.getElementById("inspTransmitters").textContent = info.transmitters;
      document.getElementById("inspRole").textContent = info.role;
    }}

    let livePollTimer = null;
    let liveConnected = false;

    function startLivePolling() {{
      if (livePollTimer) clearInterval(livePollTimer);
      pollLiveTelemetry();
      livePollTimer = setInterval(pollLiveTelemetry, 90);
    }}

    function stopLivePolling() {{
      if (livePollTimer) {{
        clearInterval(livePollTimer);
        livePollTimer = null;
      }}
    }}

    async function pollLiveTelemetry() {{
      if (state.currentEpisode !== "live") return;
      try {{
        const resp = await fetch("/api/telemetry", {{ cache: "no-store" }});
        if (!resp.ok) throw new Error("Status " + resp.status);
        const data = await resp.json();
        liveConnected = Boolean(data.connected);
        state.liveTick = {{
          action: data.action || "FORWARD",
          health: data.health ?? 100,
          player_kills: data.kills ?? 0,
          friendly_fire_kills: 0,
          damage_dealt: data.damage_dealt ?? 0.0,
          norm_asymmetry: Number((data.neural && data.neural.norm_asymmetry) ?? 0.0),
          firing_solution_locked: (data.action === "FIRE") ? 1.0 : 0.0,
          door_candidate: (data.action === "USE") ? 1.0 : 0.0,
          health_priority_active: (data.health && data.health < 40) ? 1.0 : 0.0,
          combat_active: (data.action === "FIRE" || (data.kills && data.kills > 0)),
          step: data.step ?? 0
        }};
        document.getElementById("telemCondBadge").textContent = liveConnected ? "🔴 LIVE NATIVE GZDOOM" : "WAITING FOR GZDOOM...";
        document.getElementById("telemCondBadge").style.color = liveConnected ? "var(--green)" : "var(--amber)";
        updateUI();
      }} catch (e) {{
        liveConnected = false;
        document.getElementById("telemCondBadge").textContent = "WAITING FOR GZDOOM...";
        document.getElementById("telemCondBadge").style.color = "var(--amber)";
      }}
    }}

    function onEpisodeChange() {{
      state.currentEpisode = document.getElementById("episodeSelector").value;
      if (state.currentEpisode === "live") {{
        if (state.isPlaying) togglePlay();
        document.getElementById("stepSlider").disabled = true;
        startLivePolling();
      }} else {{
        stopLivePolling();
        document.getElementById("stepSlider").disabled = false;
        state.step = 0;
        document.getElementById("telemCondBadge").textContent = state.currentEpisode.replace("ModelD_", "").replace("_Saccade_3", "");
        document.getElementById("telemCondBadge").style.color = "var(--amber)";
      }}
      updateUI();
    }}

    function onScrub(v) {{
      if (state.currentEpisode === "live") return;
      state.step = parseInt(v, 10);
      updateUI();
    }}

    function stepDelta(d) {{
      if (state.currentEpisode === "live") return;
      const ep = DATA.episodes[state.currentEpisode];
      if (!ep) return;
      state.step = Math.max(0, Math.min(ep.ticks.length - 1, state.step + d));
      updateUI();
    }}

    function togglePlay() {{
      if (state.currentEpisode === "live") return;
      state.isPlaying = !state.isPlaying;
      const b = document.getElementById("btnPlay");
      b.textContent = state.isPlaying ? "❚❚ PAUSE" : "▶ PLAY";
      b.classList.toggle("active", state.isPlaying);

      if (state.isPlaying) {{
        if (state.playTimer) clearInterval(state.playTimer);
        state.playTimer = setInterval(() => {{
          const ep = DATA.episodes[state.currentEpisode];
          if (!ep) return;
          state.step = (state.step >= ep.ticks.length - 1) ? 0 : state.step + 1;
          updateUI();
        }}, 100);
      }} else if (state.playTimer) {{
        clearInterval(state.playTimer);
        state.playTimer = null;
      }}
    }}

    function triggerStim(type) {{
      state.stimType = type;
      state.stimTimer = 1.0;
    }}

    function updateUI() {{
      if (state.currentEpisode === "live") {{
        const t = state.liveTick || {{ action: "WAITING...", health: 100, player_kills: 0, friendly_fire_kills: 0, damage_dealt: 0, norm_asymmetry: 0, step: 0 }};
        document.getElementById("lblStep").textContent = `LIVE #${{String(t.step).padStart(4, "0")}}`;
        document.getElementById("lblTime").textContent = liveConnected ? "STREAMING" : "CONNECTING";
        document.getElementById("lblTime").style.color = liveConnected ? "var(--green)" : "var(--amber)";
        document.getElementById("telemAction").textContent = t.action;
        document.getElementById("telemHealth").textContent = `${{Math.round(t.health)}} HP`;
        document.getElementById("telemKills").textContent = `${{t.player_kills}} PK`;
        document.getElementById("telemDmg").textContent = `${{Math.round(t.damage_dealt)}}`;
        document.getElementById("telemAsym").textContent = `${{t.norm_asymmetry.toFixed(3)}}`;
        const lock = (t.firing_solution_locked === 1.0);
        document.getElementById("telemLock").textContent = lock ? "LOCKED (≤18°)" : (t.combat_active ? "ACQUIRING..." : "NO TARGET");
        document.getElementById("telemLock").style.color = lock ? "var(--green)" : (t.combat_active ? "var(--amber)" : "var(--text-muted)");
        return;
      }}

      const ep = DATA.episodes[state.currentEpisode];
      if (!ep || !ep.ticks[state.step]) return;
      const t = ep.ticks[state.step];

      document.getElementById("lblStep").textContent = `${{String(state.step).padStart(3, "0")}} / ${{ep.ticks.length - 1}}`;
      document.getElementById("lblTime").textContent = `${{(state.step * 0.1).toFixed(2)}}s`;
      document.getElementById("lblTime").style.color = "var(--text)";
      document.getElementById("stepSlider").value = state.step;
      document.getElementById("stepSlider").max = ep.ticks.length - 1;

      document.getElementById("telemAction").textContent = t.action;
      document.getElementById("telemHealth").textContent = `${{Math.round(t.health)}} HP`;
      document.getElementById("telemKills").textContent = `${{t.player_kills}} (PK) / ${{t.friendly_fire_kills}} (FF)`;
      document.getElementById("telemDmg").textContent = `${{Math.round(t.damage_dealt)}}`;
      document.getElementById("telemAsym").textContent = `${{t.norm_asymmetry.toFixed(3)}}`;

      const lock = (t.firing_solution_locked === 1.0);
      document.getElementById("telemLock").textContent = lock ? "LOCKED (≤18°)" : (t.combat_active ? "ACQUIRING..." : "NO TARGET");
      document.getElementById("telemLock").style.color = lock ? "var(--green)" : (t.combat_active ? "var(--amber)" : "var(--text-muted)");
    }}

    function blendRgb(hex, targetR, targetG, targetB, t) {{
      t = Math.max(0, Math.min(1, t));
      const num = parseInt(hex.replace('#', ''), 16) || 0;
      const r1 = (num >> 16) & 255;
      const g1 = (num >> 8) & 255;
      const b1 = num & 255;
      const r = Math.round(r1 * (1 - t) + targetR * t);
      const g = Math.round(g1 * (1 - t) + targetG * t);
      const b = Math.round(b1 * (1 - t) + targetB * t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}

    let particlePhase = 0.0;

    function render() {{
      requestAnimationFrame(render);

      if (state.autoRotate) {{
        state.yaw += 0.005;
        updateReadout();
      }}

      let asym = 0.0;
      let isFire = false;
      let isDoor = false;
      let isDamage = false;

      if (state.currentEpisode === "live" && state.liveTick) {{
        const t = state.liveTick;
        asym = t.norm_asymmetry || 0.0;
        isFire = (t.action === "FIRE" || t.firing_solution_locked === 1.0);
        isDoor = (t.action === "USE" || t.door_candidate === 1.0);
        isDamage = (t.health_priority_active === 1.0);
      }} else {{
        const ep = DATA.episodes[state.currentEpisode];
        if (ep && ep.ticks[state.step]) {{
          const t = ep.ticks[state.step];
          asym = t.norm_asymmetry || 0.0;
          isFire = (t.action === "FIRE" || t.firing_solution_locked === 1.0);
          isDoor = (t.action === "USE" || t.door_candidate === 1.0);
          isDamage = (t.health_priority_active === 1.0);
        }}
      }}

      if (state.stimTimer > 0) {{
        state.stimTimer -= 0.015;
        const decay = Math.max(0, state.stimTimer);
        if (state.stimType === "turn_right") asym = 0.9 * decay;
        else if (state.stimType === "turn_left") asym = -0.9 * decay;
        else if (state.stimType === "fire") isFire = true;
        else if (state.stimType === "door") isDoor = true;
        else if (state.stimType === "panic") isDamage = true;
      }}

      particlePhase += 0.04;

      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;
      ctx.clearRect(0, 0, w, h);

      // 1. Render 3D Exoskeleton Cuticle Wireframe
      if (state.cuticleOpacity > 0.01) {{
        const lines = DATA.exoskeleton.lines;
        for (let i = 0; i < lines.length; i++) {{
          const l = lines[i];
          const p1 = project(l[0], l[1], l[2], w, h);
          const p2 = project(l[3], l[4], l[5], w, h);
          if (!p1 || !p2) continue;

          let alpha = state.cuticleOpacity * 0.45;
          let stroke = "rgba(0, 229, 255, " + alpha + ")";
          let lw = 0.8;

          if (l[6] === "wing" || l[6] === "wing_vein") {{
            stroke = "rgba(100, 220, 255, " + (state.cuticleOpacity * 0.35) + ")";
            lw = (l[6] === "wing") ? 1.4 : 0.7;
          }} else if (l[6] === "eye") {{
            stroke = "rgba(180, 100, 255, " + (state.cuticleOpacity * 0.6) + ")";
            lw = 1.0;
          }} else if (l[6] === "leg") {{
            stroke = "rgba(0, 200, 255, " + (state.cuticleOpacity * 0.5) + ")";
            lw = 1.2;
          }}

          ctx.strokeStyle = stroke;
          ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }}
      }}

      // 2. Render Official JFRC2 Whole-Brain Surface Mesh (6,654 biological edges)
      if (state.showJFRC2 && DATA.brain_mesh) {{
        ctx.strokeStyle = "rgba(40, 140, 220, 0.18)";
        ctx.lineWidth = 0.7;
        const bEdges = DATA.brain_mesh;
        for (let i = 0; i < bEdges.length; i += 2) {{
          const e = bEdges[i];
          const p1 = project(e[0], e[1], e[2], w, h);
          const p2 = project(e[3], e[4], e[5], w, h);
          if (!p1 || !p2) continue;

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }}
      }}

      // 3. Render Authentic Neuropil Meshes (EB, FB, PB, ME, LOP, LO, SEZ)
      if (state.showNeuropils && DATA.neuropil_meshes) {{
        for (let np of DATA.neuropil_meshes) {{
          let npColor = np.color;
          let npAlpha = 0.35;

          const isRight = np.code.endsWith("_R");
          const isLeft = np.code.endsWith("_L");
          const drive = isRight ? Math.max(0, asym) : (isLeft ? Math.max(0, -asym) : 0);

          if (isFire && (np.code.startsWith("LO_") || np.code.startsWith("VNC_"))) {{
            npColor = "#ff3300";
            npAlpha = 0.9;
          }} else if (isDoor && np.code.startsWith("GNG")) {{
            npColor = "#00ffaa";
            npAlpha = 0.9;
          }} else if (drive > 0.04) {{
            const blend = Math.min(1.0, (drive - 0.04) / 0.45);
            npColor = blendRgb(np.color, 245, 158, 11, blend);
            npAlpha = 0.35 + 0.45 * blend;
          }}

          ctx.strokeStyle = npColor;
          ctx.globalAlpha = npAlpha;
          ctx.lineWidth = 0.9;

          for (let e of np.edges) {{
            const p1 = project(e[0], e[1], e[2], w, h);
            const p2 = project(e[3], e[4], e[5], w, h);
            if (!p1 || !p2) continue;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }}
          ctx.globalAlpha = 1.0;
        }}
      }}

      // 4. Render Dense FlyWire Connectome Fibers (3,030 neuron fibers)
      if (state.showFibers) {{
        const fibers = DATA.connectome.fibers;
        const count = Math.floor(fibers.length * state.fiberFraction);

        for (let i = 0; i < count; i++) {{
          const f = fibers[i];
          const pts = f.points;
          if (pts.length < 2) continue;

          let strokeColor = f.color;
          let alpha = 0.65;

          const isRight = f.neuropil.endsWith("_R") || f.points[0][0] > 0;
          const isLeft = f.neuropil.endsWith("_L") || f.points[0][0] < 0;
          const drive = isRight ? Math.max(0, asym) : (isLeft ? Math.max(0, -asym) : 0);

          if (isFire && (f.neuropil.startsWith("LO_") || f.neuropil.startsWith("VNC_") || f.neuropil === "DN_TRUNK")) {{
            strokeColor = "#ff3300";
            alpha = 1.0;
          }} else if (isDoor && (f.neuropil === "SEZ" || f.neuropil === "VNC_T1")) {{
            strokeColor = "#00ffaa";
            alpha = 1.0;
          }} else if (isDamage && f.neuropil.startsWith("MB_")) {{
            strokeColor = "#ff1744";
            alpha = 1.0;
          }} else if (drive > 0.04) {{
            const blend = Math.min(1.0, (drive - 0.04) / 0.45);
            strokeColor = blendRgb(f.color, 245, 158, 11, blend);
            alpha = 0.65 + 0.3 * blend;
          }}

          ctx.strokeStyle = strokeColor;
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 1.0;
          ctx.beginPath();

          let started = false;
          for (let j = 0; j < pts.length; j++) {{
            const p = project(pts[j][0], pts[j][1], pts[j][2], w, h);
            if (!p) {{ started = false; continue; }}
            if (!started) {{
              ctx.moveTo(p.x, p.y);
              started = true;
            }} else {{
              ctx.lineTo(p.x, p.y);
            }}
          }}
          if (started) ctx.stroke();
          ctx.globalAlpha = 1.0;

          // Traveling Action Potential Pulses
          if (state.showParticles && (i % 4 === 0)) {{
            const t = (particlePhase + i * 0.08) % 1.0;
            const idx = Math.min(pts.length - 2, Math.floor(t * (pts.length - 1)));
            const frac = (t * (pts.length - 1)) - idx;
            const pA = pts[idx];
            const pB = pts[idx + 1];
            const interpX = pA[0] + (pB[0] - pA[0]) * frac;
            const interpY = pA[1] + (pB[1] - pA[1]) * frac;
            const interpZ = pA[2] + (pB[2] - pA[2]) * frac;

            const pScreen = project(interpX, interpY, interpZ, w, h);
            if (pScreen) {{
              ctx.fillStyle = (alpha > 0.8) ? "#ffdd00" : "#00ffff";
              ctx.beginPath();
              ctx.arc(pScreen.x, pScreen.y, 1.4 * pScreen.scale, 0, 2 * Math.PI);
              ctx.fill();
            }}
          }}
        }}
      }}

      // 5. Central Esophageal Foramen
      if (state.showForamen) {{
        const foramenTop = project(0, -15, 0, w, h);
        const foramenBot = project(0, -65, -10, w, h);
        if (foramenTop && foramenBot) {{
          ctx.strokeStyle = "rgba(255, 50, 100, 0.5)";
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.ellipse(foramenTop.x, (foramenTop.y + foramenBot.y) / 2, 8 * foramenTop.scale, 20 * foramenTop.scale, 0, 0, 2 * Math.PI);
          ctx.stroke();
        }}
      }}

      // 6. Volumetric Synaptic Nuclei
      if (state.showPoints) {{
        const nodes = DATA.anatomy.nodes;
        for (let i = 0; i < nodes.length; i += 3) {{
          const n = nodes[i];
          const p = project(n.x, n.y, n.z, w, h);
          if (!p) continue;

          let ptCol = "rgba(0, 229, 255, 0.35)";
          if (asym > 0.15 && n.x > 0) ptCol = "rgba(255, 152, 0, 0.7)";
          else if (asym < -0.15 && n.x < 0) ptCol = "rgba(255, 152, 0, 0.7)";
          else if (isFire && n.compartment.startsWith("VNC_T3")) ptCol = "rgba(255, 50, 0, 0.9)";

          ctx.fillStyle = ptCol;
          ctx.beginPath();
          ctx.arc(p.x, p.y, 1.5 * p.scale, 0, 2 * Math.PI);
          ctx.fill();
        }}
      }}

      // 7. Retinotopic Cartridges (128 visual columns)
      if (state.showCartridges && DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.cartridges) {{
        const cartridges = DATA.high_fidelity_pathways.cartridges;
        for (let i = 0; i < cartridges.length; i++) {{
          const c = cartridges[i];
          const pEye = project(c.ommatidium[0], c.ommatidium[1], c.ommatidium[2], w, h);
          const pMed = project(c.medulla[0], c.medulla[1], c.medulla[2], w, h);
          const pLop = project(c.lobula_plate[0], c.lobula_plate[1], c.lobula_plate[2], w, h);
          if (!pEye || !pMed) continue;

          let colColor = "rgba(0, 229, 255, 0.4)";
          let colAlpha = 0.5;
          let colWidth = 0.8;

          // Optic flow / combat excitation
          const cDrive = (c.eye === "R") ? Math.max(0, asym) : Math.max(0, -asym);
          if (isFire) {{
            colColor = "#00ffaa";
            colAlpha = 0.8;
            colWidth = 1.2;
          }} else if (cDrive > 0.04) {{
            const blend = Math.min(1.0, (cDrive - 0.04) / 0.45);
            colColor = blendRgb("#00e5ff", 245, 158, 11, blend);
            colAlpha = 0.5 + 0.4 * blend;
            colWidth = 0.8 + 0.6 * blend;
          }}

          ctx.strokeStyle = colColor;
          ctx.globalAlpha = colAlpha;
          ctx.lineWidth = colWidth;

          ctx.beginPath();
          ctx.moveTo(pEye.x, pEye.y);
          ctx.lineTo(pMed.x, pMed.y);
          if (pLop) ctx.lineTo(pLop.x, pLop.y);
          ctx.stroke();

          // Ommatidial facet bead
          ctx.fillStyle = colColor;
          ctx.beginPath();
          ctx.arc(pEye.x, pEye.y, 1.2 * pEye.scale, 0, 2 * Math.PI);
          ctx.fill();
        }}
        ctx.globalAlpha = 1.0;
      }}

      // 8. SWC Morphology Skeletons (14 canonical neurons)
      if (state.showSWC && DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.neurons) {{
        const neurons = DATA.high_fidelity_pathways.neurons;
        const activeLesion = state.lesionMode;

        for (let i = 0; i < neurons.length; i++) {{
          const nrn = neurons[i];
          const isLesioned = (activeLesion && activeLesion !== "intact" && nrn.cell_type === activeLesion);
          const nodeMap = new Map();
          for (let k = 0; k < nrn.nodes.length; k++) {{
            nodeMap.set(nrn.nodes[k][0], nrn.nodes[k]);
          }}

          let strokeColor = isLesioned ? "rgba(100, 116, 139, 0.45)" : nrn.color_hex;
          let alpha = isLesioned ? 0.35 : 0.9;
          let lwScale = isLesioned ? 0.7 : 1.0;

          // Dynamic excitation response
          if (!isLesioned) {{
            const nrnDrive = (nrn.cell_type === "T4a" || nrn.cell_type === "Mi1" || nrn.cell_type === "Tm3")
              ? Math.max(0, asym)
              : ((nrn.cell_type === "Mi4" || nrn.cell_type === "Mi9") ? Math.max(0, -asym) : 0);

            if (isFire && nrn.cell_type === "DNpe017") {{
              strokeColor = "#ff1744";
              alpha = 1.0;
              lwScale = 2.0;
            }} else if (nrnDrive > 0.04) {{
              const blend = Math.min(1.0, (nrnDrive - 0.04) / 0.45);
              strokeColor = (nrn.cell_type === "Mi4" || nrn.cell_type === "Mi9")
                ? blendRgb(nrn.color_hex, 244, 63, 94, blend)
                : blendRgb(nrn.color_hex, 255, 234, 0, blend);
              alpha = 0.9 + 0.1 * blend;
              lwScale = 1.0 + 0.6 * blend;
            }}
          }}

          ctx.strokeStyle = strokeColor;
          ctx.globalAlpha = alpha;
          if (isLesioned) ctx.setLineDash([4, 4]);
          else ctx.setLineDash([]);

          for (let k = 0; k < nrn.nodes.length; k++) {{
            const nd = nrn.nodes[k];
            const pId = nd[6];
            if (pId === -1) {{
              // Soma
              const pSoma = project(nd[2], nd[3], nd[4], w, h);
              if (pSoma) {{
                ctx.fillStyle = strokeColor;
                ctx.beginPath();
                ctx.arc(pSoma.x, pSoma.y, Math.max(3.0, nd[5] * 2.5 * pSoma.scale), 0, 2 * Math.PI);
                ctx.fill();

                if (isLesioned) {{
                  ctx.strokeStyle = "#ff1744";
                  ctx.lineWidth = 2.0;
                  ctx.beginPath();
                  ctx.moveTo(pSoma.x - 5, pSoma.y - 5);
                  ctx.lineTo(pSoma.x + 5, pSoma.y + 5);
                  ctx.moveTo(pSoma.x + 5, pSoma.y - 5);
                  ctx.lineTo(pSoma.x - 5, pSoma.y + 5);
                  ctx.stroke();
                }}

                if (state.dist < 650) {{
                  ctx.font = "9px monospace";
                  ctx.fillStyle = isLesioned ? "#ff1744" : "#ffffff";
                  ctx.fillText(nrn.cell_type + (isLesioned ? " [KO]" : ""), pSoma.x + 6, pSoma.y - 4);
                }}
              }}
              continue;
            }}

            const parentNd = nodeMap.get(pId);
            if (!parentNd) continue;

            const p1 = project(nd[2], nd[3], nd[4], w, h);
            const p2 = project(parentNd[2], parentNd[3], parentNd[4], w, h);
            if (!p1 || !p2) continue;

            const lineW = Math.max(1.2, Math.min(4.5, nd[5] * 2.0 * p1.scale * lwScale));
            ctx.lineWidth = lineW;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }}
          ctx.setLineDash([]);
          ctx.globalAlpha = 1.0;
        }}
      }}

      // 9. Chemical Synapse Point Cloud (262 EM-registered contacts)
      if (state.showSynapses && DATA.high_fidelity_pathways && DATA.high_fidelity_pathways.synapses) {{
        const synapses = DATA.high_fidelity_pathways.synapses;
        const activeLesion = state.lesionMode;

        for (let i = 0; i < synapses.length; i++) {{
          const syn = synapses[i];
          if (state.synapseFilter !== "all" && syn.transmitter !== state.synapseFilter) continue;

          const p = project(syn.pos[0], syn.pos[1], syn.pos[2], w, h);
          if (!p) continue;

          const isLesioned = (activeLesion && activeLesion !== "intact" && syn.pre_type === activeLesion);
          let synColor = isLesioned ? "rgba(100, 116, 139, 0.4)" : (syn.color || "#00e5ff");
          let synRadius = Math.max(2.0, (1.8 + syn.weight * 1.6) * p.scale);
          let alpha = isLesioned ? 0.3 : 0.85;

          if (!isLesioned) {{
            if (isFire && syn.pre_type === "DNpe017") {{
              synColor = "#ff0033";
              synRadius *= 1.8;
              alpha = 1.0;
            }} else if (asym > 0.15 && syn.pre_type === "Mi1") {{
              synColor = "#ffdd00";
              synRadius *= 1.5;
              alpha = 1.0;
            }} else if (isDamage && syn.transmitter === "GABA") {{
              synColor = "#e879f9";
              synRadius *= 1.7;
              alpha = 1.0;
            }}
          }}

          ctx.fillStyle = synColor;
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.arc(p.x, p.y, synRadius, 0, 2 * Math.PI);
          ctx.fill();

          if (isLesioned) {{
            ctx.strokeStyle = "#ff1744";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(p.x, p.y, synRadius + 3, 0, 2 * Math.PI);
            ctx.stroke();
          }}

          // Highlight ring if hovered
          if (state.hoveredSynapse && state.hoveredSynapse.id === syn.id) {{
            ctx.strokeStyle = isLesioned ? "#ff1744" : "#ffffff";
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.arc(p.x, p.y, synRadius + 4, 0, 2 * Math.PI);
            ctx.stroke();
          }}
        }}
        ctx.globalAlpha = 1.0;
      }}

      // Dynamic Vm Readout Update
      const vmLbl = document.getElementById("lblVmDynamic");
      if (vmLbl) {{
        if (asym > 0.15) {{
          vmLbl.textContent = (state.lesionMode === "Mi4") ? "-38 mV (DISINHIBITED)" : "-32 mV (PREFERRED SPIKE)";
          vmLbl.style.color = (state.lesionMode === "Mi4") ? "#ff3366" : "#ffea00";
        }} else if (asym < -0.15) {{
          vmLbl.textContent = (state.lesionMode === "Mi4") ? "-42 mV (NULL VETO COLLAPSED)" : "-72 mV (SHUNTER CLAMP)";
          vmLbl.style.color = (state.lesionMode === "Mi4") ? "#ff1744" : "#7c3aed";
        }} else if (isFire) {{
          vmLbl.textContent = "-30 mV (MOTOR BURST)";
          vmLbl.style.color = "#ff1744";
        }} else {{
          vmLbl.textContent = "-65 mV (REST)";
          vmLbl.style.color = "#06b6d4";
        }}
      }}
    }}

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""

    artifact_file = Path("/Users/ljp176/.gemini/antigravity/brain/fd394349-d70b-4c24-b0d6-2c95c6d775d7/fly_3d_visible_nervous_system.html")
    with open(artifact_file, "w") as f:
        f.write(html)
    print(f"Saved artifact to {artifact_file} ({artifact_file.stat().st_size / 1024:.1f} KB)")

    web_file = Path("web/fly_3d_visible_nervous_system.html")
    with open(web_file, "w") as f:
        f.write(html)
    print(f"Saved web file to {web_file} ({web_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
