"""Generate the complete, self-contained interactive 3D Fly Nervous System Digital Twin.

Embeds authentic 3D Drosophila CNS anatomy and live combat episode replays.
Uses pure hardware-accelerated Canvas/WebGL with zero external CDN dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_html_content(data: dict) -> str:
    data_json = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Drosophila melanogaster — Digital Nervous System Observatory</title>
  <style>
    :root {{
      --bg: #070d11;
      --panel-bg: rgba(11, 23, 30, 0.88);
      --border: #1a3844;
      --border-bright: #2d5a6c;
      --cyan: #00e5ff;
      --cyan-glow: rgba(0, 229, 255, 0.4);
      --amber: #ff9800;
      --amber-glow: rgba(255, 152, 0, 0.5);
      --purple: #d07af5;
      --red: #ff3366;
      --green: #00ffaa;
      --text: #e2f1f8;
      --text-muted: #8aa4b2;
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

    /* Top Navigation / Header */
    header {{
      height: 52px;
      background: #09151c;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      z-index: 20;
    }}
    .title-group {{ display: flex; align-items: center; gap: 14px; }}
    .title-badge {{
      background: rgba(0, 229, 255, 0.15);
      border: 1px solid var(--cyan);
      color: var(--cyan);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
    .title-text {{ font-size: 14px; font-weight: 700; letter-spacing: 0.06em; }}
    .subtitle-text {{ font-size: 11px; color: var(--text-muted); }}

    .header-metrics {{
      display: flex;
      gap: 24px;
      font-size: 11px;
    }}
    .metric-chip {{ display: flex; flex-direction: column; }}
    .metric-chip span {{ color: var(--text-muted); font-size: 9px; letter-spacing: 0.05em; }}
    .metric-chip strong {{ color: var(--cyan); font-size: 12px; }}

    /* Main Container */
    #main {{
      flex: 1;
      position: relative;
      display: flex;
      overflow: hidden;
    }}

    /* 3D Viewport */
    #viewport-container {{
      flex: 1;
      position: relative;
      height: 100%;
      background: radial-gradient(circle at 50% 50%, #0d1e27 0%, #060b0e 80%);
    }}
    canvas#cnsCanvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    canvas#cnsCanvas:active {{ cursor: grabbing; }}

    /* Grid overlay */
    .viewport-grid {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(var(--border) 1px, transparent 1px),
                        linear-gradient(90deg, var(--border) 1px, transparent 1px);
      background-size: 40px 40px;
      opacity: 0.15;
    }}

    /* Viewport Floating HUD */
    .viewport-hud-top {{
      position: absolute;
      top: 16px;
      left: 20px;
      display: flex;
      gap: 10px;
      pointer-events: auto;
      z-index: 10;
    }}
    .hud-btn {{
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
    }}
    .hud-btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
    .hud-btn.active {{
      background: rgba(0, 229, 255, 0.18);
      border-color: var(--cyan);
      color: var(--cyan);
      box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
    }}

    .viewport-hud-coords {{
      position: absolute;
      bottom: 16px;
      left: 20px;
      font-size: 10px;
      color: var(--text-muted);
      pointer-events: none;
      line-height: 1.6;
    }}

    /* Side Control Panels */
    aside#controls {{
      width: 360px;
      background: var(--panel-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      backdrop-filter: blur(12px);
      z-index: 15;
      overflow-y: auto;
    }}

    .panel-section {{
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .section-header {{
      font-size: 11px;
      font-weight: 700;
      color: var(--cyan);
      letter-spacing: 0.1em;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    /* Replay Control Bar */
    .mode-tabs {{
      display: flex;
      background: #060e13;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 2px;
      margin-bottom: 14px;
    }}
    .mode-tab {{
      flex: 1;
      padding: 7px 0;
      text-align: center;
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      cursor: pointer;
      border-radius: 4px;
      transition: all 0.15s;
    }}
    .mode-tab.active {{
      background: #112833;
      color: var(--cyan);
      box-shadow: 0 0 8px rgba(0, 229, 255, 0.2);
    }}

    .select-dropdown {{
      width: 100%;
      background: #09171f;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      border-radius: 4px;
      outline: none;
      margin-bottom: 12px;
    }}
    .select-dropdown:focus {{ border-color: var(--cyan); }}

    .scrubber-group {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .scrubber-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-muted);
    }}
    .timeline-slider {{
      width: 100%;
      height: 6px;
      -webkit-appearance: none;
      background: #0c1a22;
      border-radius: 3px;
      outline: none;
      cursor: pointer;
    }}
    .timeline-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 8px var(--cyan);
      cursor: pointer;
    }}

    .playback-btns {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    .play-btn {{
      flex: 1;
      background: #112934;
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 8px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }}
    .play-btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
    .play-btn.playing {{ background: rgba(0, 229, 255, 0.2); border-color: var(--cyan); color: var(--cyan); }}

    /* Live Telemetry Card */
    .telemetry-card {{
      background: #08141b;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 14px;
      margin-top: 12px;
    }}
    .telem-item {{ display: flex; flex-direction: column; }}
    .telem-label {{ font-size: 9px; color: var(--text-muted); }}
    .telem-val {{ font-size: 13px; font-weight: 700; color: var(--text); }}
    .telem-val.action {{ color: var(--amber); }}
    .telem-val.lock {{ color: var(--green); }}
    .telem-val.damage {{ color: var(--red); }}

    /* Activity Trigger Buttons */
    .stim-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }}
    .stim-btn {{
      background: #091720;
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 9px 8px;
      border-radius: 4px;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s;
    }}
    .stim-btn:hover {{
      border-color: var(--amber);
      background: rgba(255, 152, 0, 0.1);
      color: var(--amber);
    }}
    .stim-btn.active {{
      border-color: var(--amber);
      background: rgba(255, 152, 0, 0.25);
      color: #fff;
      box-shadow: 0 0 10px var(--amber-glow);
    }}

    /* Inspector Card */
    .inspector-card {{
      background: #08141b;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font-size: 11px;
    }}
    .insp-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
    .insp-label {{ color: var(--text-muted); }}
    .insp-val {{ font-weight: 700; color: var(--cyan); }}
    .insp-desc {{ color: var(--text); font-size: 10px; line-height: 1.4; margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.08); }}

    /* Legend */
    .legend-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 10px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

    /* Tooltip */
    #tooltip {{
      position: absolute;
      background: rgba(5, 12, 17, 0.94);
      border: 1px solid var(--cyan);
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 11px;
      pointer-events: none;
      display: none;
      z-index: 100;
      box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    }}
  </style>
</head>
<body>

  <header>
    <div class="title-group">
      <span class="title-badge">BIOLOGICAL DIGITAL TWIN</span>
      <div>
        <div class="title-text">DROSOPHILA CENTRAL NERVOUS SYSTEM</div>
        <div class="subtitle-text">Janelia MaleCNS v1.0 / JRC2018 Connectome Architecture</div>
      </div>
    </div>
    <div class="header-metrics">
      <div class="metric-chip">
        <span>TOTAL NEURONS</span>
        <strong>~130,000 CNS</strong>
      </div>
      <div class="metric-chip">
        <span>OPTIC LOBES</span>
        <strong>2 × 70,000</strong>
      </div>
      <div class="metric-chip">
        <span>CENTRAL COMPLEX</span>
        <strong>16 E-PG WEDGES</strong>
      </div>
      <div class="metric-chip">
        <span>SYNAPSE COUNT</span>
        <strong>5.4 × 10⁷</strong>
      </div>
      <div class="metric-chip">
        <span>BIOPHYSICAL MODEL</span>
        <strong id="modelHeader">MODEL D (ACTIVE TREE)</strong>
      </div>
    </div>
  </header>

  <div id="main">
    <div id="viewport-container">
      <div class="viewport-grid"></div>
      <canvas id="cnsCanvas"></canvas>

      <div class="viewport-hud-top">
        <button class="hud-btn active" id="btnViewDorsal" onclick="setViewPreset('dorsal')">FRONTAL / DORSAL</button>
        <button class="hud-btn" id="btnViewOblique" onclick="setViewPreset('oblique')">OBLIQUE 3D</button>
        <button class="hud-btn" id="btnViewLateral" onclick="setViewPreset('lateral')">SAGITTAL (SIDE)</button>
        <button class="hud-btn" id="btnViewTop" onclick="setViewPreset('top')">SUPERIOR (TOP)</button>
        <button class="hud-btn" id="btnTractsToggle" onclick="toggleTracts()">TRACT STREAMLINES: ON</button>
        <button class="hud-btn" id="btnRotateAuto" onclick="toggleAutoRotate()">AUTO ROTATE: OFF</button>
      </div>

      <div class="viewport-hud-coords">
        <div>MOUSE: Left-drag to Orbit · Right-drag to Pan · Scroll to Zoom</div>
        <div>AXES: +X (Right Eye / +300μm) · +Y (Dorsal / +250μm) · +Z (Posterior / +120μm)</div>
        <div id="cameraReadout">CAM: Yaw 0.0° · Pitch 0.0° · Dist 600μm</div>
      </div>

      <div id="tooltip"></div>
    </div>

    <aside id="controls">
      <!-- Section 1: Operating Mode -->
      <div class="panel-section">
        <div class="section-header">
          <span>EXPERIMENT MODE</span>
          <span style="font-size:9px; color:var(--text-muted);">AUDITABLE TELEMETRY</span>
        </div>
        <div class="mode-tabs">
          <div class="mode-tab active" id="tabReplay" onclick="setMode('replay')">COMBAT REPLAY</div>
          <div class="mode-tab" id="tabInteractive" onclick="setMode('interactive')">DIRECT STIMULATION</div>
        </div>

        <!-- Combat Replay Pane -->
        <div id="replayPane">
          <select class="select-dropdown" id="episodeSelector" onchange="onEpisodeChange()">
            <!-- populated by JS -->
          </select>

          <div class="scrubber-group">
            <div class="scrubber-labels">
              <span>STEP <strong id="stepLabel" style="color:var(--cyan);">000 / 150</strong></span>
              <span>TIME <strong id="timeLabel">0.00s</strong></span>
            </div>
            <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="150" value="0" oninput="onScrub(this.value)">
            <div class="playback-btns">
              <button class="play-btn" onclick="stepOffset(-1)">◀ STEP</button>
              <button class="play-btn" id="playPauseBtn" onclick="togglePlayPause()">▶ PLAY</button>
              <button class="play-btn" onclick="stepOffset(1)">STEP ▶</button>
            </div>
          </div>

          <div class="telemetry-card">
            <div class="telem-item">
              <span class="telem-label">ACTION</span>
              <span class="telem-val action" id="telemAction">FORWARD</span>
            </div>
            <div class="telem-item">
              <span class="telem-label">PLAYER HEALTH</span>
              <span class="telem-val" id="telemHealth">100 HP</span>
            </div>
            <div class="telem-item">
              <span class="telem-label">KILLS (PK / FF)</span>
              <span class="telem-val" id="telemKills">0 / 0</span>
            </div>
            <div class="telem-item">
              <span class="telem-label">DAMAGE DEALT</span>
              <span class="telem-val damage" id="telemDmg">0.0</span>
            </div>
            <div class="telem-item">
              <span class="telem-label">ASYMMETRY (Δ̂)</span>
              <span class="telem-val" id="telemAsym">0.000</span>
            </div>
            <div class="telem-item">
              <span class="telem-label">TARGET LOCK</span>
              <span class="telem-val lock" id="telemLock">NO TARGET</span>
            </div>
            <div class="telem-item" style="grid-column: span 2;">
              <span class="telem-label">T4 BILATERAL SOMATIC POTENTIAL</span>
              <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:2px;">
                <span>LEFT: <b id="telemT4L" style="color:var(--cyan);">-65.0 mV</b></span>
                <span>RIGHT: <b id="telemT4R" style="color:var(--amber);">-65.0 mV</b></span>
              </div>
            </div>
          </div>
        </div>

        <!-- Interactive Stimulation Pane -->
        <div id="interactivePane" style="display:none;">
          <div style="font-size:10px; color:var(--text-muted); margin-bottom:6px;">TRIGGER BIOLOGICAL PATHWAY EXCITATION:</div>
          <div class="stim-grid">
            <button class="stim-btn" onclick="triggerStim('turn_right')">▶ OPTIC SLIP (RIGHT TURN)</button>
            <button class="stim-btn" onclick="triggerStim('turn_left')">◀ OPTIC SLIP (LEFT TURN)</button>
            <button class="stim-btn" onclick="triggerStim('forward')">▲ FORWARD THRUST (EXPANSION)</button>
            <button class="stim-btn" onclick="triggerStim('door_use')">⎔ DOOR MECHANOSENSORY (USE)</button>
            <button class="stim-btn" onclick="triggerStim('combat_fire')">★ TARGET LOCK & FIRE BURST</button>
            <button class="stim-btn" onclick="triggerStim('threat_panic')">⚠ HEALTH THREAT / EVASION</button>
          </div>
          <div style="margin-top:12px; padding:10px; background:#08141b; border:1px solid var(--border); border-radius:6px;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px;">
              <span>Mi4 GABA SHUNTING</span>
              <button class="hud-btn" id="btnMi4Toggle" onclick="toggleMi4Lesion()" style="padding:4px 8px; font-size:10px;">INTACT</button>
            </div>
            <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">
              Disables GABAergic shunting in leading dendrites. Watch directional asymmetry collapse!
            </div>
          </div>
        </div>
      </div>

      <!-- Section 2: Compartment & Anatomical Inspector -->
      <div class="panel-section">
        <div class="section-header">
          <span>NEUROPIL INSPECTOR</span>
          <span style="font-size:9px; color:var(--cyan);" id="inspStatus">HOVER / CLICK TO INSPECT</span>
        </div>
        <div class="inspector-card">
          <div class="insp-row">
            <span class="insp-label">NAME</span>
            <span class="insp-val" id="inspName">Lobula Plate (Right)</span>
          </div>
          <div class="insp-row">
            <span class="insp-label">SYSTEM</span>
            <span class="insp-val" id="inspSystem">Optic Lobe (Motion)</span>
          </div>
          <div class="insp-row">
            <span class="insp-label">CELL TYPES</span>
            <span class="insp-val" id="inspCells">T4a-d, T5a-d, LPTC-HS/VS</span>
          </div>
          <div class="insp-row">
            <span class="insp-label">TRANSMITTERS</span>
            <span class="insp-val" id="inspTransmitters">ACh, GABA</span>
          </div>
          <div class="insp-row">
            <span class="insp-label">ACTIVE VOLTAGE</span>
            <span class="insp-val" id="inspVoltage">-65.0 mV</span>
          </div>
          <div class="insp-desc" id="inspDesc">
            Terminus of direction-selective T4/T5 motion columns. Synapses onto giant Lobula Plate Tangential Cells (LPTCs) for wide-field optomotor yaw and pitch stabilization.
          </div>
        </div>
      </div>

      <!-- Section 3: Color Legend & Biological Grounding -->
      <div class="panel-section">
        <div class="section-header">
          <span>BIOLUMINESCENT SPECTRUM</span>
        </div>
        <div class="legend-list">
          <div class="legend-item">
            <div class="legend-dot" style="background:var(--cyan); box-shadow:0 0 6px var(--cyan);"></div>
            <span>Resting Sensory Potentials (-65 mV baseline, Retino-Lamina)</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot" style="background:var(--amber); box-shadow:0 0 6px var(--amber);"></div>
            <span>Suprathreshold Active Depolarization (Motion Asymmetry & Turning)</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot" style="background:#ff3300; box-shadow:0 0 6px #ff3300;"></div>
            <span>Combat Firing Command (Lobula Feature Detectors & VNC Motor Discharge)</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot" style="background:var(--green); box-shadow:0 0 6px var(--green);"></div>
            <span>Mechanosensory Contact (SEZ Tactile & Prothoracic T1 Door Push)</span>
          </div>
          <div class="legend-item">
            <div class="legend-dot" style="background:var(--red); box-shadow:0 0 6px var(--red);"></div>
            <span>Threat Nociception / Health Panic (Mushroom Body & Lateral Horn)</span>
          </div>
        </div>
      </div>
    </aside>
  </div>

  <script>
    // Embedded Data Payload
    const CNS_DATA = {data_json};

    // Application State
    const state = {{
      mode: "replay", // 'replay' | 'interactive'
      currentEpisode: "ModelD_ActiveTree_Saccade_3",
      currentStep: 0,
      isPlaying: false,
      playbackTimer: null,
      showTracts: true,
      autoRotate: false,
      mi4LesionActive: false,

      // Camera State
      camYaw: 0.0,
      camPitch: 0.0,
      camDist: 640.0,
      camTarget: [0, -50, 0],
      isDragging: false,
      isPanning: false,
      lastMouseX: 0,
      lastMouseY: 0,

      // Hover/Inspect
      hoveredNode: null,
      selectedCompartment: "LP_R",

      // Interactive Stimulation State
      stimType: null,
      stimTimer: 0,
      stimAsym: 0.0,
      stimAction: "NOOP"
    }};

    // DOM Elements
    const canvas = document.getElementById("cnsCanvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");

    // Initialize Viewport
    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}
    window.addEventListener("resize", resizeCanvas);

    // Populate Episode Selector
    function initEpisodes() {{
      const selector = document.getElementById("episodeSelector");
      selector.innerHTML = "";
      const epKeys = Object.keys(CNS_DATA.episodes);
      epKeys.forEach(key => {{
        const ep = CNS_DATA.episodes[key];
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = `${{key}} (${{ep.player_kills}} Kills · ${{ep.health_remaining}} HP)`;
        selector.appendChild(opt);
      }});
      if (epKeys.length > 0) {{
        state.currentEpisode = epKeys[0];
        selector.value = state.currentEpisode;
      }}
    }}

    // 3D Projection Math
    function project3D(x, y, z, width, height) {{
      // Translate to origin
      let tx = x - state.camTarget[0];
      let ty = y - state.camTarget[1];
      let tz = z - state.camTarget[2];

      // Rotate Yaw (around Y axis)
      const cosY = Math.cos(state.camYaw);
      const sinY = Math.sin(state.camYaw);
      let x1 = tx * cosY - tz * sinY;
      let z1 = tx * sinY + tz * cosY;

      // Rotate Pitch (around X axis)
      const cosP = Math.cos(state.camPitch);
      const sinP = Math.sin(state.camPitch);
      let y2 = ty * cosP - z1 * sinP;
      let z2 = ty * sinP + z1 * cosP;

      // Perspective scale
      const focal = 580.0;
      const eyeZ = z2 + state.camDist;
      if (eyeZ <= 10.0) return null; // Behind camera

      const scale = focal / eyeZ;
      const screenX = width / 2 + x1 * scale;
      const screenY = height / 2 - y2 * scale; // Invert Y for graphics screen coords

      return {{ screenX, screenY, scale, eyeZ }};
    }}

    // Camera Presets
    function setViewPreset(preset) {{
      document.querySelectorAll(".viewport-hud-top .hud-btn").forEach(b => {{
        if (b.id.startsWith("btnView")) b.classList.remove("active");
      }});

      if (preset === "dorsal") {{
        state.camYaw = 0.0;
        state.camPitch = 0.0;
        state.camDist = 640.0;
        document.getElementById("btnViewDorsal").classList.add("active");
      }} else if (preset === "oblique") {{
        state.camYaw = 0.55;
        state.camPitch = 0.35;
        state.camDist = 680.0;
        document.getElementById("btnViewOblique").classList.add("active");
      }} else if (preset === "lateral") {{
        state.camYaw = Math.PI / 2;
        state.camPitch = 0.0;
        state.camDist = 660.0;
        document.getElementById("btnViewLateral").classList.add("active");
      }} else if (preset === "top") {{
        state.camYaw = 0.0;
        state.camPitch = Math.PI / 2 - 0.02;
        state.camDist = 640.0;
        document.getElementById("btnViewTop").classList.add("active");
      }}
      updateCameraReadout();
    }}

    function toggleTracts() {{
      state.showTracts = !state.showTracts;
      const btn = document.getElementById("btnTractsToggle");
      btn.textContent = `TRACT STREAMLINES: ${{state.showTracts ? "ON" : "OFF"}}`;
      btn.classList.toggle("active", state.showTracts);
    }}

    function toggleAutoRotate() {{
      state.autoRotate = !state.autoRotate;
      const btn = document.getElementById("btnRotateAuto");
      btn.textContent = `AUTO ROTATE: ${{state.autoRotate ? "ON" : "OFF"}}`;
      btn.classList.toggle("active", state.autoRotate);
    }}

    function updateCameraReadout() {{
      const yawDeg = ((state.camYaw * 180 / Math.PI) % 360).toFixed(1);
      const pitchDeg = ((state.camPitch * 180 / Math.PI) % 360).toFixed(1);
      document.getElementById("cameraReadout").textContent = 
        `CAM: Yaw ${{yawDeg}}° · Pitch ${{pitchDeg}}° · Dist ${{state.camDist.toFixed(0)}}μm`;
    }}

    // Mouse / Touch Interaction
    canvas.addEventListener("mousedown", e => {{
      state.isDragging = (e.button === 0);
      state.isPanning = (e.button === 2 || e.button === 1);
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;
    }});

    window.addEventListener("mouseup", () => {{
      state.isDragging = false;
      state.isPanning = false;
    }});

    window.addEventListener("mousemove", e => {{
      const dx = e.clientX - state.lastMouseX;
      const dy = e.clientY - state.lastMouseY;
      state.lastMouseX = e.clientX;
      state.lastMouseY = e.clientY;

      if (state.isDragging) {{
        state.camYaw += dx * 0.008;
        state.camPitch = Math.max(-1.4, Math.min(1.4, state.camPitch + dy * 0.008));
        updateCameraReadout();
      }} else if (state.isPanning) {{
        state.camTarget[0] -= dx * 0.6;
        state.camTarget[1] += dy * 0.6;
      }} else {{
        checkHover(e.clientX, e.clientY);
      }}
    }});

    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      state.camDist = Math.max(200.0, Math.min(1400.0, state.camDist + e.deltaY * 0.8));
      updateCameraReadout();
    }}, {{ passive: false }});

    canvas.addEventListener("contextmenu", e => e.preventDefault());

    // Hover Inspection
    function checkHover(clientX, clientY) {{
      const rect = canvas.getBoundingClientRect();
      const mouseX = clientX - rect.left;
      const mouseY = clientY - rect.top;

      let closest = null;
      let minD = 18.0;

      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;

      for (let i = 0; i < CNS_DATA.anatomy.nodes.length; i += 2) {{
        const n = CNS_DATA.anatomy.nodes[i];
        const p = project3D(n.x, n.y, n.z, w, h);
        if (!p) continue;
        const d = Math.hypot(p.screenX - mouseX, p.screenY - mouseY);
        if (d < minD) {{
          minD = d;
          closest = n;
        }}
      }}

      state.hoveredNode = closest;
      if (closest) {{
        tooltip.style.display = "block";
        tooltip.style.left = `${{clientX + 14}}px`;
        tooltip.style.top = `${{clientY + 14}}px`;
        tooltip.innerHTML = `<strong>${{closest.display_name}}</strong><br/><span style="color:var(--text-muted); font-size:10px;">${{closest.role}}</span>`;
        updateInspector(closest);
      }} else {{
        tooltip.style.display = "none";
      }}
    }}

    function updateInspector(node) {{
      document.getElementById("inspName").textContent = node.display_name;
      document.getElementById("inspSystem").textContent = node.system.replace("_", " ").toUpperCase();
      document.getElementById("inspCells").textContent = node.cell_types.join(", ");
      document.getElementById("inspTransmitters").textContent = node.transmitters.join(", ");
      document.getElementById("inspDesc").textContent = node.role;
    }}

    // Playback Logic
    function setMode(mode) {{
      state.mode = mode;
      document.getElementById("tabReplay").classList.toggle("active", mode === "replay");
      document.getElementById("tabInteractive").classList.toggle("active", mode === "interactive");
      document.getElementById("replayPane").style.display = (mode === "replay") ? "block" : "none";
      document.getElementById("interactivePane").style.display = (mode === "interactive") ? "block" : "none";
    }}

    function onEpisodeChange() {{
      state.currentEpisode = document.getElementById("episodeSelector").value;
      state.currentStep = 0;
      updateUI();
    }}

    function onScrub(step) {{
      state.currentStep = parseInt(step, 10);
      updateUI();
    }}

    function stepOffset(delta) {{
      const ep = CNS_DATA.episodes[state.currentEpisode];
      if (!ep) return;
      state.currentStep = Math.max(0, Math.min(ep.ticks.length - 1, state.currentStep + delta));
      updateUI();
    }}

    function togglePlayPause() {{
      state.isPlaying = !state.isPlaying;
      const btn = document.getElementById("playPauseBtn");
      btn.textContent = state.isPlaying ? "❚❚ PAUSE" : "▶ PLAY";
      btn.classList.toggle("playing", state.isPlaying);

      if (state.isPlaying) {{
        if (state.playbackTimer) clearInterval(state.playbackTimer);
        state.playbackTimer = setInterval(() => {{
          const ep = CNS_DATA.episodes[state.currentEpisode];
          if (!ep) return;
          if (state.currentStep >= ep.ticks.length - 1) {{
            state.currentStep = 0;
          }} else {{
            state.currentStep++;
          }}
          updateUI();
        }}, 100);
      }} else if (state.playbackTimer) {{
        clearInterval(state.playbackTimer);
        state.playbackTimer = null;
      }}
    }}

    // Interactive Stimulation Triggers
    function triggerStim(type) {{
      state.stimType = type;
      state.stimTimer = 1.0; // 1.0 down to 0 over time
    }}

    function toggleMi4Lesion() {{
      state.mi4LesionActive = !state.mi4LesionActive;
      const btn = document.getElementById("btnMi4Toggle");
      btn.textContent = state.mi4LesionActive ? "KO (LESIONED)" : "INTACT";
      btn.style.color = state.mi4LesionActive ? "var(--amber)" : "var(--cyan)";
      btn.style.borderColor = state.mi4LesionActive ? "var(--amber)" : "var(--border)";
    }}

    // Update UI Elements
    function updateUI() {{
      const ep = CNS_DATA.episodes[state.currentEpisode];
      if (!ep || !ep.ticks[state.currentStep]) return;
      const tick = ep.ticks[state.currentStep];

      document.getElementById("stepLabel").textContent = `${{String(state.currentStep).padStart(3, "0")}} / ${{ep.ticks.length - 1}}`;
      document.getElementById("timeLabel").textContent = `${{(state.currentStep * 0.1).toFixed(2)}}s`;
      document.getElementById("timelineSlider").value = state.currentStep;
      document.getElementById("timelineSlider").max = ep.ticks.length - 1;

      document.getElementById("telemAction").textContent = tick.action;
      document.getElementById("telemHealth").textContent = `${{Math.round(tick.health)}} HP`;
      document.getElementById("telemKills").textContent = `${{tick.player_kills}} / ${{tick.friendly_fire_kills}}`;
      document.getElementById("telemDmg").textContent = `${{Math.round(tick.damage_dealt)}}`;
      document.getElementById("telemAsym").textContent = `${{tick.norm_asymmetry.toFixed(3)}}`;

      const lock = (tick.firing_solution_locked === 1.0);
      document.getElementById("telemLock").textContent = lock ? "LOCKED (≤18°)" : (tick.combat_active ? "ACQUIRING..." : "NO ENEMY");
      document.getElementById("telemLock").style.color = lock ? "var(--green)" : (tick.combat_active ? "var(--amber)" : "var(--text-muted)");

      document.getElementById("telemT4L").textContent = `${{tick.t4_l_v.toFixed(1)}} mV`;
      document.getElementById("telemT4R").textContent = `${{tick.t4_r_v.toFixed(1)}} mV`;
      document.getElementById("modelHeader").textContent = state.currentEpisode.replace("ModelD_", "").replace("_Saccade_3", "");
    }}

    // Animation & Rendering Loop
    let particlePhase = 0;

    function render() {{
      requestAnimationFrame(render);

      if (state.autoRotate) {{
        state.camYaw += 0.005;
        updateCameraReadout();
      }}

      // Dynamic activity values
      let normAsym = 0.0;
      let action = "FORWARD";
      let isCombat = false;
      let isFire = false;
      let isDoor = false;
      let isDamage = false;
      let t4L = -65.0;
      let t4R = -65.0;

      if (state.mode === "replay") {{
        const ep = CNS_DATA.episodes[state.currentEpisode];
        if (ep && ep.ticks[state.currentStep]) {{
          const t = ep.ticks[state.currentStep];
          normAsym = t.norm_asymmetry || 0.0;
          action = t.action;
          isCombat = (t.combat_active === 1.0);
          isFire = (t.action === "FIRE" || t.firing_solution_locked === 1.0);
          isDoor = (t.action === "USE" || t.door_candidate === 1.0);
          isDamage = (t.health_priority_active === 1.0);
          t4L = t.t4_l_v || -65.0;
          t4R = t.t4_r_v || -65.0;
        }}
      }} else if (state.mode === "interactive") {{
        if (state.stimTimer > 0) {{
          state.stimTimer -= 0.015;
          const decay = Math.max(0, state.stimTimer);
          if (state.stimType === "turn_right") {{
            normAsym = state.mi4LesionActive ? 0.0 : 0.85 * decay;
            action = "TURN_RIGHT";
          }} else if (state.stimType === "turn_left") {{
            normAsym = state.mi4LesionActive ? 0.0 : -0.85 * decay;
            action = "TURN_LEFT";
          }} else if (state.stimType === "combat_fire") {{
            isFire = true;
            action = "FIRE";
          }} else if (state.stimType === "door_use") {{
            isDoor = true;
            action = "USE";
          }} else if (state.stimType === "threat_panic") {{
            isDamage = true;
          }}
        }}
      }}

      particlePhase += 0.035;

      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;

      ctx.clearRect(0, 0, w, h);

      // 1. Draw Connectome Tract Streamlines
      if (state.showTracts) {{
        ctx.lineWidth = 1.0;
        for (let i = 0; i < CNS_DATA.anatomy.tracts.length; i++) {{
          const tr = CNS_DATA.anatomy.tracts[i];
          const sNode = CNS_DATA.anatomy.nodes[tr.source];
          const tNode = CNS_DATA.anatomy.nodes[tr.target];
          const p1 = project3D(sNode.x, sNode.y, sNode.z, w, h);
          const p2 = project3D(tNode.x, tNode.y, tNode.z, w, h);
          if (!p1 || !p2) continue;

          // Tract activity excitation
          let tractAlpha = 0.08;
          let tractColor = "rgba(0, 229, 255, 0.12)";

          if (isFire && tr.type.includes("descending")) {{
            tractAlpha = 0.8;
            tractColor = "rgba(255, 80, 0, 0.7)";
          }} else if (isDoor && tr.type.includes("tactile")) {{
            tractAlpha = 0.8;
            tractColor = "rgba(0, 255, 200, 0.7)";
          }} else if (normAsym > 0.15 && (sNode.x > 0 || tNode.x > 0)) {{
            tractAlpha = 0.5;
            tractColor = "rgba(255, 160, 0, 0.5)";
          }} else if (normAsym < -0.15 && (sNode.x < 0 || tNode.x < 0)) {{
            tractAlpha = 0.5;
            tractColor = "rgba(255, 160, 0, 0.5)";
          }}

          ctx.strokeStyle = tractColor;
          ctx.beginPath();
          ctx.moveTo(p1.screenX, p1.screenY);
          ctx.lineTo(p2.screenX, p2.screenY);
          ctx.stroke();

          // Traveling Action Potential Photon
          const tOffset = (particlePhase + i * 0.12) % 1.0;
          const px = p1.screenX + (p2.screenX - p1.screenX) * tOffset;
          const py = p1.screenY + (p2.screenY - p1.screenY) * tOffset;
          const pRadius = 1.2 * p1.scale;

          ctx.fillStyle = (tractAlpha > 0.3) ? "#ffb300" : "#58dfc2";
          ctx.beginPath();
          ctx.arc(px, py, pRadius, 0, 2 * Math.PI);
          ctx.fill();
        }}
      }}

      // 2. Depth Sort Anatomical Nodes
      const projectedNodes = [];
      for (let i = 0; i < CNS_DATA.anatomy.nodes.length; i++) {{
        const n = CNS_DATA.anatomy.nodes[i];
        const p = project3D(n.x, n.y, n.z, w, h);
        if (!p) continue;
        projectedNodes.push({{ node: n, p: p }});
      }}
      projectedNodes.sort((a, b) => b.p.eyeZ - a.p.eyeZ);

      // 3. Render Biological Nodes
      for (let i = 0; i < projectedNodes.length; i++) {{
        const item = projectedNodes[i];
        const n = item.node;
        const p = item.p;

        // Biological Excitation Transfer Function
        let excitation = 0.0;

        // Unilateral Optic Flow Excitation (HS / T4 / Medulla / Lobula Plate)
        if (n.system === "optic_lobe" || n.system === "visual_sensory") {{
          if (normAsym > 0.15 && n.x > 0) {{
            excitation = Math.min(1.0, Math.abs(normAsym) * 1.2);
          }} else if (normAsym < -0.15 && n.x < 0) {{
            excitation = Math.min(1.0, Math.abs(normAsym) * 1.2);
          }}
        }}

        // Central Complex Excitation (Heading Compass & Steering)
        if (n.system === "central_complex") {{
          excitation = 0.3 + 0.7 * Math.abs(normAsym);
        }}

        // Weapon Fire Shockwave (Lobula & Metathoracic VNC)
        if (isFire && (n.compartment === "VNC_T3" || n.compartment === "VNC_T2" || n.system === "descending_motor" || n.compartment.startsWith("LO_"))) {{
          excitation = 1.0;
        }}

        // Mechanosensory Door Interaction (SEZ & Prothoracic VNC_T1)
        if (isDoor && (n.compartment === "VNC_T1" || n.compartment === "SEZ" || n.compartment.startsWith("AL_"))) {{
          excitation = 0.95;
        }}

        // Threat Shock (Mushroom Body & Lateral Horn)
        if (isDamage && (n.compartment.startsWith("MB_") || n.system === "learning_context")) {{
          excitation = 1.0;
        }}

        // Compute Color Interpolation
        let r, g, b;
        if (excitation > 0.05) {{
          const t = Math.min(1.0, excitation);
          r = Math.round(n.base_color[0] + (n.active_color[0] - n.base_color[0]) * t);
          g = Math.round(n.base_color[1] + (n.active_color[1] - n.base_color[1]) * t);
          b = Math.round(n.base_color[2] + (n.active_color[2] - n.base_color[2]) * t);
        }} else {{
          r = n.base_color[0];
          g = n.base_color[1];
          b = n.base_color[2];
        }}

        const alpha = Math.min(1.0, Math.max(0.3, 0.4 + 0.6 * excitation));
        const radius = Math.max(1.0, (1.8 + 2.2 * excitation) * p.scale);

        // Core Point
        ctx.fillStyle = `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
        ctx.beginPath();
        ctx.arc(p.screenX, p.screenY, radius, 0, 2 * Math.PI);
        ctx.fill();

        // Excitation Bloom Halo
        if (excitation > 0.35) {{
          ctx.fillStyle = `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha * 0.35}})`;
          ctx.beginPath();
          ctx.arc(p.screenX, p.screenY, radius * 2.8, 0, 2 * Math.PI);
          ctx.fill();
        }}
      }}
    }}

    // Initialization
    window.addEventListener("DOMContentLoaded", () => {{
      resizeCanvas();
      initEpisodes();
      updateUI();
      setViewPreset("dorsal");
      render();
    }});
  </script>
</body>
</html>
"""
    return html


def main():
    data_path = Path("web/fly_cns_data.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    html = generate_html_content(data)

    # 1. Standalone Artifact for chat embed & review
    artifact_path = Path("/Users/ljp176/.gemini/antigravity/brain/fd394349-d70b-4c24-b0d6-2c95c6d775d7/fly_nervous_system.html")
    with open(artifact_path, "w") as f:
        f.write(html)
    print(f"Wrote artifact to {artifact_path} ({artifact_path.stat().st_size / 1024:.1f} KB)")

    # 2. Permanent Observatory App file in web/
    web_path = Path("web/fly_nervous_system.html")
    with open(web_path, "w") as f:
        f.write(html)
    print(f"Wrote web app to {web_path} ({web_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
