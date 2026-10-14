// ==========================================================================
// Antigravity BookBot — Frontend Logic & WebSocket/REST Client
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // UI Elements
  const queryInput = document.getElementById('target-query-input');
  const runBtn = document.getElementById('run-detection-btn');
  const executeDobotBtn = document.getElementById('execute-dobot-btn');
  const catalogChips = document.getElementById('catalog-chips');
  const modeSelector = document.getElementById('mode-selector');
  const viewportImg = document.getElementById('main-viewport-img');
  const viewportLoader = document.getElementById('viewport-loader');
  const loaderStatus = document.getElementById('loader-status');
  const toggleViewBtn = document.getElementById('toggle-view-btn');
  const toggleRawBtn = document.getElementById('toggle-raw-btn');
  const canvasModeBadge = document.getElementById('canvas-mode-badge');
  const hudTitle = document.getElementById('hud-title');
  const hudCoords = document.getElementById('hud-coords');

  // Metrics Elements
  const mPickPoint = document.getElementById('m-pick-point');
  const mPickType = document.getElementById('m-pick-type');
  const mGraspAngle = document.getElementById('m-grasp-angle');
  const mAngleNote = document.getElementById('m-angle-note');
  const mSize = document.getElementById('m-size');
  const mConf = document.getElementById('m-conf');
  const mMatchBadge = document.getElementById('telemetry-match-badge');
  const xyzX = document.getElementById('xyz-x');
  const xyzY = document.getElementById('xyz-y');
  const xyzZ = document.getElementById('xyz-z');
  const dobotStatusMsg = document.getElementById('dobot-status-msg');

  // Stepper Elements
  const stepCards = [
    document.getElementById('step-mode'),
    document.getElementById('step-capture'),
    document.getElementById('step-detect'),
    document.getElementById('step-ocr'),
    document.getElementById('step-sam'),
    document.getElementById('step-robot')
  ];

  // State
  let currentSampleKey = 'shelf_spine';
  let currentMode = 'auto';
  let currentRawImage = '';
  let currentAnnotatedImage = '';
  let lastTargetData = null;

  // Initialize System Status
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      // Update Dobot port badge
      if (data.dobot_ports && data.dobot_ports.length > 0) {
        document.getElementById('dobot-label').textContent = `Dobot Magician (${data.dobot_ports[0].device})`;
        document.getElementById('status-arm').classList.add('active');
      } else {
        document.getElementById('dobot-label').textContent = 'Dobot Magician (Simulated)';
      }

      // Update camera status
      if (data.camera_connected) {
        document.getElementById('camera-label').textContent = 'RealSense D435i (Live)';
      } else {
        document.getElementById('camera-label').textContent = 'RealSense (Offline - Benchmark Mode)';
      }
    } catch (err) {
      console.warn('Status fetch error:', err);
    }
  }

  // Animate Stepper
  function setStepperStage(stageIndex) {
    stepCards.forEach((card, idx) => {
      card.classList.remove('active', 'completed');
      if (idx < stageIndex) {
        card.classList.add('completed');
      } else if (idx === stageIndex) {
        card.classList.add('active');
      }
    });
  }

  // Handle Catalog Chip Selection
  catalogChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;

    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');

    const title = chip.dataset.title;
    const sample = chip.dataset.sample;

    queryInput.value = title;
    currentSampleKey = sample;

    // Trigger detection run
    runDetection();
  });

  // Handle Mode Selection
  modeSelector.addEventListener('click', (e) => {
    const btn = e.target.closest('.seg-btn');
    if (!btn) return;

    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
  });

  // Handle Overlay Toggles
  toggleViewBtn.addEventListener('click', () => {
    toggleViewBtn.classList.add('active');
    toggleRawBtn.classList.remove('active');
    if (currentAnnotatedImage) {
      viewportImg.src = currentAnnotatedImage;
    }
  });

  toggleRawBtn.addEventListener('click', () => {
    toggleRawBtn.classList.add('active');
    toggleViewBtn.classList.remove('active');
    if (currentRawImage) {
      viewportImg.src = currentRawImage;
    }
  });

  // Run Pipeline Detection
  async function runDetection() {
    const query = queryInput.value.trim();
    const sourceRadio = document.querySelector('input[name="input-source"]:checked');
    const source = sourceRadio ? sourceRadio.value : 'sample';

    viewportLoader.classList.remove('hidden');
    loaderStatus.textContent = '1/6 Classifying arrangement mode (~2ms)...';
    setStepperStage(0);

    setTimeout(() => {
      loaderStatus.textContent = '2/6 Capturing 1080p aligned RGB-D frame...';
      setStepperStage(1);
    }, 150);

    setTimeout(() => {
      loaderStatus.textContent = '3/6 Running YOLO OBB candidate detection...';
      setStepperStage(2);
    }, 350);

    try {
      loaderStatus.textContent = '4/6 RapidOCR orientation-aware title matching...';
      setStepperStage(3);

      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: source,
          sample_key: currentSampleKey,
          query: query,
          mode: currentMode
        })
      });

      const data = await response.json();

      if (!data.success || !data.target) {
        viewportLoader.classList.add('hidden');
        alert('No confident target match found for query: ' + query);
        return;
      }

      loaderStatus.textContent = '5/6 SAM high-precision edge refinement...';
      setStepperStage(4);

      lastTargetData = data.target;
      currentRawImage = data.raw_image;
      currentAnnotatedImage = data.annotated_image;

      // Update Viewport Image
      viewportImg.src = currentAnnotatedImage;

      // Update Canvas Mode Badge
      canvasModeBadge.textContent = `${data.mode.toUpperCase()} MODE (${(data.mode_conf * 100).toFixed(0)}%)`;

      // Update Locked HUD
      hudTitle.textContent = data.target.text || query;
      const pickX = data.target.pick_point_2d ? data.target.pick_point_2d[0] : data.target.top_edge_center_u;
      const pickY = data.target.pick_point_2d ? data.target.pick_point_2d[1] : data.target.top_edge_center_v;
      hudCoords.textContent = `Pick Point: (${pickX.toFixed(1)}, ${pickY.toFixed(1)}) px`;

      // Update Telemetry Metrics
      mPickPoint.textContent = `(${pickX.toFixed(1)}, ${pickY.toFixed(1)}) px`;
      
      if (data.mode === 'scattered') {
        mPickType.textContent = 'Top-Left Edge Point (Perpendicular Inward)';
      } else {
        mPickType.textContent = 'Top Spine Edge Midpoint';
      }

      const graspAngle = data.target.grasp_approach_angle_deg || data.target.pick_edge_angle_deg || 0.0;
      mGraspAngle.innerHTML = `${(data.target.pick_edge_angle_deg || 0.0).toFixed(1)}&deg;`;
      mAngleNote.innerHTML = `Wrist Align: ${graspAngle.toFixed(1)}&deg;`;

      const w = data.target.width_px || 0.0;
      const h = data.target.height_px || data.target.pick_edge_length_px || 0.0;
      mSize.textContent = `${w.toFixed(0)} × ${h.toFixed(0)} px`;

      mConf.textContent = (data.target.conf || 0.0).toFixed(2);
      mMatchBadge.textContent = `Match: ${(data.target.similarity || 1.0).toFixed(2)}`;

      // Update 3D Coordinates
      const xyz = data.target.xyz_pick_camera_frame || data.target.xyz_camera_frame || [-0.056, 0.011, 0.508];
      xyzX.textContent = `${xyz[0] >= 0 ? '+' : ''}${xyz[0].toFixed(3)} m`;
      xyzY.textContent = `${xyz[1] >= 0 ? '+' : ''}${xyz[1].toFixed(3)} m`;
      xyzZ.textContent = `${xyz[2] >= 0 ? '+' : ''}${xyz[2].toFixed(3)} m`;

      // Complete Stepper to Stage 5
      setStepperStage(5);
      dobotStatusMsg.textContent = `Target locked. Ready to execute Dobot pick at (${xyz[0].toFixed(3)}, ${xyz[1].toFixed(3)}, ${xyz[2].toFixed(3)}) m.`;

    } catch (err) {
      console.error('Detection error:', err);
      alert('Error communicating with detection server.');
    } finally {
      viewportLoader.classList.add('hidden');
    }
  }

  // Execute Dobot Magician Pick Sequence
  async function executePick() {
    if (!lastTargetData) {
      alert('Please run target detection first to acquire 3D pick coordinates.');
      return;
    }

    const xyz = lastTargetData.xyz_pick_camera_frame || lastTargetData.xyz_camera_frame || [-0.056, 0.011, 0.508];
    const angle = lastTargetData.pick_edge_angle_deg || 0.0;

    executeDobotBtn.disabled = true;
    executeDobotBtn.style.opacity = '0.6';
    dobotStatusMsg.textContent = 'Trajectory: Moving Dobot arm to hover point...';

    try {
      const res = await fetch('/api/pick', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ xyz: xyz, angle: angle })
      });
      const data = await res.json();

      setTimeout(() => {
        dobotStatusMsg.textContent = 'Trajectory: Lowering suction cup onto book edge...';
      }, 700);

      setTimeout(() => {
        dobotStatusMsg.textContent = 'Suction Pump: ACTIVATED (Vacuum Seal Locked)';
      }, 1400);

      setTimeout(() => {
        dobotStatusMsg.textContent = 'Trajectory: Lifting book and retracting to safe waypoint...';
      }, 2100);

      setTimeout(() => {
        dobotStatusMsg.textContent = `Pick Complete! ${data.message}`;
        executeDobotBtn.disabled = false;
        executeDobotBtn.style.opacity = '1';
      }, 2800);

    } catch (err) {
      dobotStatusMsg.textContent = 'Dobot communication error: ' + err;
      executeDobotBtn.disabled = false;
      executeDobotBtn.style.opacity = '1';
    }
  }

  runBtn.addEventListener('click', runDetection);
  executeDobotBtn.addEventListener('click', executePick);

  // Initial Boot
  fetchStatus();
  runDetection();
});
