/**
 * ============================================================================
 * OPERATOR UI — CLIENT LOGIC & REST INTERFACE
 * ============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
  // UI Elements
  const queryInput = document.getElementById('query-input');
  const findBtn = document.getElementById('find-target-btn');
  const findSpinner = document.getElementById('find-spinner');
  const findBtnText = document.getElementById('find-btn-text');

  const sendArmBtn = document.getElementById('send-arm-btn');
  const armSpinner = document.getElementById('arm-spinner');
  const armBtnText = document.getElementById('arm-btn-text');

  const modeTabs = document.querySelectorAll('.mode-tab');
  const catalogChips = document.querySelectorAll('.chip');
  const headerActiveMode = document.getElementById('header-active-mode');

  // Telemetry Elements
  const telemetryBadge = document.getElementById('telemetry-badge');
  const tTitle = document.getElementById('t-title');
  const tMode = document.getElementById('t-mode');
  const tConf = document.getElementById('t-conf');
  const tSim = document.getElementById('t-sim');
  const camX = document.getElementById('cam-x');
  const camY = document.getElementById('cam-y');
  const camZ = document.getElementById('cam-z');
  const robX = document.getElementById('rob-x');
  const robY = document.getElementById('rob-y');
  const robZ = document.getElementById('rob-z');
  const tPickNote = document.getElementById('t-pick-note');
  const streamLockBadge = document.getElementById('stream-lock-badge');
  const streamBannerText = document.getElementById('stream-banner-text');

  const motionProgress = document.getElementById('motion-progress');
  const motionStatusText = document.getElementById('motion-status-text');

  // State
  let currentMode = 'auto';
  let lastFoundTarget = null;

  // Handle Mode Tabs
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modeTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentMode = tab.dataset.mode;
      headerActiveMode.textContent = currentMode.toUpperCase() + (currentMode === 'auto' ? ' ADAPTIVE' : '');
    });
  });

  // Handle Preset Catalog Chips
  catalogChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.dataset.query;
      const mode = chip.dataset.mode;
      queryInput.value = query;

      // Match mode tab
      modeTabs.forEach(tab => {
        if (tab.dataset.mode === mode) {
          tab.click();
        }
      });

      // Quick trigger find
      triggerFind();
    });
  });

  // Scan & Locate Target
  findBtn.addEventListener('click', triggerFind);

  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      triggerFind();
    }
  });

  async function triggerFind() {
    const query = queryInput.value.trim();
    if (!query) {
      queryInput.focus();
      return;
    }

    // Set Loading UI
    findBtn.disabled = true;
    findSpinner.style.display = 'inline-block';
    findBtnText.textContent = 'SCANNING & REFINING...';
    telemetryBadge.className = 'status-badge standby';
    telemetryBadge.textContent = 'SCANNING';
    motionProgress.style.width = '20%';
    motionStatusText.textContent = `Analyzing optical frame for: "${query}"...`;

    try {
      const response = await fetch('/api/find', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, mode: currentMode })
      });

      const data = await response.json();

      if (data.found) {
        lastFoundTarget = data;

        // Update Telemetry Panel
        telemetryBadge.className = 'status-badge locked';
        telemetryBadge.textContent = 'TARGET LOCKED';
        streamLockBadge.textContent = 'LOCKED';
        streamLockBadge.style.color = 'var(--accent-green)';
        streamBannerText.textContent = `TARGET LOCKED: "${data.title}" | DOBOT: (${data.robot_xyz[0]}, ${data.robot_xyz[1]}, ${data.robot_xyz[2]}) mm`;

        tTitle.textContent = data.title;
        tMode.textContent = (data.mode || 'auto').toUpperCase();
        tConf.textContent = `${Math.round((data.conf || 0.9) * 100)}%`;
        tSim.textContent = `${Math.round((data.sim || 1.0) * 100)}%`;

        if (data.cam_xyz && data.cam_xyz.length >= 3) {
          camX.textContent = data.cam_xyz[0].toFixed(3);
          camY.textContent = data.cam_xyz[1].toFixed(3);
          camZ.textContent = data.cam_xyz[2].toFixed(3);
        }

        if (data.robot_xyz && data.robot_xyz.length >= 3) {
          robX.textContent = data.robot_xyz[0].toFixed(1);
          robY.textContent = data.robot_xyz[1].toFixed(1);
          robZ.textContent = data.robot_xyz[2].toFixed(1);
        }

        tPickNote.textContent = data.pick_note || 'Upper spine, parallel grasp';

        // Enable Arm Execution
        sendArmBtn.disabled = false;
        motionProgress.style.width = '100%';
        motionStatusText.textContent = `Target locked. Dobot ready at [X=${data.robot_xyz[0]}, Y=${data.robot_xyz[1]}, Z=${data.robot_xyz[2]}] mm.`;

      } else {
        lastFoundTarget = null;
        sendArmBtn.disabled = true;
        telemetryBadge.className = 'status-badge error';
        telemetryBadge.textContent = 'NOT FOUND';
        streamLockBadge.textContent = 'NO MATCH';
        streamLockBadge.style.color = 'var(--accent-red)';
        streamBannerText.textContent = `No book found matching: "${query}". Check title spelling or camera view.`;

        tTitle.textContent = 'None Found';
        tMode.textContent = '\u2014';
        tConf.textContent = '\u2014';
        tSim.textContent = '\u2014';
        camX.textContent = '0.000';
        camY.textContent = '0.000';
        camZ.textContent = '0.000';
        robX.textContent = '0.0';
        robY.textContent = '0.0';
        robZ.textContent = '0.0';
        tPickNote.textContent = 'No target detected.';

        motionProgress.style.width = '0%';
        motionStatusText.textContent = `No book candidates matched query "${query}".`;
      }
    } catch (err) {
      console.error('API Find error:', err);
      telemetryBadge.className = 'status-badge error';
      telemetryBadge.textContent = 'API ERROR';
      motionStatusText.textContent = 'Network or server error during scan.';
    } finally {
      findBtn.disabled = false;
      findSpinner.style.display = 'none';
      findBtnText.textContent = 'SCAN & LOCATE TARGET';
    }
  }

  // Retrieve & Drop Command
  sendArmBtn.addEventListener('click', async () => {
    if (!lastFoundTarget || !lastFoundTarget.robot_xyz) return;

    sendArmBtn.disabled = true;
    findBtn.disabled = true;
    armSpinner.style.display = 'inline-block';
    armBtnText.textContent = 'EXECUTING RETRIEVAL...';

    // Progressive stage feedback
    const stages = [
      { progress: '15%', text: 'Stage 1/8: Elevating to Maximum Clearance (J2 Elevation)...' },
      { progress: '30%', text: 'Stage 2/8: Pre-rotating Gripper Jaws Parallel to Spine...' },
      { progress: '45%', text: 'Stage 3/8: Transiting Overhead to Target Coordinates...' },
      { progress: '60%', text: 'Stage 4/8: Strictly Vertical Descent Straight onto Book Spine...' },
      { progress: '75%', text: 'Stage 5/8: Clamping Pneumatic Jaws with Full Grip...' },
      { progress: '85%', text: 'Stage 6/8: Lifting Book Clear of Bookshelf...' },
      { progress: '95%', text: 'Stage 7/8: Transferring to Drop Zone Station...' },
      { progress: '100%', text: 'Stage 8/8: Releasing Book & Retracting Arm to Standby...' }
    ];

    let stageIdx = 0;
    const stageInterval = setInterval(() => {
      if (stageIdx < stages.length) {
        motionProgress.style.width = stages[stageIdx].progress;
        motionStatusText.textContent = stages[stageIdx].text;
        stageIdx++;
      }
    }, 1200);

    try {
      const response = await fetch('/api/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const res = await response.json();
      clearInterval(stageInterval);

      if (res.ok) {
        motionProgress.style.width = '100%';
        motionStatusText.textContent = 'SUCCESS: Book retrieved and deposited cleanly!';
        telemetryBadge.className = 'status-badge locked';
        telemetryBadge.textContent = 'COMPLETED';
        armBtnText.textContent = 'RETRIEVE COMPLETE';
        setTimeout(() => {
          armBtnText.textContent = 'RETRIEVE & DROP BOOK';
          sendArmBtn.disabled = false;
          findBtn.disabled = false;
        }, 3000);
      } else {
        motionProgress.style.width = '0%';
        motionStatusText.textContent = `EXECUTION FAILED: ${res.error || 'Check Dobot physical connection.'}`;
        armBtnText.textContent = 'RETRIEVE & DROP BOOK';
        sendArmBtn.disabled = false;
        findBtn.disabled = false;
      }
    } catch (err) {
      clearInterval(stageInterval);
      console.error('API Send error:', err);
      motionProgress.style.width = '0%';
      motionStatusText.textContent = 'Network or Dobot controller error.';
      armBtnText.textContent = 'RETRIEVE & DROP BOOK';
      sendArmBtn.disabled = false;
      findBtn.disabled = false;
    } finally {
      armSpinner.style.display = 'none';
    }
  });
});

// Stream reconnection fallback
function handleStreamError(img) {
  setTimeout(() => {
    img.src = '/video_feed?t=' + Date.now();
  }, 2000);
}
