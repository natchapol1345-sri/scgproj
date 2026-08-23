/* ================================================================
   VGGT 3D Reconstruction Pipeline — Application Logic
   Full API integration with real FastAPI backend + Three.js viewer
   ================================================================ */

import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── API base URL (same origin when served by FastAPI) ──
const API_BASE = '';

document.addEventListener('DOMContentLoaded', () => {
  // ── DOM References ──
  const navbar = document.getElementById('navbar');
  const navLinks = document.querySelectorAll('.nav-link');
  const navToggle = document.getElementById('nav-toggle');
  const navLinksContainer = document.getElementById('nav-links');
  const screens = document.querySelectorAll('.screen');

  // Home
  const heroStartBtn = document.getElementById('hero-start-btn');
  const heroLearnBtn = document.getElementById('hero-learn-btn');

  // Input
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const previewGrid = document.getElementById('preview-grid');
  const uploadInfo = document.getElementById('upload-info');
  const imageCount = document.getElementById('image-count');
  const totalSize = document.getElementById('total-size');
  const clearBtn = document.getElementById('clear-btn');
  const confThreshold = document.getElementById('conf-threshold');
  const confValue = document.getElementById('conf-value');
  const startPipelineBtn = document.getElementById('start-pipeline-btn');

  // Processing
  const overallProgress = document.getElementById('overall-progress');
  const progressPercent = document.getElementById('progress-percent');
  const progressEta = document.getElementById('progress-eta');
  const terminalBody = document.getElementById('terminal-body');

  // Result
  const viewerBtns = document.querySelectorAll('.viewer-btn');
  const resultRestartBtn = document.getElementById('result-restart-btn');
  const resultDownloadBtn = document.getElementById('result-download-btn');

  // ── State ──
  let uploadedFiles = [];
  let currentJobId = null;
  let pollTimer = null;

  // ── Three.js State ──
  let threeScene = null;
  let threeCamera = null;
  let threeRenderer = null;
  let threeControls = null;
  let currentObject3D = null;
  let animFrameId = null;

  // ================================================================
  //  NAVIGATION
  // ================================================================
  function switchScreen(screenId) {
    screens.forEach(s => s.classList.remove('active'));
    navLinks.forEach(l => l.classList.remove('active'));

    const target = document.getElementById(`screen-${screenId}`);
    if (target) {
      target.classList.add('active');
      target.style.animation = 'none';
      target.offsetHeight; // reflow
      target.style.animation = '';
    }

    const activeLink = document.querySelector(`.nav-link[data-screen="${screenId}"]`);
    if (activeLink) activeLink.classList.add('active');

    navLinksContainer.classList.remove('open');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Initialize Three.js when result screen is shown
    if (screenId === 'result') {
      requestAnimationFrame(() => initThreeJS());
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const screen = link.dataset.screen;
      switchScreen(screen);
    });
  });

  navToggle.addEventListener('click', () => {
    navLinksContainer.classList.toggle('open');
  });

  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });

  document.getElementById('nav-logo').addEventListener('click', e => {
    e.preventDefault();
    switchScreen('home');
  });

  // ================================================================
  //  HOME SCREEN
  // ================================================================
  heroStartBtn.addEventListener('click', () => switchScreen('input'));
  heroLearnBtn.addEventListener('click', () => {
    document.getElementById('pipeline-overview').scrollIntoView({ behavior: 'smooth' });
  });

  // Animate pipeline steps on scroll
  const pipelineSteps = document.querySelectorAll('.pipeline-step');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        entry.target.style.animation = `fadeInUp 0.5s ${i * 0.08}s var(--ease-out) both`;
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  pipelineSteps.forEach(step => observer.observe(step));

  // ================================================================
  //  INPUT SCREEN — Upload
  // ================================================================
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
    fileInput.value = '';
  });

  function handleFiles(fileList) {
    const validExts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'];

    Array.from(fileList).forEach(file => {
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (validExts.includes(ext)) {
        uploadedFiles.push(file);
      }
    });

    renderPreviews();
  }

  function renderPreviews() {
    previewGrid.innerHTML = '';

    if (uploadedFiles.length === 0) {
      uploadInfo.style.display = 'none';
      return;
    }

    uploadedFiles.forEach((file, idx) => {
      const item = document.createElement('div');
      item.className = 'preview-item';
      item.style.animationDelay = `${idx * 0.05}s`;

      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.alt = file.name;

      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove-btn';
      removeBtn.innerHTML = '<i class="fas fa-times"></i>';
      removeBtn.addEventListener('click', e => {
        e.stopPropagation();
        uploadedFiles.splice(idx, 1);
        renderPreviews();
      });

      const name = document.createElement('span');
      name.className = 'preview-name';
      name.textContent = file.name;

      item.appendChild(img);
      item.appendChild(removeBtn);
      item.appendChild(name);
      previewGrid.appendChild(item);
    });

    uploadInfo.style.display = 'flex';
    imageCount.textContent = `${uploadedFiles.length} image${uploadedFiles.length > 1 ? 's' : ''}`;
    const sizeMB = uploadedFiles.reduce((sum, f) => sum + f.size, 0) / (1024 * 1024);
    totalSize.textContent = `${sizeMB.toFixed(1)} MB`;
  }

  clearBtn.addEventListener('click', () => {
    uploadedFiles = [];
    renderPreviews();
  });

  if (confThreshold) {
    confThreshold.addEventListener('input', () => {
      if (confValue) confValue.textContent = `${confThreshold.value}%`;
    });
  }

  // ================================================================
  //  START PIPELINE (real API)
  // ================================================================
  startPipelineBtn.addEventListener('click', async () => {
    if (uploadedFiles.length === 0) {
      alert('Please upload at least one image before running the pipeline.');
      return;
    }

    // Disable button
    startPipelineBtn.disabled = true;
    startPipelineBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Uploading...</span>';

    try {
      // Get ref size from input
      const refSizeInput = document.getElementById('ref-size');
      const refSize = refSizeInput ? parseFloat(refSizeInput.value) || 14.0 : 14.0;

      // Step 1: Upload images → create job
      const formData = new FormData();
      uploadedFiles.forEach(f => formData.append('images', f));
      formData.append('ref_size_cm', refSize.toString());

      const createResp = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        body: formData,
      });

      if (!createResp.ok) {
        const err = await createResp.json().catch(() => ({ detail: createResp.statusText }));
        throw new Error(err.detail || 'Failed to create job');
      }

      const { job_id } = await createResp.json();
      currentJobId = job_id;

      // Step 2: Start pipeline
      const runResp = await fetch(`${API_BASE}/api/jobs/${job_id}/run`, { method: 'POST' });
      if (!runResp.ok) {
        const err = await runResp.json().catch(() => ({ detail: runResp.statusText }));
        throw new Error(err.detail || 'Failed to start pipeline');
      }

      // Switch to processing screen and start polling
      switchScreen('processing');
      resetProcessingUI();
      startPolling(job_id);

    } catch (error) {
      console.error('Pipeline start error:', error);
      alert(`Error: ${error.message}`);
    } finally {
      startPipelineBtn.disabled = false;
      startPipelineBtn.innerHTML = '<i class="fas fa-rocket"></i> <span>Run Pipeline</span>';
    }
  });

  // ================================================================
  //  PROCESSING SCREEN — Real polling
  // ================================================================
  function resetProcessingUI() {
    for (let i = 1; i <= 7; i++) {
      const card = document.getElementById(`stage-${i}`);
      if (!card) continue;
      card.classList.remove('active', 'completed');
      card.querySelector('.status-indicator').className = 'status-indicator pending';
      card.querySelector('.mini-bar').style.width = '0%';
    }
    overallProgress.style.width = '0%';
    progressPercent.textContent = '0%';
    progressEta.textContent = 'Starting pipeline...';
    if (terminalBody) {
      terminalBody.innerHTML = '<div class="terminal-line"><span class="term-prompt">$</span> <span class="term-cmd">Pipeline started — waiting for server...</span></div>';
    }
  }

  function startPolling(jobId) {
    if (pollTimer) clearInterval(pollTimer);

    let lastStageMessages = {};

    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
        if (!resp.ok) return;

        const status = await resp.json();
        updateProcessingUI(status, lastStageMessages);

        // Check if done or error
        if (status.overall === 'done') {
          clearInterval(pollTimer);
          pollTimer = null;
          // Wait a moment then switch to result
          appendTerminalLine('', '');
          appendTerminalLine('╔══════════════════════════════════════════════════════════╗', 'term-success');
          appendTerminalLine('║  Pipeline Complete                                       ║', 'term-success');
          appendTerminalLine('╚══════════════════════════════════════════════════════════╝', 'term-success');
          setTimeout(() => loadAndShowResults(jobId), 1500);
        } else if (status.overall === 'error') {
          clearInterval(pollTimer);
          pollTimer = null;
          appendTerminalLine(`ERROR: ${status.error || 'Unknown error'}`, 'term-error');
          progressEta.textContent = 'Pipeline failed';
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    }, 1500);
  }

  function updateProcessingUI(status, lastMsgs) {
    let doneCount = 0;

    status.stages.forEach(stage => {
      const card = document.getElementById(`stage-${stage.id}`);
      if (!card) return;

      const indicator = card.querySelector('.status-indicator');
      const miniBar = card.querySelector('.mini-bar');

      card.classList.remove('active', 'completed');

      switch (stage.status) {
        case 'running':
          card.classList.add('active');
          indicator.className = 'status-indicator active';
          miniBar.style.width = '50%';
          break;
        case 'done':
          card.classList.add('completed');
          indicator.className = 'status-indicator completed';
          miniBar.style.width = '100%';
          doneCount++;
          break;
        case 'skipped':
          card.classList.add('completed');
          indicator.className = 'status-indicator completed';
          miniBar.style.width = '100%';
          doneCount++;
          break;
        case 'error':
          indicator.className = 'status-indicator error';
          miniBar.style.width = '100%';
          miniBar.style.background = 'var(--c-error, #ef4444)';
          break;
        default:
          indicator.className = 'status-indicator pending';
          miniBar.style.width = '0%';
      }

      // Add terminal log when message changes
      if (stage.message && stage.message !== lastMsgs[stage.id]) {
        lastMsgs[stage.id] = stage.message;
        const cls = stage.status === 'done' ? 'term-success'
          : stage.status === 'error' ? 'term-error'
          : stage.status === 'running' ? 'term-info'
          : '';
        appendTerminalLine(`[Stage ${stage.id}] ${stage.message}`, cls);
      }
    });

    // Update overall progress
    const pct = Math.round((doneCount / 7) * 100);
    overallProgress.style.width = `${pct}%`;
    progressPercent.textContent = `${pct}%`;

    if (doneCount < 7 && doneCount > 0) {
      progressEta.textContent = `${doneCount}/7 stages complete`;
    } else if (doneCount === 7) {
      progressEta.textContent = 'All stages complete!';
    }
  }

  function appendTerminalLine(text, cls) {
    if (!terminalBody) return;
    const line = document.createElement('div');
    line.className = 'terminal-line';
    if (cls) {
      line.innerHTML = `<span class="${cls}">${escapeHtml(text)}</span>`;
    } else {
      line.textContent = text;
    }
    terminalBody.appendChild(line);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ================================================================
  //  RESULT SCREEN — Load real data
  // ================================================================
  async function loadAndShowResults(jobId) {
    switchScreen('result');

    try {
      // Fetch result data
      const resultResp = await fetch(`${API_BASE}/api/jobs/${jobId}/result`);
      if (!resultResp.ok) throw new Error('Failed to fetch results');
      const result = await resultResp.json();

      // Populate summary cards
      document.getElementById('result-time').textContent = `${result.total_time}s`;
      document.getElementById('result-points').textContent = result.point_count.toLocaleString();
      document.getElementById('result-faces').textContent = result.face_count.toLocaleString();
      document.getElementById('result-watertight').textContent = result.is_watertight ? 'Yes' : 'No';

      // Populate volume table
      const volumeTable = document.getElementById('volume-table');
      if (volumeTable && result.volumes && result.volumes.length > 0) {
        const tbody = volumeTable.querySelector('tbody');
        tbody.innerHTML = '';

        result.volumes.forEach(vol => {
          const tr = document.createElement('tr');
          const label = vol.is_ref ? 'ArUco Ref' : vol.name.replace('.ply', '').replace('_', ' ');
          const tagClass = vol.is_ref ? 'ref' : 'obj';

          tr.innerHTML = `
            <td><span class="object-tag ${tagClass}">${escapeHtml(label)}</span></td>
            <td>${vol.size_x_cm.toFixed(2)} cm</td>
            <td>${vol.size_y_cm.toFixed(2)} cm</td>
            <td>${vol.size_z_cm.toFixed(2)} cm</td>
            <td><strong>${vol.real_vol_cm3.toLocaleString(undefined, {minimumFractionDigits:1, maximumFractionDigits:1})}</strong></td>
            <td><span class="method-badge">${escapeHtml(vol.method)}</span></td>
          `;
          tbody.appendChild(tr);
        });
      }

      // Populate scale info
      const scaleInfo = document.querySelector('.scale-info');
      if (scaleInfo && result.k !== null) {
        scaleInfo.innerHTML = `
          <div class="scale-formula">
            <span class="formula-label">Scale Factor</span>
            <code>k = V_real / V_mesh = ${result.k.toLocaleString()} cm³/unit³</code>
          </div>
          <div class="scale-formula">
            <span class="formula-label">Linear Scale</span>
            <code>s = k^(1/3) = ${result.linear_scale} cm/unit</code>
          </div>
        `;
      }

      // Fetch and populate file tree
      const filesResp = await fetch(`${API_BASE}/api/jobs/${jobId}/files`);
      if (filesResp.ok) {
        const { files } = await filesResp.json();
        buildFileTree(files, jobId);
      }

      // Load the default 3D view (object mesh)
      loadViewerModel('object');

    } catch (error) {
      console.error('Error loading results:', error);
    }
  }

  function buildFileTree(files, jobId) {
    const fileTreeContainer = document.querySelector('.file-tree');
    if (!fileTreeContainer) return;

    // Build a tree structure from flat file list
    const tree = {};
    files.forEach(f => {
      const parts = f.path.split('/');
      let node = tree;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!node[parts[i]]) node[parts[i]] = {};
        node = node[parts[i]];
      }
      node[parts[parts.length - 1]] = f;
    });

    fileTreeContainer.innerHTML = '';
    fileTreeContainer.appendChild(renderTreeNode('output/', tree, jobId, true));
  }

  function renderTreeNode(name, node, jobId, isOpen = false) {
    // Check if this is a file (has path property)
    if (node.path) {
      const div = document.createElement('div');
      div.className = 'file-item';
      div.innerHTML = `<i class="fas fa-file"></i> <span>${escapeHtml(node.name)}</span> <span class="file-size">${node.size_mb} MB</span>`;
      div.style.cursor = 'pointer';
      div.addEventListener('click', () => {
        window.open(`${API_BASE}/api/jobs/${jobId}/files/${node.path}`, '_blank');
      });
      return div;
    }

    // It's a folder
    const folder = document.createElement('div');
    folder.className = `file-folder${isOpen ? ' open' : ''}`;

    const header = document.createElement('div');
    header.className = 'folder-header';
    header.innerHTML = `
      <i class="fas fa-chevron-${isOpen ? 'down' : 'right'}"></i>
      <i class="fas fa-folder${isOpen ? '-open' : ''}"></i>
      <span>${escapeHtml(name)}</span>
    `;

    const children = document.createElement('div');
    children.className = 'folder-children';
    children.style.display = isOpen ? 'block' : 'none';

    Object.keys(node).sort().forEach(key => {
      children.appendChild(renderTreeNode(key, node[key], jobId, false));
    });

    header.addEventListener('click', () => {
      folder.classList.toggle('open');
      const isNowOpen = folder.classList.contains('open');
      children.style.display = isNowOpen ? 'block' : 'none';
      header.querySelector('i:first-child').className = `fas fa-chevron-${isNowOpen ? 'down' : 'right'}`;
      header.querySelector('i:nth-child(2)').className = `fas fa-folder${isNowOpen ? '-open' : ''}`;
    });

    folder.appendChild(header);
    folder.appendChild(children);
    return folder;
  }

  // ================================================================
  //  THREE.JS 3D VIEWER
  // ================================================================
  function initThreeJS() {
    const canvas = document.getElementById('three-canvas');
    if (!canvas || threeRenderer) return; // already initialized

    const container = document.getElementById('viewer-canvas');
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    // Scene
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x0a0a1a);

    // Camera
    threeCamera = new THREE.PerspectiveCamera(50, width / height, 0.001, 1000);
    threeCamera.position.set(0, 0, 0.5);

    // Renderer
    threeRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    threeControls = new OrbitControls(threeCamera, threeRenderer.domElement);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.08;
    threeControls.enableZoom = true;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    threeScene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(1, 1, 1);
    threeScene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.4);
    dirLight2.position.set(-1, -0.5, -1);
    threeScene.add(dirLight2);

    // Grid helper (subtle)
    const grid = new THREE.GridHelper(2, 20, 0x333355, 0x222244);
    grid.material.opacity = 0.3;
    grid.material.transparent = true;
    threeScene.add(grid);

    // Animation loop
    function animate() {
      animFrameId = requestAnimationFrame(animate);
      threeControls.update();
      threeRenderer.render(threeScene, threeCamera);
    }
    animate();

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight || 400;
      threeCamera.aspect = w / h;
      threeCamera.updateProjectionMatrix();
      threeRenderer.setSize(w, h);
    });
    resizeObserver.observe(container);
  }

  // Map viewer button data-view to PLY file paths
  const VIEW_FILE_MAP = {
    object: 'mesh/obj.ply',
    aruco: 'mesh/box.ply',
    scene: 'mesh/scene_colour.ply',
    pointcloud: 'points.ply',
  };

  function loadViewerModel(viewName) {
    if (!currentJobId || !threeScene) return;

    const filePath = VIEW_FILE_MAP[viewName];
    if (!filePath) return;

    const loadingEl = document.getElementById('viewer-loading');
    if (loadingEl) loadingEl.style.display = 'flex';

    // Remove previous object
    if (currentObject3D) {
      threeScene.remove(currentObject3D);
      if (currentObject3D.geometry) currentObject3D.geometry.dispose();
      if (currentObject3D.material) currentObject3D.material.dispose();
      currentObject3D = null;
    }

    const url = `${API_BASE}/api/jobs/${currentJobId}/files/${filePath}`;
    const loader = new PLYLoader();

    loader.load(
      url,
      (geometry) => {
        geometry.computeVertexNormals();

        // Center the geometry
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        // Calculate bounding sphere for camera positioning
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere.radius;

        let object3D;

        if (viewName === 'pointcloud') {
          // Render as points
          const hasColors = geometry.hasAttribute('color');
          const material = new THREE.PointsMaterial({
            size: 0.003,
            vertexColors: hasColors,
            color: hasColors ? undefined : 0x38bdf8,
            sizeAttenuation: true,
          });
          object3D = new THREE.Points(geometry, material);
        } else {
          // Render as mesh
          const hasColors = geometry.hasAttribute('color');
          const material = new THREE.MeshStandardMaterial({
            vertexColors: hasColors,
            color: hasColors ? undefined : 0x6366f1,
            side: THREE.DoubleSide,
            roughness: 0.6,
            metalness: 0.1,
            flatShading: false,
          });
          object3D = new THREE.Mesh(geometry, material);
        }

        threeScene.add(object3D);
        currentObject3D = object3D;

        // Position camera to fit the model
        const dist = radius * 2.5;
        threeCamera.position.set(dist * 0.7, dist * 0.5, dist * 0.7);
        threeCamera.lookAt(0, 0, 0);
        threeControls.target.set(0, 0, 0);
        threeControls.update();

        if (loadingEl) loadingEl.style.display = 'none';
      },
      (progress) => {
        // Progress callback
      },
      (error) => {
        console.error('PLY load error:', error);
        if (loadingEl) {
          loadingEl.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Failed to load model</span>';
        }
      }
    );
  }

  // Viewer tab buttons
  viewerBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      viewerBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;
      loadViewerModel(view);
    });
  });

  // ================================================================
  //  RESULT SCREEN — Actions
  // ================================================================

  // Restart button
  resultRestartBtn.addEventListener('click', () => {
    uploadedFiles = [];
    currentJobId = null;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    // Clean up Three.js
    if (currentObject3D && threeScene) {
      threeScene.remove(currentObject3D);
      currentObject3D = null;
    }
    renderPreviews();
    switchScreen('input');
  });

  // Download button — real zip download
  resultDownloadBtn.addEventListener('click', async () => {
    if (!currentJobId) return;

    const btn = resultDownloadBtn;
    const origText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Preparing...</span>';
    btn.disabled = true;

    try {
      const resp = await fetch(`${API_BASE}/api/jobs/${currentJobId}/files.zip`);
      if (!resp.ok) throw new Error('Download failed');

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `output_${currentJobId}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      btn.innerHTML = '<i class="fas fa-check"></i> <span>Downloaded!</span>';
      btn.style.borderColor = 'var(--c-success)';
      btn.style.color = 'var(--c-success)';
      setTimeout(() => {
        btn.innerHTML = origText;
        btn.style.borderColor = '';
        btn.style.color = '';
      }, 2000);
    } catch (error) {
      console.error('Download error:', error);
      btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> <span>Download failed</span>';
      setTimeout(() => {
        btn.innerHTML = origText;
      }, 2000);
    } finally {
      btn.disabled = false;
    }
  });

  // ================================================================
  //  SMOOTH SCROLL INTERSECTION ANIMATIONS
  // ================================================================
  const animateOnScroll = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        animateOnScroll.unobserve(entry.target);
      }
    });
  }, { threshold: 0.05 });

  // Observe tech cards
  document.querySelectorAll('.tech-card').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = `all 0.5s ${i * 0.08}s var(--ease-out)`;
    animateOnScroll.observe(card);
  });

  // Observe summary cards on result page
  document.querySelectorAll('.summary-card').forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = `all 0.5s ${i * 0.1}s var(--ease-out)`;
    animateOnScroll.observe(card);
  });
});
