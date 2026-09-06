(() => {
  const formNode = document.getElementById("form");
  const grid = formNode?.querySelector(".grid");
  const frameStripNode = document.getElementById("frameStrip");
  const geometry = document.getElementById("geometryMode");
  const multiplier = document.getElementById("rifeMultiplier");
  const statusNode = document.getElementById("status");
  const dropZone = document.getElementById("dropZone");
  if (!formNode || !grid || !frameStripNode || !geometry || !multiplier) return;
  if (formNode.dataset.advancedEngines === "1") return;
  formNode.dataset.advancedEngines = "1";

  const style = document.createElement("style");
  style.textContent = `
    .advanced-engine-details { margin-top:16px; border:1px solid color-mix(in srgb, CanvasText 16%, transparent); border-radius:12px; padding:12px 14px; }
    .advanced-engine-details > summary { cursor:pointer; font-weight:750; }
    .advanced-engine-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:14px; }
    .advanced-engine-grid label { display:grid; gap:6px; font-size:.9rem; }
    .advanced-engine-grid .wide { grid-column:1 / -1; }
    .advanced-engine-grid select[multiple] { min-height:112px; }
    .smart-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .smart-actions button { width:auto; margin:0; padding:8px 11px; border-radius:9px; border:1px solid color-mix(in srgb, CanvasText 18%, transparent); background:Canvas; color:CanvasText; font-weight:700; }
    #smartAnalysisResult { white-space:pre-wrap; font-size:.82rem; opacity:.82; line-height:1.4; }
    .analysis-progress, .upload-progress { display:none; margin-top:10px; }
    .analysis-progress.visible, .upload-progress.visible { display:block; }
    .compact-progress-top { display:flex; justify-content:space-between; gap:10px; margin-bottom:6px; font-size:.82rem; }
    .compact-progress-track { height:10px; border-radius:999px; overflow:hidden; background:color-mix(in srgb, CanvasText 12%, transparent); }
    .compact-progress-bar { width:0%; height:100%; background:Highlight; transition:width .15s ease; }
    .upload-progress { margin:10px 0 2px; padding:0 2px; }
    @media(max-width:620px){ .advanced-engine-grid { grid-template-columns:1fr; } .advanced-engine-grid .wide { grid-column:auto; } }
  `;
  document.head.appendChild(style);

  if (!geometry.querySelector('option[value="none"]')) {
    const node = document.createElement("option");
    node.value = "none";
    node.textContent = "None · preserve source geometry";
    geometry.insertBefore(node, geometry.firstChild);
  }

  function option(value, text) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = text;
    return node;
  }

  function labelledSelect(labelText, id, name, values) {
    const label = document.createElement("label");
    label.className = "option";
    label.append(document.createTextNode(labelText));
    const select = document.createElement("select");
    select.id = id;
    select.name = name;
    values.forEach(([value, text]) => select.append(option(value, text)));
    label.append(select);
    return [label, select];
  }

  const oldMultiplierLabel = multiplier.closest("label.option");
  const [interpolatorLabel, interpolator] = labelledSelect(
    "Interpolator",
    "interpolator",
    "interpolator",
    [["none", "None"], ["rife", "RIFE"], ["amt", "AMT"]],
  );
  const interpolatorHint = document.createElement("span");
  interpolatorHint.className = "hint";
  interpolatorHint.textContent = "Conventional temporal interpolation. Can work on all gaps, manual targets, automatic missing gaps, and the loop seam.";
  interpolatorLabel.append(interpolatorHint);

  const [generatorLabel, frameGenerator] = labelledSelect(
    "Frame generator",
    "frameGenerator",
    "frame_generator",
    [["none", "None"], ["eden", "EDEN"], ["speed", "SPEED"]],
  );
  const generatorHint = document.createElement("span");
  generatorHint.className = "hint";
  generatorHint.textContent = "Structural frame generation. With an interpolator selected, the generated midpoint becomes an anchor for further filling.";
  generatorLabel.append(generatorHint);

  if (oldMultiplierLabel) {
    oldMultiplierLabel.insertAdjacentElement("beforebegin", interpolatorLabel);
    interpolatorLabel.insertAdjacentElement("afterend", generatorLabel);
    for (const node of oldMultiplierLabel.childNodes) {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
        node.textContent = "\n          Interpolation density\n          ";
        break;
      }
    }
    const hint = oldMultiplierLabel.querySelector(".hint");
    if (hint) hint.textContent = "Used for conventional all-gap interpolation. Smart/manual gap filling uses Frames to fill below.";
    [...multiplier.options].forEach(item => {
      if (item.value === "1") item.textContent = "1× · no conventional interpolation";
    });
  } else {
    grid.append(interpolatorLabel, generatorLabel);
  }

  const details = document.createElement("details");
  details.className = "advanced-engine-details";
  details.innerHTML = `
    <summary>Smart temporal optimisation and targeted gaps</summary>
    <div class="advanced-engine-grid">
      <label>
        <span><input id="smartReduction" type="checkbox" name="smart_reduction"> Smart frame reduction</span>
        <span class="hint">Find existing frames that add little unique temporal information.</span>
      </label>
      <label>
        Reduction threshold (%)
        <input id="reductionThreshold" type="number" name="reduction_threshold" value="2" min="0" max="100" step="0.1">
      </label>
      <label>
        <span><input id="smartMissing" type="checkbox" name="smart_missing"> Smart missing-frame finder</span>
        <span class="hint">Automatically find sparse temporal gaps. Manual targets remain explicit overrides.</span>
      </label>
      <label>
        Missing-frame threshold (%)
        <input id="missingThreshold" type="number" name="missing_threshold" value="12" min="0" max="100" step="0.1">
      </label>
      <label>
        Loop analysis
        <select id="loopAnalysis" name="loop_analysis">
          <option value="off">Off</option>
          <option value="alongside">Analyse last → first alongside normal gaps</option>
          <option value="only">Analyse last → first only</option>
        </select>
        <span class="hint">The loop seam is treated as a real temporal gap between the final and first frame.</span>
      </label>
      <label>
        Frames to fill
        <input id="framesToFill" type="number" name="frames_to_fill" value="1" min="0" step="1" inputmode="numeric">
        <span class="hint">Positive = exact inserted count per selected/discovered gap. 0 = keep filling until the missing threshold is met, subject to a safety ceiling.</span>
      </label>
      <label class="wide">
        Target gaps (optional)
        <select id="targetGapSelect" multiple></select>
        <span class="hint">Manual selections are always processed. With Smart Missing enabled, automatic analysis may add more gaps. Loop: Last ↔ First can be selected directly.</span>
      </label>
      <input id="targetGaps" type="hidden" name="target_gaps" value="">
      <div class="smart-actions wide">
        <button id="analyseSequence" type="button">Analyse sequence</button>
        <button id="applyReduction" type="button" disabled>Apply suggested reduction</button>
        <button id="undoReduction" type="button" disabled>Undo reduction</button>
        <button id="selectMissingGaps" type="button" disabled>Add suggested missing gaps</button>
      </div>
      <div id="analysisProgress" class="analysis-progress wide">
        <div class="compact-progress-top"><span id="analysisProgressLabel">Queued</span><strong id="analysisProgressPercent">0%</strong></div>
        <div class="compact-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="analysisProgressTrack"><div id="analysisProgressBar" class="compact-progress-bar"></div></div>
      </div>
      <div id="smartAnalysisResult" class="wide">No analysis run yet.</div>
    </div>
  `;
  grid.insertAdjacentElement("afterend", details);

  // Dedicated byte-based upload progress immediately below the upload control.
  let uploadWrap = document.getElementById("uploadProgress");
  if (!uploadWrap && dropZone) {
    uploadWrap = document.createElement("div");
    uploadWrap.id = "uploadProgress";
    uploadWrap.className = "upload-progress";
    uploadWrap.innerHTML = `
      <div class="compact-progress-top"><span id="uploadProgressLabel">Uploading…</span><strong id="uploadProgressPercent">0%</strong></div>
      <div class="compact-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="uploadProgressTrack"><div id="uploadProgressBar" class="compact-progress-bar"></div></div>
    `;
    dropZone.insertAdjacentElement("afterend", uploadWrap);
  }

  const uploadLabel = document.getElementById("uploadProgressLabel");
  const uploadPercent = document.getElementById("uploadProgressPercent");
  const uploadTrack = document.getElementById("uploadProgressTrack");
  const uploadBar = document.getElementById("uploadProgressBar");
  let uploadHideTimer = null;

  function formatBytes(value) {
    const bytes = Math.max(0, Number(value) || 0);
    if (bytes < 1024) return `${Math.round(bytes)} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  function setUploadProgressRatio(ratio, expectedBytes, label = "Uploading") {
    if (!uploadWrap || !uploadBar) return;
    clearTimeout(uploadHideTimer);
    const safeRatio = Math.max(0, Math.min(1, Number(ratio) || 0));
    const pct = Math.round(safeRatio * 100);
    const equivalent = Math.round(Math.max(0, Number(expectedBytes) || 0) * safeRatio);
    uploadWrap.classList.add("visible");
    uploadBar.style.width = `${pct}%`;
    uploadPercent.textContent = `${pct}%`;
    uploadTrack?.setAttribute("aria-valuenow", String(pct));
    uploadLabel.textContent = expectedBytes > 0
      ? `${label} · ${formatBytes(equivalent)} / ${formatBytes(expectedBytes)}`
      : label;
  }

  function finishUploadProgress(expectedBytes, label = "Upload complete") {
    setUploadProgressRatio(1, expectedBytes, label);
    uploadHideTimer = setTimeout(() => uploadWrap?.classList.remove("visible"), 1400);
  }

  function failUploadProgress(label = "Upload failed") {
    if (!uploadWrap) return;
    clearTimeout(uploadHideTimer);
    uploadWrap.classList.add("visible");
    if (uploadLabel) uploadLabel.textContent = label;
  }

  function xhrJsonUpload(url, data, expectedBytes, label) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      setUploadProgressRatio(0, expectedBytes, label);
      xhr.upload.addEventListener("progress", event => {
        let ratio = 0;
        if (event.lengthComputable && event.total > 0) ratio = event.loaded / event.total;
        else if (expectedBytes > 0) ratio = Math.min(1, event.loaded / expectedBytes);
        setUploadProgressRatio(ratio, expectedBytes, label);
      });
      xhr.addEventListener("load", () => {
        if (xhr.status < 200 || xhr.status >= 300) {
          failUploadProgress(`${label} failed`);
          reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`));
          return;
        }
        finishUploadProgress(expectedBytes);
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { reject(new Error("Server returned invalid JSON.")); }
      });
      xhr.addEventListener("error", () => {
        failUploadProgress(`${label} failed`);
        reject(new Error("Network error while uploading."));
      });
      xhr.send(data);
    });
  }

  window.webpUploadProgress = {
    setRatio: setUploadProgressRatio,
    complete: finishUploadProgress,
    fail: failUploadProgress,
  };

  // Replace the legacy upload functions so upload bytes no longer occupy the
  // generation progress bar. XHR gives real request-body upload progress.
  const trackedExtractWebP = file => {
    const data = new FormData();
    data.append("webp", file, file.name);
    return xhrJsonUpload("/extract-webp", data, Number(file.size || 0), `Uploading ${file.name}`);
  };

  const trackedUploadJob = formData => {
    const expected = frames.reduce((sum, item) => sum + Number(item?.file?.size || 0), 0);
    return xhrJsonUpload("/generate", formData, expected, "Uploading frames");
  };

  try { extractWebP = trackedExtractWebP; } catch {}
  try { uploadJob = trackedUploadJob; } catch {}
  window.extractWebP = trackedExtractWebP;
  window.uploadJob = trackedUploadJob;

  // Keep the normal generation bar purely server-side now that upload has its own bar.
  async function pollJobServerProgress(jobId, generation) {
    let transientFailures = 0;
    while (generation === monitorGeneration) {
      try {
        const response = await fetch(`/progress?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
        if (response.status === 404) {
          clearActiveJob(jobId);
          throw new Error("The saved job is no longer available on the server.");
        }
        if (!response.ok) throw new Error(`Progress check failed (${response.status})`);
        transientFailures = 0;
        const data = await response.json();
        setProgress(Number(data.progress || 0), data.message || "Processing");
        status.textContent = data.message || "";
        if (data.status === "done") return data;
        if (data.status === "error") {
          clearActiveJob(jobId);
          throw new Error(data.error || data.message || "Generation failed.");
        }
      } catch (error) {
        if (error.message.includes("saved job is no longer available")) throw error;
        transientFailures += 1;
        status.textContent = `Processing continues on the server. Reconnecting…${transientFailures > 1 ? ` (${transientFailures})` : ""}`;
        await new Promise(resolve => setTimeout(resolve, Math.min(5000, 750 * transientFailures)));
        continue;
      }
      await new Promise(resolve => setTimeout(resolve, 750));
    }
    return null;
  }

  async function monitorJobServerProgress(jobId, { resumed = false, autoDownload = true } = {}) {
    const generation = ++monitorGeneration;
    saveActiveJob(jobId);
    button.disabled = true;
    button.textContent = resumed ? "Resuming job…" : "Generating…";
    downloadLink.classList.remove("visible");
    setProgress(0, resumed ? "Reconnecting to server job" : "Processing on server");
    status.textContent = resumed
      ? "Reconnected. Processing continued on the server while the page was away."
      : "Upload complete. Processing is now independent of this page.";
    try {
      const result = await pollJobServerProgress(jobId, generation);
      if (!result || generation !== monitorGeneration) return;
      clearActiveJob(jobId);
      setProgress(100, "Complete");
      status.textContent = result.message || "Done.";
      showDownload(jobId, autoDownload && !document.hidden);
    } catch (error) {
      if (generation !== monitorGeneration) return;
      status.textContent = `Error: ${error.message}`;
      progressLabel.textContent = "Failed";
    } finally {
      if (generation === monitorGeneration) {
        button.disabled = false;
        button.textContent = "Generate WebP";
      }
    }
  }

  try { pollJob = pollJobServerProgress; } catch {}
  try { monitorJob = monitorJobServerProgress; } catch {}
  window.pollJob = pollJobServerProgress;
  window.monitorJob = monitorJobServerProgress;

  const targetGapSelect = document.getElementById("targetGapSelect");
  const targetGaps = document.getElementById("targetGaps");
  const analyseButton = document.getElementById("analyseSequence");
  const applyReduction = document.getElementById("applyReduction");
  const undoReduction = document.getElementById("undoReduction");
  const selectMissingGaps = document.getElementById("selectMissingGaps");
  const analysisResult = document.getElementById("smartAnalysisResult");
  const smartReduction = document.getElementById("smartReduction");
  const smartMissing = document.getElementById("smartMissing");
  const analysisProgress = document.getElementById("analysisProgress");
  const analysisProgressLabel = document.getElementById("analysisProgressLabel");
  const analysisProgressPercent = document.getElementById("analysisProgressPercent");
  const analysisProgressTrack = document.getElementById("analysisProgressTrack");
  const analysisProgressBar = document.getElementById("analysisProgressBar");
  let lastAnalysis = null;
  let reductionUndo = null;
  let engineState = null;
  let analysisMonitorToken = 0;

  function refreshGapChoices() {
    const selected = new Set([...targetGapSelect.selectedOptions].map(item => item.value));
    targetGapSelect.innerHTML = "";
    for (let index = 0; index < Math.max(0, frames.length - 1); index += 1) {
      const value = String(index + 1);
      const item = option(value, `Frame ${index + 1} ↔ ${index + 2}`);
      item.selected = selected.has(value);
      targetGapSelect.append(item);
    }
    if (frames.length >= 2) {
      const loop = option("loop", `Loop: Frame ${frames.length} ↔ Frame 1`);
      loop.selected = selected.has("loop");
      targetGapSelect.append(loop);
    }
    syncTargetGaps();
  }

  function syncTargetGaps() {
    targetGaps.value = [...targetGapSelect.selectedOptions].map(item => item.value).join(",");
  }
  targetGapSelect.addEventListener("change", syncTargetGaps);

  function setAnalysisProgress(value, label) {
    const pct = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    analysisProgress.classList.add("visible");
    analysisProgressBar.style.width = `${pct}%`;
    analysisProgressPercent.textContent = `${pct}%`;
    analysisProgressTrack.setAttribute("aria-valuenow", String(pct));
    if (label) analysisProgressLabel.textContent = label;
  }

  function setEngineAvailability() {
    if (!engineState) return;
    const selected = interpolator.value;
    try { rifeReady = selected !== "rife" || Boolean(engineState.rife?.ready); } catch {}
    if (selected === "none") {
      multiplier.value = "1";
      multiplier.disabled = true;
    } else {
      multiplier.disabled = false;
      if (multiplier.value === "1") multiplier.value = "2";
    }
    try { updateDurationHint(); } catch {}
  }

  interpolator.addEventListener("change", setEngineAvailability);

  async function loadEngineStatus() {
    try {
      const response = await fetch("/engine-status", { cache: "no-store" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      engineState = await response.json();
      const bits = [];
      for (const name of ["rife", "amt", "eden", "speed"]) {
        bits.push(`${name.toUpperCase()}: ${engineState[name]?.ready ? "ready" : "missing"}`);
      }
      const accel = engineState.acceleration;
      bits.push(accel?.cuda ? `GPU matching: ${accel.device || "CUDA"}` : "GPU matching: CPU fallback");
      const node = document.getElementById("rifeStatus");
      if (node) node.textContent = bits.join(" · ");
      setEngineAvailability();
    } catch (error) {
      const node = document.getElementById("rifeStatus");
      if (node) node.textContent = `Could not check engine status: ${error.message}`;
    }
  }

  function engineReadyForSubmit() {
    if (!engineState) return true;
    if (interpolator.value !== "none" && !engineState[interpolator.value]?.ready) {
      statusNode.textContent = `${interpolator.value.toUpperCase()} is selected but is not installed.`;
      return false;
    }
    if (frameGenerator.value !== "none" && !engineState[frameGenerator.value]?.ready) {
      statusNode.textContent = `${frameGenerator.value.toUpperCase()} is selected but is not installed.`;
      return false;
    }
    return true;
  }

  formNode.addEventListener("submit", event => {
    syncTargetGaps();
    setEngineAvailability();
    if (!engineReadyForSubmit()) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);

  function applyAnalysisResult(result) {
    lastAnalysis = result || null;
    if (!lastAnalysis) return;
    const removed = lastAnalysis.reduction_removed || [];
    const missing = lastAnalysis.missing_gaps || [];
    const lines = [
      `Suggested frame count: ${lastAnalysis.source_count} → ${lastAnalysis.suggested_count}`,
      removed.length ? `Redundant source frames: ${removed.join(", ")}` : "No source frames are below the reduction threshold.",
      missing.length
        ? `Sparse gaps: ${missing.map(item => `${item.label || item.gap_key} (${Number(item.score).toFixed(2)}%)`).join(", ")}`
        : "No automatically analysed gaps exceed the missing-frame threshold.",
    ];
    if (lastAnalysis.loop_gap_score !== null && lastAnalysis.loop_gap_score !== undefined) {
      lines.push(`Loop last → first score: ${Number(lastAnalysis.loop_gap_score).toFixed(2)}%`);
    }
    if (lastAnalysis.loop_dwell?.detected) lines.push(lastAnalysis.loop_dwell.message);
    if (lastAnalysis.manual_gaps?.length) lines.push(`Manual targets retained: ${lastAnalysis.manual_gaps.join(", ")}`);
    analysisResult.textContent = lines.join("\n");
    applyReduction.disabled = removed.length === 0;
    selectMissingGaps.disabled = missing.length === 0;
  }

  async function monitorAnalysis(jobId, analysisId) {
    const token = ++analysisMonitorToken;
    analyseButton.disabled = true;
    let failures = 0;
    while (token === analysisMonitorToken) {
      try {
        const response = await fetch(`/analysis?id=${encodeURIComponent(jobId)}&analysis_id=${encodeURIComponent(analysisId)}`, { cache: "no-store" });
        if (!response.ok) throw new Error(await response.text());
        const state = await response.json();
        failures = 0;
        setAnalysisProgress(state.progress || 0, state.message || "Analysing");
        if (state.status === "done") {
          setAnalysisProgress(100, "Analysis complete");
          applyAnalysisResult(state.result || {});
          analyseButton.disabled = false;
          return;
        }
        if (state.status === "error") throw new Error(state.error || state.message || "Analysis failed.");
      } catch (error) {
        failures += 1;
        analysisProgressLabel.textContent = `Server analysis continues · reconnecting${failures > 1 ? ` (${failures})` : ""}`;
        if (failures >= 12) {
          analysisResult.textContent = `Could not reconnect to analysis: ${error.message}`;
          analyseButton.disabled = false;
          return;
        }
      }
      await new Promise(resolve => setTimeout(resolve, Math.min(3000, 650 + failures * 250)));
    }
  }

  async function restoreAnalysis(jobId) {
    if (!jobId) return;
    try {
      const response = await fetch(`/analysis?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
      if (!response.ok) return;
      const state = await response.json();
      setAnalysisProgress(state.progress || 0, state.message || "Analysis");
      if (state.status === "done" && state.result) {
        applyAnalysisResult(state.result);
        return;
      }
      if (["queued", "running"].includes(state.status) && state.id) {
        analysisResult.textContent = "Reconnected to the server-side analysis. You can leave this page while it continues.";
        monitorAnalysis(jobId, state.id);
      }
    } catch {}
  }

  async function analyse() {
    if (!frames.length) {
      analysisResult.textContent = "Add frames before analysing the sequence.";
      return;
    }
    if (!smartMissing.checked && !smartReduction.checked) {
      analysisResult.textContent = "No automatic analysis is enabled. Manual target gaps can be generated directly without analysis.";
      return;
    }
    analyseButton.disabled = true;
    analysisResult.textContent = "Uploading a durable analysis snapshot to the server…";
    setAnalysisProgress(0, "Uploading analysis frames");
    try {
      syncTargetGaps(); // Must happen before FormData is constructed.
      const data = new FormData(formNode);
      data.delete("frames");
      frames.forEach(item => data.append("frames", item.file, item.file.name));
      const expected = frames.reduce((sum, item) => sum + Number(item.file?.size || 0), 0);
      const created = await xhrJsonUpload("/analyse", data, expected, "Uploading analysis frames");
      analysisResult.textContent = "Analysis is running independently on the server. You can leave this page and return later.";
      await monitorAnalysis(created.job_id || created.global_job_id, created.analysis_id);
    } catch (error) {
      lastAnalysis = null;
      analysisResult.textContent = `Analysis failed: ${error.message}`;
      analysisProgressLabel.textContent = "Failed";
      applyReduction.disabled = true;
      selectMissingGaps.disabled = true;
      analyseButton.disabled = false;
    }
  }
  analyseButton.addEventListener("click", analyse);

  applyReduction.addEventListener("click", () => {
    if (!lastAnalysis?.reduction_removed?.length) return;
    reductionUndo = frames.slice();
    const removed = new Set(lastAnalysis.reduction_removed.map(value => Number(value) - 1));
    frames = frames.filter((_, index) => !removed.has(index));
    renderFrames();
    undoReduction.disabled = false;
    applyReduction.disabled = true;
    analysisResult.textContent += `\nApplied reduction locally. ${frames.length} frames remain.`;
  });

  undoReduction.addEventListener("click", () => {
    if (!reductionUndo) return;
    frames = reductionUndo;
    reductionUndo = null;
    renderFrames();
    undoReduction.disabled = true;
    analysisResult.textContent += "\nRestored the pre-reduction frame set.";
  });

  selectMissingGaps.addEventListener("click", () => {
    const wanted = new Set([...targetGapSelect.selectedOptions].map(item => item.value));
    for (const item of lastAnalysis?.missing_gaps || []) {
      const key = String(item.gap_key ?? item.gap_index ?? "");
      if (key) wanted.add(key);
    }
    [...targetGapSelect.options].forEach(item => { item.selected = wanted.has(item.value); });
    syncTargetGaps();
  });

  geometry.addEventListener("change", () => {
    const disabled = geometry.value === "none";
    const movement = formNode.querySelector('select[name="axis"]');
    if (movement) movement.disabled = disabled;
  });

  new MutationObserver(() => {
    refreshGapChoices();
    lastAnalysis = null;
    applyReduction.disabled = true;
    selectMissingGaps.disabled = true;
    scheduleFrameCache();
  }).observe(frameStripNode, { childList: true });

  // Dedicated source-frame cache. It never replaces a non-empty snapshot with an
  // empty one, so a metadata-only restore cannot erase the last usable browser copy.
  const FRAME_DB = "webp-animator-source-cache-v2";
  const FRAME_STORE = "jobs";
  let cacheTimer = null;

  function openFrameDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(FRAME_DB, 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(FRAME_STORE)) {
          request.result.createObjectStore(FRAME_STORE, { keyPath: "id" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Could not open frame cache."));
    });
  }

  async function cachePut(value) {
    const db = await openFrameDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(FRAME_STORE, "readwrite");
      tx.objectStore(FRAME_STORE).put(value);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error("Could not save frame cache."));
      tx.onabort = () => reject(tx.error || new Error("Frame cache write aborted."));
    });
    db.close();
  }

  async function cacheGet(id) {
    const db = await openFrameDb();
    const value = await new Promise((resolve, reject) => {
      const tx = db.transaction(FRAME_STORE, "readonly");
      const request = tx.objectStore(FRAME_STORE).get(id);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("Could not read frame cache."));
    });
    db.close();
    return value;
  }

  async function persistFrames() {
    if (!frames.length) return;
    const id = window.webpAnimatorWorkspace?.currentId?.();
    if (!id) return;
    await cachePut({
      id,
      updated: Date.now(),
      frames: frames.map(item => ({
        name: item.displayName || item.file.name,
        fileName: item.file.name,
        type: item.file.type || "application/octet-stream",
        lastModified: item.file.lastModified || Date.now(),
        blob: item.file,
      })),
    });
  }

  function scheduleFrameCache() {
    clearTimeout(cacheTimer);
    cacheTimer = setTimeout(() => persistFrames().catch(() => {}), 450);
  }

  async function restoreCachedFrames(id, metadata) {
    if (frames.length) {
      scheduleFrameCache();
      return;
    }
    const cached = await cacheGet(id).catch(() => null);
    const restored = [];
    if (cached?.frames?.length) {
      for (const item of cached.frames) {
        if (!(item.blob instanceof Blob)) continue;
        const file = new File([item.blob], item.fileName || item.name || "frame.png", {
          type: item.type || item.blob.type || "image/png",
          lastModified: Number(item.lastModified || Date.now()),
        });
        restored.push(makeFrameItem(file, item.name || file.name));
      }
    } else if (metadata?.source_count) {
      for (let index = 0; index < Number(metadata.source_count); index += 1) {
        const response = await fetch(`/job/source?id=${encodeURIComponent(id)}&index=${index}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Could not restore source frame ${index + 1}.`);
        const blob = await response.blob();
        const name = metadata.source_names?.[index] || `frame-${index + 1}.png`;
        const file = new File([blob], name, { type: blob.type || "image/png" });
        restored.push(makeFrameItem(file, name));
      }
    }
    if (restored.length) {
      frames = restored;
      renderFrames();
      await persistFrames();
      const jobStatus = document.getElementById("globalJobStatus");
      if (jobStatus) jobStatus.textContent += ` · restored ${restored.length} cached source frames`;
    }
  }

  document.addEventListener("webp-global-job-restored", event => {
    const id = event.detail?.id || window.webpAnimatorWorkspace?.currentId?.();
    if (id) {
      restoreCachedFrames(id, event.detail?.metadata || null).catch(error => {
        const jobStatus = document.getElementById("globalJobStatus");
        if (jobStatus) jobStatus.textContent = `Job metadata restored, but source-frame recovery failed: ${error.message}`;
      });
      restoreAnalysis(id);
    }
  });

  document.addEventListener("webp-global-job-new", event => {
    ++analysisMonitorToken;
    refreshGapChoices();
    analysisProgress.classList.remove("visible");
    analysisResult.textContent = "No analysis run yet.";
    if (event.detail?.id) restoreAnalysis(event.detail.id);
  });

  refreshGapChoices();
  geometry.dispatchEvent(new Event("change"));
  loadEngineStatus();
  scheduleFrameCache();
  setTimeout(() => restoreAnalysis(window.webpAnimatorWorkspace?.currentId?.()), 500);
})();