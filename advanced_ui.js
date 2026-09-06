(() => {
  const formNode = document.getElementById("form");
  const grid = formNode?.querySelector(".grid");
  const frameStripNode = document.getElementById("frameStrip");
  const geometry = document.getElementById("geometryMode");
  const multiplier = document.getElementById("rifeMultiplier");
  const statusNode = document.getElementById("status");
  if (!formNode || !grid || !frameStripNode || !geometry || !multiplier) return;
  if (formNode.dataset.advancedEngines === "1") return;
  formNode.dataset.advancedEngines = "1";

  const style = document.createElement("style");
  style.textContent = `
    .advanced-engine-details { margin-top: 16px; border: 1px solid color-mix(in srgb, CanvasText 16%, transparent); border-radius: 12px; padding: 12px 14px; }
    .advanced-engine-details > summary { cursor: pointer; font-weight: 750; }
    .advanced-engine-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:14px; }
    .advanced-engine-grid label { display:grid; gap:6px; font-size:.9rem; }
    .advanced-engine-grid .wide { grid-column:1 / -1; }
    .advanced-engine-grid select[multiple] { min-height: 112px; }
    .smart-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .smart-actions button { width:auto; margin:0; padding:8px 11px; border-radius:9px; border:1px solid color-mix(in srgb, CanvasText 18%, transparent); background:Canvas; color:CanvasText; font-weight:700; }
    #smartAnalysisResult { white-space:pre-wrap; font-size:.82rem; opacity:.82; line-height:1.4; }
    @media(max-width:620px){ .advanced-engine-grid { grid-template-columns:1fr; } .advanced-engine-grid .wide { grid-column:auto; } }
  `;
  document.head.appendChild(style);

  if (!geometry.querySelector('option[value="none"]')) {
    const option = document.createElement("option");
    option.value = "none";
    option.textContent = "None · preserve source geometry";
    geometry.insertBefore(option, geometry.firstChild);
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
  interpolatorHint.textContent = "Conventional temporal interpolation. Can run on all gaps, selected gaps, or only smart-missing gaps.";
  interpolatorLabel.append(interpolatorHint);

  const [generatorLabel, frameGenerator] = labelledSelect(
    "Frame generator",
    "frameGenerator",
    "frame_generator",
    [["none", "None"], ["eden", "EDEN"], ["speed", "SPEED"]],
  );
  const generatorHint = document.createElement("span");
  generatorHint.className = "hint";
  generatorHint.textContent = "Generates one structural midpoint for eligible sparse gaps before any remaining interpolation depth.";
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
    if (hint) hint.textContent = "Final recursive density for eligible source gaps. With a generator selected, its midpoint counts as the first level.";
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
        <span class="hint">Remove frames only when their unique temporal contribution is within the selected error threshold.</span>
      </label>
      <label>
        Reduction threshold (%)
        <input id="reductionThreshold" type="number" name="reduction_threshold" value="2" min="0" max="100" step="0.1">
      </label>
      <label>
        <span><input id="smartMissing" type="checkbox" name="smart_missing"> Smart missing-frame finder</span>
        <span class="hint">Only generate/interpolate gaps whose visual-change score exceeds the selected threshold.</span>
      </label>
      <label>
        Missing-frame threshold (%)
        <input id="missingThreshold" type="number" name="missing_threshold" value="12" min="0" max="100" step="0.1">
      </label>
      <label class="wide">
        Target gaps (optional)
        <select id="targetGapSelect" multiple></select>
        <span class="hint">Select one or more specific gaps to process. Leave empty for all gaps. Smart missing filtering, when enabled, is applied inside this selection.</span>
      </label>
      <input id="targetGaps" type="hidden" name="target_gaps" value="">
      <div class="smart-actions wide">
        <button id="analyseSequence" type="button">Analyse sequence</button>
        <button id="applyReduction" type="button" disabled>Apply suggested reduction</button>
        <button id="undoReduction" type="button" disabled>Undo reduction</button>
        <button id="selectMissingGaps" type="button" disabled>Select suggested missing gaps</button>
      </div>
      <div id="smartAnalysisResult" class="wide">No analysis run yet.</div>
    </div>
  `;
  grid.insertAdjacentElement("afterend", details);

  const targetGapSelect = document.getElementById("targetGapSelect");
  const targetGaps = document.getElementById("targetGaps");
  const analyseButton = document.getElementById("analyseSequence");
  const applyReduction = document.getElementById("applyReduction");
  const undoReduction = document.getElementById("undoReduction");
  const selectMissingGaps = document.getElementById("selectMissingGaps");
  const analysisResult = document.getElementById("smartAnalysisResult");
  const smartReduction = document.getElementById("smartReduction");
  const smartMissing = document.getElementById("smartMissing");
  let lastAnalysis = null;
  let reductionUndo = null;
  let engineState = null;

  function refreshGapChoices() {
    const selected = new Set([...targetGapSelect.selectedOptions].map(item => item.value));
    targetGapSelect.innerHTML = "";
    for (let index = 0; index < Math.max(0, frames.length - 1); index += 1) {
      const value = String(index + 1);
      const item = option(value, `Frame ${index + 1} ↔ ${index + 2}`);
      item.selected = selected.has(value);
      targetGapSelect.append(item);
    }
    syncTargetGaps();
  }

  function syncTargetGaps() {
    targetGaps.value = [...targetGapSelect.selectedOptions].map(item => item.value).join(",");
  }
  targetGapSelect.addEventListener("change", syncTargetGaps);

  function setEngineAvailability() {
    if (!engineState) return;
    const selected = interpolator.value;
    try {
      rifeReady = selected !== "rife" || Boolean(engineState.rife?.ready);
    } catch {}
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
  frameGenerator.addEventListener("change", () => {
    if (frameGenerator.value !== "none" && interpolator.value === "none" && multiplier.value === "1") {
      // Generator-only mode intentionally remains one midpoint per eligible gap.
    }
  });

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

  async function analyse() {
    if (!frames.length) {
      analysisResult.textContent = "Add frames before analysing the sequence.";
      return;
    }
    analyseButton.disabled = true;
    analysisResult.textContent = "Analysing geometry and temporal information…";
    try {
      const data = new FormData(formNode);
      data.delete("frames");
      syncTargetGaps();
      frames.forEach(item => data.append("frames", item.file, item.file.name));
      const response = await fetch("/analyse", { method: "POST", body: data });
      if (!response.ok) throw new Error(await response.text());
      lastAnalysis = await response.json();
      const removed = lastAnalysis.reduction_removed || [];
      const missing = lastAnalysis.missing_gaps || [];
      analysisResult.textContent = [
        `Suggested frame count: ${lastAnalysis.source_count} → ${lastAnalysis.suggested_count}`,
        removed.length ? `Redundant source frames: ${removed.join(", ")}` : "No source frames are below the reduction threshold.",
        missing.length
          ? `Sparse gaps: ${missing.map(item => `${item.gap_index} (${item.score.toFixed(2)}%)`).join(", ")}`
          : "No gaps exceed the missing-frame threshold.",
      ].join("\n");
      applyReduction.disabled = removed.length === 0;
      selectMissingGaps.disabled = missing.length === 0;
    } catch (error) {
      lastAnalysis = null;
      analysisResult.textContent = `Analysis failed: ${error.message}`;
      applyReduction.disabled = true;
      selectMissingGaps.disabled = true;
    } finally {
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
    const wanted = new Set((lastAnalysis?.missing_gaps || []).map(item => String(item.gap_index)));
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
    if (id) restoreCachedFrames(id, event.detail?.metadata || null).catch(error => {
      const jobStatus = document.getElementById("globalJobStatus");
      if (jobStatus) jobStatus.textContent = `Job metadata restored, but source-frame recovery failed: ${error.message}`;
    });
  });

  document.addEventListener("webp-global-job-new", refreshGapChoices);
  refreshGapChoices();
  geometry.dispatchEvent(new Event("change"));
  loadEngineStatus();
  scheduleFrameCache();
})();
