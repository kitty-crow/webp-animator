(() => {
  const formElement = document.getElementById("form");
  const multiplier = document.getElementById("rifeMultiplier");
  const geometry = document.getElementById("geometryMode");
  const durationInput = document.getElementById("duration");
  const durationHintElement = document.getElementById("durationHint");
  const statusElement = document.getElementById("status");
  const legacyStatus = document.getElementById("rifeStatus");
  if (!formElement || !multiplier || !geometry) return;

  if (!geometry.querySelector('option[value="none"]')) {
    const option = document.createElement("option");
    option.value = "none";
    option.textContent = "None · keep source size and position unchanged";
    geometry.insertBefore(option, geometry.firstChild);
  }

  const oldLabel = multiplier.closest("label.option");
  if (!oldLabel) return;

  const interpolatorLabel = document.createElement("label");
  interpolatorLabel.className = "option";
  interpolatorLabel.innerHTML = `
    Interpolator
    <select id="interpolator" name="interpolator">
      <option value="none">None</option>
      <option value="rife">RIFE</option>
      <option value="amt">AMT</option>
    </select>
    <span class="hint">Conventional frame interpolation. Choose None to leave the frame count unchanged by the interpolator.</span>`;
  oldLabel.parentNode.insertBefore(interpolatorLabel, oldLabel);

  for (const node of oldLabel.childNodes) {
    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
      node.textContent = "\n          Interpolation multiplier\n          ";
      break;
    }
  }
  const oldHint = oldLabel.querySelector(".hint");
  if (oldHint) oldHint.textContent = "Applied after optional frame generation. 1× leaves the interpolator as a no-op.";

  const generatorLabel = document.createElement("label");
  generatorLabel.className = "option";
  generatorLabel.innerHTML = `
    Frame generator
    <select id="frameGenerator" name="frame_generator">
      <option value="none">None</option>
      <option value="eden">EDEN</option>
      <option value="speed">SPEED</option>
    </select>
    <span class="hint">Generates one structural midpoint between each neighbouring source frame before interpolation.</span>`;
  oldLabel.parentNode.insertBefore(generatorLabel, oldLabel.nextSibling);

  const interpolator = document.getElementById("interpolator");
  const frameGenerator = document.getElementById("frameGenerator");
  const readiness = { rife: false, amt: false, eden: false, speed: false };

  function effectiveFactor() {
    const interpolationFactor = interpolator.value === "none" ? 1 : Math.max(1, Number(multiplier.value) || 1);
    const generatorFactor = frameGenerator.value === "none" ? 1 : 2;
    return interpolationFactor * generatorFactor;
  }

  function refreshDurationHint() {
    multiplier.disabled = interpolator.value === "none";
    const sourceMs = Math.max(1, Number(durationInput?.value) || 1);
    const outputMs = Math.max(1, Math.round(sourceMs / effectiveFactor()));
    const fps = (1000 / outputMs).toFixed(outputMs < 100 ? 1 : 2);
    if (durationHintElement) durationHintElement.textContent = `Output: ${outputMs} ms/frame · about ${fps} fps`;
  }

  function statusText(data) {
    return ["rife", "amt", "eden", "speed"]
      .map(name => `${name.toUpperCase()} ${data?.[name]?.ready ? "ready" : "not installed"}`)
      .join(" · ");
  }

  async function checkEngines() {
    try {
      const response = await fetch("/engine-status", { cache: "no-store" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const data = await response.json();
      for (const name of Object.keys(readiness)) readiness[name] = Boolean(data?.[name]?.ready);
      if (legacyStatus) legacyStatus.textContent = statusText(data);
    } catch {
      if (legacyStatus) legacyStatus.textContent = "Could not check interpolation/generator engine status.";
    }
  }

  function selectedMissingEngine() {
    if (interpolator.value !== "none" && Number(multiplier.value) > 1 && !readiness[interpolator.value]) {
      return interpolator.value;
    }
    if (frameGenerator.value !== "none" && !readiness[frameGenerator.value]) {
      return frameGenerator.value;
    }
    return "";
  }

  async function restoreServerFramesIntoBrowser(jobId, metadata) {
    if (!metadata?.source_count) return;
    try {
      if (typeof frames !== "undefined" && frames.length) return;
      const restored = [];
      for (let index = 0; index < Number(metadata.source_count); index += 1) {
        const response = await fetch(`/job/source?id=${encodeURIComponent(jobId)}&index=${index}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`Could not restore source frame ${index + 1}.`);
        const blob = await response.blob();
        const name = metadata.source_names?.[index] || `frame-${index + 1}.png`;
        const file = new File([blob], name, { type: blob.type || "image/png" });
        restored.push(makeFrameItem(file, name));
      }
      frames = restored;
      renderFrames();
      await new Promise(resolve => setTimeout(resolve, 0));
      await window.webpAnimatorWorkspace?.persistNow?.();
    } catch (error) {
      if (statusElement) statusElement.textContent = `Job metadata restored, but source-frame recovery failed: ${error.message}`;
    }
  }

  formElement.addEventListener("submit", event => {
    // The legacy handler only knows about RIFE. Keep it from vetoing AMT/EDEN/SPEED;
    // this capture-phase validator performs the engine-specific check instead.
    try { rifeReady = true; } catch {}
    window.webpAnimatorWorkspace?.persistNow?.();
    const missing = selectedMissingEngine();
    if (!missing) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (statusElement) statusElement.textContent = `${missing.toUpperCase()} is selected but is not installed. Run python setup_${missing}.py on the server first.`;
  }, true);

  document.addEventListener("webp-global-job-restored", event => {
    const detail = event.detail || {};
    restoreServerFramesIntoBrowser(detail.id, detail.metadata);
  });

  const strip = document.getElementById("frameStrip");
  let persistTimer = null;
  if (strip) {
    new MutationObserver(() => {
      clearTimeout(persistTimer);
      persistTimer = setTimeout(() => window.webpAnimatorWorkspace?.persistNow?.(), 100);
    }).observe(strip, { childList: true });
  }

  interpolator.addEventListener("change", refreshDurationHint);
  frameGenerator.addEventListener("change", refreshDurationHint);
  multiplier.addEventListener("change", refreshDurationHint);
  durationInput?.addEventListener("input", refreshDurationHint);

  refreshDurationHint();
  checkEngines();
})();
