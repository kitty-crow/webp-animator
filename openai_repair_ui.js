(() => {
  "use strict";

  const MAX_UNDO = 20;
  const PAN_PIXELS_FOR_FULL_RANGE = 300;
  const masks = new Map();
  let currentItem = null;
  let currentIndex = -1;
  let zoom = 1;
  let posX = 50;
  let posY = 50;
  let mode = "mark";
  let drawing = false;
  let panning = false;
  let lastPoint = null;
  let panPoint = null;
  let undo = [];

  const style = document.createElement("style");
  style.textContent = `
    .frame-detail-modal[hidden] { display:none !important; }
    .frame-detail-modal { position:fixed; inset:0; z-index:1000; display:grid; place-items:center; padding:clamp(8px,2vw,24px); background:color-mix(in srgb, black 72%, transparent); }
    .frame-detail-dialog { width:min(1180px,100%); max-height:calc(100dvh - 16px); display:grid; grid-template-rows:auto minmax(0,1fr) auto; overflow:hidden; border:1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius:16px; background:Canvas; color:CanvasText; box-shadow:0 18px 70px rgb(0 0 0 / .45); }
    .frame-detail-head,.frame-detail-toolbar,.frame-detail-actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:10px 12px; }
    .frame-detail-head { border-bottom:1px solid color-mix(in srgb, CanvasText 14%, transparent); }
    .frame-detail-title { margin-right:auto; min-width:0; font-weight:850; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .frame-detail-toolbar { border-bottom:1px solid color-mix(in srgb, CanvasText 12%, transparent); }
    .frame-detail-toolbar button,.frame-detail-actions button { width:auto; margin:0; padding:7px 10px; border:1px solid color-mix(in srgb, CanvasText 20%, transparent); border-radius:8px; background:Canvas; color:CanvasText; font:inherit; font-weight:750; }
    .frame-detail-toolbar button[aria-pressed="true"] { background:Highlight; color:HighlightText; border-color:Highlight; }
    .frame-detail-toolbar label { display:flex; align-items:center; gap:6px; font-size:.8rem; }
    .frame-detail-toolbar input[type="range"] { width:110px; }
    .frame-detail-viewport { min-height:260px; overflow:hidden; display:grid; place-items:center; position:relative; padding:12px; background-image:linear-gradient(45deg,color-mix(in srgb,CanvasText 8%,Canvas) 25%,transparent 25%),linear-gradient(-45deg,color-mix(in srgb,CanvasText 8%,Canvas) 25%,transparent 25%),linear-gradient(45deg,transparent 75%,color-mix(in srgb,CanvasText 8%,Canvas) 75%),linear-gradient(-45deg,transparent 75%,color-mix(in srgb,CanvasText 8%,Canvas) 75%); background-size:20px 20px; background-position:0 0,0 10px,10px -10px,-10px 0; }
    .frame-detail-stage { position:relative; display:inline-block; max-width:100%; max-height:68dvh; transform-origin:50% 50%; transition:transform .16s ease,transform-origin .08s linear; user-select:none; -webkit-user-select:none; }
    .frame-detail-stage.panning { cursor:move; touch-action:none; }
    .frame-detail-stage.marking { cursor:crosshair; touch-action:none; }
    .frame-detail-image { display:block; width:auto; height:auto; max-width:100%; max-height:68dvh; object-fit:contain; pointer-events:none; -webkit-user-drag:none; }
    .frame-detail-mask { position:absolute; inset:0; width:100%; height:100%; pointer-events:auto; }
    .frame-detail-mask.pan-mode { cursor:move; }
    .frame-detail-status { flex:1 1 260px; font-size:.8rem; opacity:.78; line-height:1.35; }
    .frame-detail-actions { border-top:1px solid color-mix(in srgb, CanvasText 14%, transparent); }
    .frame-detail-actions .primary { background:Highlight; color:HighlightText; border-color:Highlight; }
    .frame-detail-close { font-size:1.1rem !important; padding:5px 9px !important; }
    .thumb[data-frame-detail-ready="1"] { cursor:zoom-in; }
    @media (max-width:620px) {
      .frame-detail-modal { padding:0; }
      .frame-detail-dialog { width:100%; height:100dvh; max-height:100dvh; border-radius:0; border-inline:0; }
      .frame-detail-stage,.frame-detail-image { max-height:62dvh; }
      .frame-detail-toolbar { gap:6px; }
    }
  `;
  document.head.appendChild(style);

  const modal = document.createElement("div");
  modal.className = "frame-detail-modal";
  modal.hidden = true;
  modal.innerHTML = `
    <section class="frame-detail-dialog" role="dialog" aria-modal="true" aria-labelledby="frameDetailTitle">
      <div>
        <div class="frame-detail-head">
          <div class="frame-detail-title" id="frameDetailTitle">Frame detail</div>
          <button type="button" class="frame-detail-close" data-detail-close aria-label="Close frame detail">×</button>
        </div>
        <div class="frame-detail-toolbar">
          <button type="button" data-mode="mark" aria-pressed="true">Mark defects</button>
          <button type="button" data-mode="pan" aria-pressed="false">Pan</button>
          <button type="button" data-zoom-out aria-label="Zoom out">−</button>
          <strong data-zoom-label>100%</strong>
          <button type="button" data-zoom-in aria-label="Zoom in">+</button>
          <button type="button" data-reset-view>Reset view</button>
          <label>Brush <input type="range" data-brush min="2" max="120" value="28"></label>
          <button type="button" data-undo>Undo</button>
          <button type="button" data-clear>Clear marks</button>
        </div>
      </div>
      <div class="frame-detail-viewport">
        <div class="frame-detail-stage marking" data-stage>
          <img class="frame-detail-image" data-detail-image alt="Selected animation frame">
          <canvas class="frame-detail-mask" data-mask aria-label="Defect marking canvas"></canvas>
        </div>
      </div>
      <div class="frame-detail-actions">
        <div class="frame-detail-status" data-detail-status>Paint over a defect to tell OpenAI where to concentrate the repair. Transparency is preserved from this candidate frame.</div>
        <button type="button" data-detail-close>Close</button>
        <button type="button" class="primary" data-repair>Repair with OpenAI</button>
      </div>
    </section>`;
  document.body.appendChild(modal);

  const stage = modal.querySelector("[data-stage]");
  const image = modal.querySelector("[data-detail-image]");
  const canvas = modal.querySelector("[data-mask]");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const status = modal.querySelector("[data-detail-status]");
  const title = modal.querySelector("#frameDetailTitle");
  const repairButton = modal.querySelector("[data-repair]");
  const brush = modal.querySelector("[data-brush]");
  const zoomLabel = modal.querySelector("[data-zoom-label]");
  const markButton = modal.querySelector('[data-mode="mark"]');
  const panButton = modal.querySelector('[data-mode="pan"]');

  function frameList() {
    try { return Array.isArray(frames) ? frames : []; }
    catch { return []; }
  }

  function setStatus(text) {
    status.textContent = String(text || "");
  }

  function updateView() {
    stage.style.transformOrigin = `${posX}% ${posY}%`;
    stage.style.transform = `scale(${zoom})`;
    zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
    stage.classList.toggle("marking", mode === "mark");
    stage.classList.toggle("panning", mode === "pan");
    canvas.classList.toggle("pan-mode", mode === "pan");
    markButton.setAttribute("aria-pressed", String(mode === "mark"));
    panButton.setAttribute("aria-pressed", String(mode === "pan"));
  }

  function setMode(next) {
    mode = next === "pan" ? "pan" : "mark";
    drawing = false;
    panning = false;
    updateView();
  }

  function resetView() {
    zoom = 1;
    posX = 50;
    posY = 50;
    updateView();
  }

  function setZoom(value) {
    zoom = Math.max(1, Math.min(5, Math.round(Number(value) * 4) / 4));
    updateView();
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: Math.max(0, Math.min(canvas.width, ((event.clientX - rect.left) / rect.width) * canvas.width)),
      y: Math.max(0, Math.min(canvas.height, ((event.clientY - rect.top) / rect.height) * canvas.height)),
    };
  }

  function saveUndo() {
    if (!canvas.width || !canvas.height) return;
    undo.push(context.getImageData(0, 0, canvas.width, canvas.height));
    if (undo.length > MAX_UNDO) undo.shift();
  }

  function drawSegment(from, to) {
    if (!from || !to) return;
    const scale = image.naturalWidth && image.getBoundingClientRect().width
      ? image.naturalWidth / image.getBoundingClientRect().width
      : 1;
    context.save();
    context.strokeStyle = "rgba(255, 48, 80, .72)";
    context.fillStyle = "rgba(255, 48, 80, .72)";
    context.lineWidth = Math.max(1, Number(brush.value || 28) * scale / Math.max(1, zoom));
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  }

  function currentMaskBlob() {
    return new Promise(resolve => canvas.toBlob(resolve, "image/png"));
  }

  function saveCurrentMask() {
    if (!currentItem) return;
    canvas.toBlob(blob => {
      if (!blob || !currentItem) return;
      masks.set(currentItem.id, blob);
    }, "image/png");
  }

  function loadMask(item) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    undo = [];
    const blob = masks.get(item.id);
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const saved = new Image();
    saved.onload = () => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(saved, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
    };
    saved.onerror = () => URL.revokeObjectURL(url);
    saved.src = url;
  }

  function hasMarkedPixels() {
    if (!canvas.width || !canvas.height) return false;
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let offset = 3; offset < pixels.length; offset += 4) {
      if (pixels[offset] > 0) return true;
    }
    return false;
  }

  function openFrame(item, index) {
    currentItem = item;
    currentIndex = index;
    resetView();
    setMode("mark");
    title.textContent = `Frame ${index + 1} · ${item.displayName || item.file?.name || "image"}`;
    repairButton.disabled = index <= 0 || index >= frameList().length - 1;
    setStatus(repairButton.disabled
      ? "OpenAI repair needs both a previous and a next frame. This edge frame can still be inspected and marked."
      : "Paint over a defect to tell OpenAI exactly where to concentrate the repair. With no marks, the whole visible candidate is eligible for repair. Candidate alpha is always preserved.");
    modal.hidden = false;
    document.documentElement.style.overflow = "hidden";
    image.onload = () => {
      canvas.width = Math.max(1, image.naturalWidth);
      canvas.height = Math.max(1, image.naturalHeight);
      loadMask(item);
    };
    image.src = item.url;
    modal.querySelector("[data-detail-close]").focus();
  }

  function closeModal() {
    if (modal.hidden) return;
    saveCurrentMask();
    modal.hidden = true;
    document.documentElement.style.overflow = "";
    image.removeAttribute("src");
    currentItem = null;
    currentIndex = -1;
    drawing = false;
    panning = false;
  }

  async function repairCurrent() {
    const list = frameList();
    if (!currentItem || currentIndex <= 0 || currentIndex >= list.length - 1) return;
    const previous = list[currentIndex - 1];
    const candidate = list[currentIndex];
    const following = list[currentIndex + 1];
    if (!previous?.file || !candidate?.file || !following?.file) return;

    repairButton.disabled = true;
    setStatus("Uploading previous + candidate + next frame for constrained OpenAI repair…");
    try {
      const data = new FormData();
      data.append("previous", previous.file, previous.file.name || "previous.png");
      data.append("candidate", candidate.file, candidate.file.name || "candidate.png");
      data.append("following", following.file, following.file.name || "following.png");
      if (hasMarkedPixels()) {
        const mask = await currentMaskBlob();
        if (mask) data.append("mask", mask, "defect-mask.png");
      }
      const response = await fetch("/openai-repair", { method: "POST", body: data, cache: "no-store" });
      const text = await response.text();
      let value;
      try { value = JSON.parse(text); }
      catch { value = { error: text || `HTTP ${response.status}` }; }
      if (!response.ok) throw new Error(value.error || `OpenAI repair failed (${response.status})`);

      const binary = atob(value.image_b64 || "");
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      const oldName = candidate.displayName || candidate.file.name || `frame-${currentIndex + 1}.png`;
      const stem = oldName.replace(/\.[^.]+$/, "");
      const file = new File([bytes], `${stem}-openai-repaired.png`, { type: "image/png" });
      URL.revokeObjectURL(candidate.url);
      candidate.file = file;
      candidate.displayName = file.name;
      candidate.url = URL.createObjectURL(file);
      masks.delete(candidate.id);
      if (typeof renderFrames === "function") renderFrames();
      image.src = candidate.url;
      setStatus(`OpenAI repair complete. ${value.stats?.alpha_policy === "candidate-alpha-authoritative" ? "Original candidate transparency was preserved exactly." : ""}`.trim());
    } catch (error) {
      setStatus(`Repair failed: ${error.message}`);
    } finally {
      repairButton.disabled = currentIndex <= 0 || currentIndex >= frameList().length - 1;
    }
  }

  function decorateThumbnails() {
    const list = frameList();
    document.querySelectorAll("#frameStrip .frame-card .thumb").forEach((thumb, index) => {
      if (thumb.dataset.frameDetailReady === "1") return;
      thumb.dataset.frameDetailReady = "1";
      thumb.tabIndex = 0;
      thumb.setAttribute("role", "button");
      thumb.setAttribute("aria-label", `${thumb.alt || `Frame ${index + 1}`} · open detailed repair view`);
      const activate = event => {
        event.preventDefault();
        event.stopPropagation();
        const current = frameList();
        const item = current[index];
        if (item) openFrame(item, index);
      };
      thumb.addEventListener("click", activate);
      thumb.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") activate(event);
      });
    });
  }

  canvas.addEventListener("pointerdown", event => {
    if (!currentItem) return;
    canvas.setPointerCapture?.(event.pointerId);
    if (mode === "pan") {
      panning = true;
      panPoint = { x: event.clientX, y: event.clientY };
      event.preventDefault();
      return;
    }
    const point = canvasPoint(event);
    if (!point) return;
    saveUndo();
    drawing = true;
    lastPoint = point;
    drawSegment(point, point);
    event.preventDefault();
  });

  canvas.addEventListener("pointermove", event => {
    if (panning && mode === "pan" && panPoint) {
      const deltaX = event.clientX - panPoint.x;
      const deltaY = event.clientY - panPoint.y;
      posX = Math.max(0, Math.min(100, posX - (deltaX / PAN_PIXELS_FOR_FULL_RANGE) * 100));
      posY = Math.max(0, Math.min(100, posY - (deltaY / PAN_PIXELS_FOR_FULL_RANGE) * 100));
      panPoint = { x: event.clientX, y: event.clientY };
      updateView();
      event.preventDefault();
      return;
    }
    if (!drawing || mode !== "mark") return;
    const point = canvasPoint(event);
    if (!point) return;
    drawSegment(lastPoint, point);
    lastPoint = point;
    event.preventDefault();
  });

  function endPointer(event) {
    if (drawing) saveCurrentMask();
    drawing = false;
    panning = false;
    lastPoint = null;
    panPoint = null;
    try { canvas.releasePointerCapture?.(event.pointerId); } catch {}
  }
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  canvas.addEventListener("wheel", event => {
    if (!currentItem) return;
    setZoom(zoom + (event.deltaY < 0 ? .5 : -.5));
    event.preventDefault();
  }, { passive: false });

  modal.querySelectorAll("[data-detail-close]").forEach(button => button.addEventListener("click", closeModal));
  modal.addEventListener("click", event => { if (event.target === modal) closeModal(); });
  document.addEventListener("keydown", event => { if (!modal.hidden && event.key === "Escape") closeModal(); });
  markButton.addEventListener("click", () => setMode("mark"));
  panButton.addEventListener("click", () => setMode("pan"));
  modal.querySelector("[data-zoom-in]").addEventListener("click", () => setZoom(zoom + .5));
  modal.querySelector("[data-zoom-out]").addEventListener("click", () => setZoom(zoom - .5));
  modal.querySelector("[data-reset-view]").addEventListener("click", resetView);
  modal.querySelector("[data-undo]").addEventListener("click", () => {
    const previous = undo.pop();
    if (!previous) return;
    context.putImageData(previous, 0, 0);
    saveCurrentMask();
  });
  modal.querySelector("[data-clear]").addEventListener("click", () => {
    if (!canvas.width || !canvas.height) return;
    saveUndo();
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (currentItem) masks.delete(currentItem.id);
  });
  repairButton.addEventListener("click", repairCurrent);

  const frameStrip = document.getElementById("frameStrip");
  if (frameStrip) new MutationObserver(decorateThumbnails).observe(frameStrip, { childList: true, subtree: true });
  window.addEventListener("DOMContentLoaded", decorateThumbnails, { once: true });
  decorateThumbnails();
})();
