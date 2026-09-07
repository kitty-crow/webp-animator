(() => {
  const progressWrap = document.getElementById("progressWrap");
  if (!progressWrap || document.getElementById("liveRunPanel")) return;

  const style = document.createElement("style");
  style.textContent = `
    .live-run-panel {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid color-mix(in srgb, CanvasText 16%, transparent);
      border-radius: 12px;
      background: color-mix(in srgb, CanvasText 2.5%, Canvas);
    }
    .live-run-head { display:flex; align-items:center; flex-wrap:wrap; gap:8px; }
    .live-run-head strong { margin-right:auto; }
    .live-stop-button {
      width:auto !important;
      margin:0 !important;
      padding:8px 12px !important;
      border-radius:9px !important;
      border:1px solid color-mix(in srgb, #b42318 48%, CanvasText) !important;
      background:color-mix(in srgb, #b42318 10%, Canvas) !important;
      color:CanvasText !important;
      font-weight:800 !important;
    }
    .live-stop-button:disabled { opacity:.45; cursor:not-allowed; }
    .live-run-note { margin-top:6px; font-size:.8rem; opacity:.72; line-height:1.35; }
    .live-frame-grid {
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(138px,1fr));
      gap:10px;
      margin-top:10px;
    }
    .live-frame-empty { grid-column:1 / -1; padding:12px 0; font-size:.82rem; opacity:.62; }
    .live-frame-card {
      min-width:0;
      padding:8px;
      border:1px solid color-mix(in srgb, CanvasText 13%, transparent);
      border-radius:10px;
      background:Canvas;
    }
    .live-frame-card img {
      display:block;
      width:100%;
      aspect-ratio:1 / 1;
      object-fit:contain;
      border-radius:7px;
      background-image:
        linear-gradient(45deg, color-mix(in srgb, CanvasText 8%, Canvas) 25%, transparent 25%),
        linear-gradient(-45deg, color-mix(in srgb, CanvasText 8%, Canvas) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, color-mix(in srgb, CanvasText 8%, Canvas) 75%),
        linear-gradient(-45deg, transparent 75%, color-mix(in srgb, CanvasText 8%, Canvas) 75%);
      background-size:14px 14px;
      background-position:0 0, 0 7px, 7px -7px, -7px 0;
    }
    .live-frame-stage { margin-top:7px; font-size:.72rem; font-weight:800; overflow-wrap:anywhere; }
    .live-frame-name { margin-top:2px; font:600 .68rem ui-monospace, SFMono-Regular, Menlo, monospace; opacity:.64; overflow-wrap:anywhere; }
    .live-frame-card a { display:inline-block; margin-top:6px; font-size:.72rem; font-weight:750; }
    @media(max-width:620px) { .live-frame-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  `;
  document.head.appendChild(style);

  const panel = document.createElement("section");
  panel.id = "liveRunPanel";
  panel.className = "live-run-panel";
  panel.innerHTML = `
    <div class="live-run-head">
      <strong>Live animation timeline</strong>
      <button id="liveStopButton" class="live-stop-button" type="button" disabled>Stop</button>
    </div>
    <div id="liveRunNote" class="live-run-note">Original frames and newly created frames will appear here in timeline order while the run is working.</div>
    <div id="liveFrameGrid" class="live-frame-grid">
      <div class="live-frame-empty">No timeline frames available yet.</div>
    </div>
  `;
  progressWrap.insertAdjacentElement("afterend", panel);

  const stopButton = document.getElementById("liveStopButton");
  const note = document.getElementById("liveRunNote");
  const grid = document.getElementById("liveFrameGrid");
  const ACTIVE_JOB_KEY = "webp-animator-active-job";

  let watchedJobId = "";
  let watchedRenderCount = null;
  let cards = new Map();
  let stoppedLocally = false;
  let polling = false;

  function activeJobId() {
    try { return localStorage.getItem(ACTIVE_JOB_KEY) || ""; }
    catch { return ""; }
  }

  function showEmpty() {
    grid.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "live-frame-empty";
    empty.textContent = "No timeline frames available yet.";
    grid.appendChild(empty);
  }

  function clearCards() {
    cards.clear();
    showEmpty();
  }

  function setJob(jobId, renderCount = null) {
    const changedJob = jobId && jobId !== watchedJobId;
    const changedRender = renderCount !== null && watchedRenderCount !== null && renderCount !== watchedRenderCount;
    if (changedJob || changedRender) clearCards();
    if (jobId) watchedJobId = jobId;
    if (renderCount !== null) watchedRenderCount = renderCount;
    if (changedJob || changedRender) stoppedLocally = false;
  }

  function makeCard(item) {
    const card = document.createElement("article");
    card.className = "live-frame-card";

    const image = document.createElement("img");
    image.alt = `${item.stage || "Frame"} ${item.name || "frame"}`;
    image.loading = "lazy";
    image.decoding = "async";

    const stage = document.createElement("div");
    stage.className = "live-frame-stage";

    const name = document.createElement("div");
    name.className = "live-frame-name";

    const download = document.createElement("a");
    download.textContent = "Download frame";
    download.download = item.name || "frame.png";

    card.append(image, stage, name, download);
    return { card, image, stage, name, download, version: null };
  }

  function renderFrames(items) {
    if (!Array.isArray(items)) return;
    if (!items.length) {
      cards.clear();
      showEmpty();
      return;
    }

    const nextCards = new Map();
    const orderedNodes = [];

    for (const item of items) {
      const key = String(item.key || "");
      if (!key) continue;
      let entry = cards.get(key);
      if (!entry) entry = makeCard(item);

      const version = String(item.mtime_ns || "0");
      if (entry.version !== version) {
        entry.version = version;
        entry.image.src = item.url;
        entry.download.href = item.url;
      }
      entry.image.alt = `${item.stage || "Frame"} ${item.name || "frame"}`;
      entry.stage.textContent = String(item.stage || "Frame");
      entry.name.textContent = String(item.name || key);
      entry.download.download = String(item.name || "frame.png");

      nextCards.set(key, entry);
      orderedNodes.push(entry.card);
    }

    cards = nextCards;
    grid.replaceChildren(...orderedNodes);
  }

  async function fetchLive(jobId) {
    const response = await fetch(`/live-frames?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Live frame check failed (${response.status})`);
    return await response.json();
  }

  async function fetchProgress(jobId) {
    const response = await fetch(`/progress?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Progress check failed (${response.status})`);
    return await response.json();
  }

  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const active = activeJobId();
      const candidate = active || watchedJobId;
      if (!candidate) {
        stopButton.disabled = true;
        note.textContent = "Original frames and newly created frames will appear here in timeline order while the run is working.";
        return;
      }

      const [progressResult, liveResult] = await Promise.allSettled([
        fetchProgress(candidate),
        fetchLive(candidate),
      ]);

      if (liveResult.status === "fulfilled") {
        const live = liveResult.value;
        setJob(candidate, Number(live.render_count || 0));
        renderFrames(live.frames || []);
      } else if (!watchedJobId) {
        setJob(candidate, null);
      }

      if (progressResult.status === "fulfilled") {
        const progress = progressResult.value;
        const state = String(progress.status || "draft");
        const running = state === "queued" || state === "running";
        stopButton.disabled = !running || stoppedLocally;
        stopButton.textContent = stoppedLocally ? "Stopping…" : "Stop";
        const count = cards.size;
        if (state === "cancelled") {
          note.textContent = `Stopped. Current timeline has ${count} frame${count === 1 ? "" : "s"} available for inspection.`;
        } else if (running) {
          note.textContent = `Current timeline: ${count} frame${count === 1 ? "" : "s"} · ${progress.message || "processing"}`;
        } else if (state === "done") {
          note.textContent = `Completed timeline: ${count} frame${count === 1 ? "" : "s"}.`;
        } else if (state === "error") {
          note.textContent = `Run failed. Last available timeline has ${count} frame${count === 1 ? "" : "s"}.`;
        }
      }
    } catch {
      // The normal job monitor owns connectivity/error messaging. Keep this panel quiet.
    } finally {
      polling = false;
    }
  }

  stopButton.addEventListener("click", async () => {
    const jobId = activeJobId() || watchedJobId;
    if (!jobId || stopButton.disabled) return;
    stoppedLocally = true;
    stopButton.disabled = true;
    stopButton.textContent = "Stopping…";
    note.textContent = "Stopping the active engine and keeping the current timeline for inspection…";
    try {
      const response = await fetch(`/job/stop?id=${encodeURIComponent(jobId)}`, {
        method: "POST",
        cache: "no-store",
      });
      if (!response.ok) throw new Error((await response.text()) || `Stop failed (${response.status})`);

      try { monitorGeneration += 1; } catch {}
      try { clearActiveJob(jobId); } catch {
        try { localStorage.removeItem(ACTIVE_JOB_KEY); } catch {}
      }
      try {
        button.disabled = false;
        button.textContent = "Generate WebP";
      } catch {}
      try {
        status.textContent = "Stopped by user. The current live timeline is kept below for inspection.";
        progressLabel.textContent = "Stopped";
      } catch {}
      note.textContent = `Stopped. Current timeline has ${cards.size} frame${cards.size === 1 ? "" : "s"} available for inspection.`;
      stopButton.textContent = "Stop";
    } catch (error) {
      stoppedLocally = false;
      stopButton.disabled = false;
      stopButton.textContent = "Stop";
      note.textContent = `Could not stop the run: ${error.message}`;
    }
    await poll();
  });

  clearCards();
  poll();
  setInterval(poll, 750);
})();
