(() => {
  const panel = document.getElementById("workspacePanel");
  if (!panel || panel.dataset.initialised === "1") return;
  panel.dataset.initialised = "1";

  const $ = id => document.getElementById(id);
  const idLabel = $("globalJobId");
  const copyButton = $("globalCopyJob");
  const newButton = $("globalNewJob");
  const restoreInput = $("globalRestoreId");
  const restoreButton = $("globalRestore");
  const jobStatus = $("globalJobStatus");
  const resultPanel = $("globalResult");
  const resultPreview = $("globalResultPreview");
  const resultInfo = $("globalResultInfo");
  const resultDownload = $("globalResultDownload");
  const continueButton = $("globalContinueJob");

  const CURRENT_KEY = "webp-animator-global-job-v1";
  const DB_NAME = "webp-animator-jobs";
  const DB_VERSION = 1;
  const STORE = "jobs";
  let globalJobId = "";
  let resultObjectUrl = "";
  let resultBlob = null;
  let saveTimer = null;
  let restoring = false;

  function randomJobId() {
    if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "").toLowerCase();
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return [...bytes].map(value => value.toString(16).padStart(2, "0")).join("");
  }

  function validJobId(value) {
    return /^[0-9a-f]{32}$/i.test(String(value || "").trim());
  }

  function currentId() {
    return globalJobId;
  }

  function setCurrentId(value) {
    globalJobId = String(value || "").trim().toLowerCase();
    idLabel.textContent = `Global Job ID: ${globalJobId}`;
    restoreInput.value = globalJobId;
    let hidden = document.querySelector('input[name="global_job_id"]');
    if (!hidden) {
      hidden = document.createElement("input");
      hidden.type = "hidden";
      hidden.name = "global_job_id";
      form.appendChild(hidden);
    }
    hidden.value = globalJobId;
    try { localStorage.setItem(CURRENT_KEY, globalJobId); } catch {}
  }

  function openDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "id" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Could not open IndexedDB."));
    });
  }

  async function dbGet(id) {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const request = tx.objectStore(STORE).get(id);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("Could not read saved job."));
      tx.oncomplete = () => db.close();
    });
  }

  async function dbPut(value) {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(value);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { const error = tx.error || new Error("Could not save job in IndexedDB."); db.close(); reject(error); };
      tx.onabort = () => { const error = tx.error || new Error("IndexedDB save was aborted."); db.close(); reject(error); };
    });
  }

  function serialiseSettings() {
    const values = {};
    document.querySelectorAll("input, select, textarea").forEach(control => {
      if (control.type === "file" || control.id === "globalRestoreId") return;
      const key = control.id ? `id:${control.id}` : control.name ? `name:${control.name}` : "";
      if (!key) return;
      if (control.type === "checkbox" || control.type === "radio") values[key] = Boolean(control.checked);
      else values[key] = control.value;
    });
    return values;
  }

  function restoreSettings(values) {
    if (!values || typeof values !== "object") return;
    for (const [key, value] of Object.entries(values)) {
      const control = key.startsWith("id:")
        ? document.getElementById(key.slice(3))
        : key.startsWith("name:")
          ? document.querySelector(`[name="${CSS.escape(key.slice(5))}"]`)
          : null;
      if (!control || control.type === "file") continue;
      if (control.type === "checkbox" || control.type === "radio") control.checked = Boolean(value);
      else control.value = String(value ?? "");
      control.dispatchEvent(new Event("change", { bubbles: true }));
    }
    try { updateDurationHint(); } catch {}
  }

  function releaseFrames() {
    try {
      frames.forEach(item => {
        if (item?.url) URL.revokeObjectURL(item.url);
      });
      frames = [];
      renderFrames();
    } catch {}
  }

  function clearResultView() {
    if (resultObjectUrl) URL.revokeObjectURL(resultObjectUrl);
    resultObjectUrl = "";
    resultBlob = null;
    resultPreview.removeAttribute("src");
    resultDownload.removeAttribute("href");
    resultPanel.hidden = true;
    try {
      downloadLink.classList.remove("visible");
      downloadLink.removeAttribute("href");
    } catch {}
  }

  function showResultBlob(blob, source = "saved") {
    clearResultView();
    resultBlob = blob;
    resultObjectUrl = URL.createObjectURL(blob);
    resultPreview.src = resultObjectUrl;
    resultDownload.href = resultObjectUrl;
    resultPanel.hidden = false;
    resultInfo.textContent = `${(blob.size / 1024 / 1024).toFixed(2)} MB · ${source === "indexeddb" ? "cached in this browser" : source === "server" ? "restored from server" : "saved result"}`;
    try {
      downloadLink.href = resultObjectUrl;
      downloadLink.classList.add("visible");
    } catch {}
  }

  async function snapshot(outputBlob = undefined) {
    if (!globalJobId || restoring) return;
    let previous = null;
    try { previous = await dbGet(globalJobId); } catch {}
    const savedFrames = [];
    try {
      for (const item of frames) {
        savedFrames.push({
          name: item.displayName || item.file.name,
          fileName: item.file.name,
          type: item.file.type || "application/octet-stream",
          lastModified: item.file.lastModified || Date.now(),
          blob: item.file,
        });
      }
    } catch {}
    const value = {
      id: globalJobId,
      created: previous?.created || Date.now(),
      updated: Date.now(),
      settings: serialiseSettings(),
      frames: savedFrames,
      outputBlob: outputBlob === undefined ? previous?.outputBlob || null : outputBlob,
      outputName: "animation.webp",
    };
    try {
      await dbPut(value);
      jobStatus.textContent = value.outputBlob
        ? "Job saved locally, including the finished WebP. You can close the page and restore it by ID."
        : "Job saved locally. The server also persists the source frames once a processing job is submitted.";
    } catch (error) {
      jobStatus.textContent = `Job ID is active, but browser storage could not save this snapshot: ${error.message}`;
    }
  }

  function scheduleSnapshot() {
    if (restoring) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => snapshot(), 350);
  }

  async function ensureServerJob(id) {
    try {
      await fetch("/job/new", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: id }),
      });
    } catch {}
  }

  async function newJob() {
    if (globalJobId) await snapshot();
    ++monitorGeneration;
    try { clearActiveJob(); } catch {}
    releaseFrames();
    clearResultView();
    try {
      status.textContent = "";
      progressWrap.classList.remove("visible");
    } catch {}
    const id = randomJobId();
    setCurrentId(id);
    await ensureServerJob(id);
    await snapshot(null);
    jobStatus.textContent = "New recoverable job created. Previous Job IDs remain stored and can be restored later.";
    document.dispatchEvent(new CustomEvent("webp-global-job-new", { detail: { id } }));
  }

  async function fetchServerJob(id) {
    const response = await fetch(`/job?id=${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!response.ok) return null;
    return await response.json();
  }

  async function restoreFramesFromServer(id, metadata) {
    const restored = [];
    for (let index = 0; index < Number(metadata.source_count || 0); index += 1) {
      const response = await fetch(`/job/source?id=${encodeURIComponent(id)}&index=${index}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Could not restore source frame ${index + 1}.`);
      const blob = await response.blob();
      const name = metadata.source_names?.[index] || `frame-${index + 1}.png`;
      const file = new File([blob], name, { type: blob.type || "image/png" });
      restored.push(makeFrameItem(file, name));
    }
    frames = restored;
    renderFrames();
  }

  async function restoreFromLocal(saved) {
    releaseFrames();
    const restored = [];
    for (const item of saved.frames || []) {
      const blob = item.blob;
      if (!(blob instanceof Blob)) continue;
      const file = new File([blob], item.fileName || item.name || "frame.png", {
        type: item.type || blob.type || "image/png",
        lastModified: Number(item.lastModified || Date.now()),
      });
      restored.push(makeFrameItem(file, item.name || file.name));
    }
    frames = restored;
    restoreSettings(saved.settings);
    renderFrames();
    if (saved.outputBlob instanceof Blob && saved.outputBlob.size) showResultBlob(saved.outputBlob, "indexeddb");
  }

  async function restoreJob(requestedId = "") {
    const id = String(requestedId || restoreInput.value || "").trim().toLowerCase();
    if (!validJobId(id)) {
      jobStatus.textContent = "Enter a valid 32-character global Job ID.";
      return;
    }
    restoring = true;
    try {
      ++monitorGeneration;
      setCurrentId(id);
      clearResultView();
      jobStatus.textContent = `Restoring ${id}…`;

      let local = null;
      try { local = await dbGet(id); } catch {}
      let server = null;
      try { server = await fetchServerJob(id); } catch {}
      if (!local && !server) throw new Error("That global Job ID is not stored in this browser or on the server.");

      if (local) await restoreFromLocal(local);
      else if (server?.source_count) await restoreFramesFromServer(id, server);
      if (!local?.settings && server?.settings) restoreSettings(server.settings);

      if ((!local?.outputBlob || !local.outputBlob.size) && server?.output_available) {
        const response = await fetch(`/download?id=${encodeURIComponent(id)}`, { cache: "no-store" });
        if (response.ok) {
          const blob = await response.blob();
          showResultBlob(blob, "server");
          restoring = false;
          await snapshot(blob);
          restoring = true;
        }
      }

      jobStatus.textContent = `Restored global job ${id}${server ? ` · server status: ${server.status}` : " · local browser copy"}.`;
      document.dispatchEvent(new CustomEvent("webp-global-job-restored", {
        detail: { id, metadata: server || null },
      }));

      if (server && ["queued", "running"].includes(server.status)) {
        try { monitorJob(id, { resumed: true, autoDownload: false }); } catch {}
      }
    } catch (error) {
      jobStatus.textContent = `Could not restore job: ${error.message}`;
    } finally {
      restoring = false;
      scheduleSnapshot();
    }
  }

  async function cacheServerResult(jobId, autoStart = false) {
    try {
      const response = await fetch(`/download?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      showResultBlob(blob, "indexeddb");
      await snapshot(blob);
      if (autoStart && !document.hidden) {
        const link = document.createElement("a");
        link.href = resultObjectUrl;
        link.download = "animation.webp";
        document.body.appendChild(link);
        link.click();
        link.remove();
      }
    } catch (error) {
      resultBlob = null;
      jobStatus.textContent = `WebP finished, but caching the local copy failed: ${error.message}. The server copy remains recoverable by Job ID.`;
      const href = `/download?id=${encodeURIComponent(jobId)}`;
      resultDownload.href = href;
      resultPanel.hidden = false;
      try { downloadLink.href = href; downloadLink.classList.add("visible"); } catch {}
    }
  }

  async function getFinishedWebPBlob() {
    if (resultBlob instanceof Blob && resultBlob.size) return resultBlob;
    if (!validJobId(globalJobId)) throw new Error("There is no finished job to continue from.");
    const response = await fetch(`/download?id=${encodeURIComponent(globalJobId)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text() || "Could not retrieve the finished WebP.");
    const blob = await response.blob();
    if (!blob.size) throw new Error("The finished WebP is empty.");
    return blob;
  }

  async function extractFinishedWebP(blob, sourceJobId) {
    const webpFile = new File(
      [blob],
      `continued-${sourceJobId.slice(0, 8) || "animation"}.webp`,
      { type: "image/webp", lastModified: Date.now() },
    );
    const data = new FormData();
    data.append("webp", webpFile, webpFile.name);
    const response = await fetch("/extract-webp", {
      method: "POST",
      cache: "no-store",
      body: data,
    });
    if (!response.ok) {
      throw new Error((await response.text()) || `Finished WebP extraction failed (${response.status}).`);
    }
    const extracted = await response.json();
    if (!Array.isArray(extracted?.frames)) {
      throw new Error("Server returned invalid finished-frame data.");
    }
    return extracted;
  }

  async function continueAsNewJob() {
    if (!continueButton) return;
    const sourceJobId = globalJobId;
    continueButton.disabled = true;
    const previousText = continueButton.textContent;
    continueButton.textContent = "Preparing frames…";
    try {
      jobStatus.textContent = "Decoding the finished WebP into editable frames…";
      const blob = await getFinishedWebPBlob();
      const extracted = await extractFinishedWebP(blob, sourceJobId);
      const decoded = (extracted.frames || []).map(extractedFrame => {
        const file = base64ToFile(
          extractedFrame.data,
          extractedFrame.name,
          extractedFrame.mime || "image/png",
        );
        return makeFrameItem(file, extractedFrame.name);
      });
      if (!decoded.length) throw new Error("The finished WebP did not contain any decodable frames.");

      await newJob();
      frames = decoded;

      if (extracted.suggested_duration && Number(extracted.suggested_duration) > 0) {
        duration.value = String(extracted.suggested_duration);
        try { updateDurationHint(); } catch {}
      }

      // Finished frames already share their final animation canvas. Do not silently
      // re-register them when the new continuation job is first rendered.
      const noneGeometry = geometryMode?.querySelector?.('option[value="none"]');
      if (noneGeometry) {
        geometryMode.value = "none";
        geometryMode.dispatchEvent(new Event("change", { bubbles: true }));
      }

      renderFrames();
      document.getElementById("uploadProgress")?.classList.remove("visible");
      await snapshot(null);
      jobStatus.textContent = `Created new job ${globalJobId} from ${decoded.length} finished frame${decoded.length === 1 ? "" : "s"}. Original job ${sourceJobId} is unchanged.`;
      status.textContent = `Continuation ready: ${decoded.length} frame${decoded.length === 1 ? "" : "s"} loaded from the finished WebP.`;
      document.dispatchEvent(new CustomEvent("webp-global-job-continued", {
        detail: { id: globalJobId, sourceJobId, frameCount: decoded.length },
      }));
    } catch (error) {
      jobStatus.textContent = `Could not continue from the finished WebP: ${error.message}`;
    } finally {
      continueButton.disabled = false;
      continueButton.textContent = previousText;
    }
  }

  function downloadFrameItem(item, index) {
    if (!item?.url) return;
    const fallback = `frame-${String(index + 1).padStart(4, "0")}.png`;
    const name = String(item.displayName || item.file?.name || fallback).trim() || fallback;
    const link = document.createElement("a");
    link.href = item.url;
    link.download = name;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function decorateFrameDownloadButtons() {
    const cards = document.querySelectorAll("#frameStrip .frame-card");
    cards.forEach((card, index) => {
      if (card.querySelector('[data-action="download-frame"]')) return;
      const item = frames.find(frame => frame.id === card.dataset.id) || frames[index];
      if (!item) return;
      const actions = card.querySelector(".frame-actions");
      if (!actions) return;
      actions.style.gridTemplateColumns = "repeat(4, minmax(0, 1fr))";
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.action = "download-frame";
      button.textContent = "↓";
      button.title = `Download frame ${index + 1}`;
      button.setAttribute("aria-label", `Download frame ${index + 1}`);
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        downloadFrameItem(item, index);
      });
      actions.appendChild(button);
    });
  }

  const originalShowDownload = typeof showDownload === "function" ? showDownload : null;
  if (originalShowDownload) {
    showDownload = function persistentShowDownload(jobId, autoStart = false) {
      if (validJobId(jobId)) setCurrentId(jobId);
      cacheServerResult(jobId, autoStart);
    };
  }

  copyButton.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(globalJobId); }
    catch { restoreInput.focus(); restoreInput.select(); }
  });
  newButton.addEventListener("click", () => newJob());
  restoreButton.addEventListener("click", () => restoreJob());
  continueButton?.addEventListener("click", () => continueAsNewJob());

  const strip = document.getElementById("frameStrip");
  if (strip) {
    new MutationObserver(() => {
      scheduleSnapshot();
      decorateFrameDownloadButtons();
    }).observe(strip, { childList: true });
    decorateFrameDownloadButtons();
  }
  document.querySelectorAll("input, select, textarea").forEach(control => {
    if (control.type === "file" || control.id === "globalRestoreId") return;
    control.addEventListener("change", scheduleSnapshot);
  });

  window.webpAnimatorWorkspace = {
    currentId,
    persistNow: snapshot,
    restore: restoreJob,
    cacheResult: cacheServerResult,
    continueAsNewJob,
  };

  (async () => {
    let savedId = "";
    try { savedId = localStorage.getItem(CURRENT_KEY) || ""; } catch {}
    if (!validJobId(savedId)) savedId = randomJobId();
    setCurrentId(savedId);
    await ensureServerJob(savedId);
    let local = null;
    try { local = await dbGet(savedId); } catch {}
    if (local) await restoreJob(savedId);
    else {
      const server = await fetchServerJob(savedId).catch(() => null);
      if (server && (server.source_count || server.output_available)) await restoreJob(savedId);
      else await snapshot(null);
    }
  })();
})();
