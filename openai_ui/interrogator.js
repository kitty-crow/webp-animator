(() => {
  const root = document.getElementById("openaiInterrogator");
  if (!root || root.dataset.initialised === "1") return;
  root.dataset.initialised = "1";

  const $ = id => document.getElementById(id);
  const sequenceMode = $("oiSequenceMode");
  const mode = $("oiMode");
  const left = $("oiLeft");
  const right = $("oiRight");
  const loopClosure = $("oiLoopClosure");
  const loopClosureWrap = $("oiLoopClosureWrap");
  const countWrap = $("oiCountWrap");
  const count = $("oiCount");
  const maxFramesWrap = $("oiMaxFramesWrap");
  const maxFrames = $("oiMaxFrames");
  const benefitWrap = $("oiBenefitWrap");
  const minBenefit = $("oiMinBenefit");
  const retries = $("oiRetries");
  const maxSpend = $("oiMaxSpend");
  const wholeContext = $("oiWholeContext");
  const instruction = $("oiInstruction");
  const plannerModel = $("oiPlannerModel");
  const plannerEffort = $("oiPlannerEffort");
  const imageModel = $("oiImageModel");
  const imageQuality = $("oiImageQuality");
  const auditorModel = $("oiAuditorModel");
  const auditorEffort = $("oiAuditorEffort");
  const statusBox = $("oiStatus");
  const costBox = $("oiCost");
  const estimateButton = $("oiEstimate");
  const startButton = $("oiStart");
  const jobBox = $("oiJob");
  const reviewBox = $("oiReview");
  const feedback = $("oiFeedback");
  const retryButton = $("oiRetry");
  const restoreId = $("oiRestoreId");
  const restoreButton = $("oiRestore");

  const JOBS_KEY = "webp-animator-openai-jobs-v1";
  const ACTIVE_KEY = "webp-animator-openai-active-v1";
  let openaiReady = false;
  let currentJob = null;
  let monitorToken = 0;
  let estimateTimer = null;

  function oiSleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function oiWaitVisible() {
    if (!document.hidden) return Promise.resolve();
    return new Promise(resolve => {
      const handler = () => {
        if (document.hidden) return;
        document.removeEventListener("visibilitychange", handler);
        resolve();
      };
      document.addEventListener("visibilitychange", handler);
    });
  }

  function readJobs() {
    try {
      const value = JSON.parse(localStorage.getItem(JOBS_KEY) || "[]");
      return Array.isArray(value) ? value : [];
    } catch {
      return [];
    }
  }

  function rememberJob(job) {
    const jobs = readJobs().filter(item => item && item.id !== job.id);
    jobs.unshift({ id: job.id, created: job.created || Date.now() / 1000 });
    try {
      localStorage.setItem(JOBS_KEY, JSON.stringify(jobs.slice(0, 20)));
      localStorage.setItem(ACTIVE_KEY, job.id);
    } catch {}
  }

  function clearActive(jobId) {
    try {
      if (!jobId || localStorage.getItem(ACTIVE_KEY) === jobId) localStorage.removeItem(ACTIVE_KEY);
    } catch {}
  }

  function activeJobId() {
    try { return localStorage.getItem(ACTIVE_KEY) || ""; } catch { return ""; }
  }

  async function jsonRequest(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const text = await response.text();
    let value;
    try { value = JSON.parse(text); } catch { value = { error: text || `HTTP ${response.status}` }; }
    if (!response.ok || value?.error) throw new Error(value?.error || `HTTP ${response.status}`);
    return value;
  }

  function currentFrames() {
    try {
      return Array.isArray(frames) ? frames : [];
    } catch {
      return [];
    }
  }

  function updatePairOptions() {
    const list = currentFrames();
    const previousLeft = Number(left.value || 0);
    const previousRight = Number(right.value || 1);
    left.innerHTML = "";
    right.innerHTML = "";
    list.forEach((item, index) => {
      const label = `Frame ${index + 1} · ${item.displayName || item.file?.name || "image"}`;
      const a = document.createElement("option");
      a.value = String(index);
      a.textContent = label;
      const b = a.cloneNode(true);
      left.appendChild(a);
      right.appendChild(b);
    });
    if (list.length >= 2) {
      left.value = String(Math.min(previousLeft, list.length - 2));
      right.value = String(Math.max(Number(left.value) + 1, Math.min(previousRight, list.length - 1)));
    }
    applyLoopMode();
    startButton.disabled = !openaiReady || list.length < 2;
    scheduleEstimate();
  }

  function applyLoopMode() {
    const list = currentFrames();
    const loop = sequenceMode.value === "loop";
    loopClosureWrap.hidden = !loop;
    if (!loop) loopClosure.checked = false;
    if (loop && loopClosure.checked && list.length >= 2) {
      left.value = String(list.length - 1);
      right.value = "0";
      left.disabled = true;
      right.disabled = true;
    } else {
      left.disabled = false;
      right.disabled = false;
      if (Number(left.value) >= Number(right.value) && list.length >= 2) {
        left.value = "0";
        right.value = "1";
      }
    }
  }

  function applyMode() {
    countWrap.hidden = mode.value !== "fixed";
    maxFramesWrap.hidden = mode.value !== "auto";
    benefitWrap.hidden = mode.value !== "auto";
    scheduleEstimate();
  }

  function requestSettings() {
    const list = currentFrames();
    const isClosure = sequenceMode.value === "loop" && loopClosure.checked;
    return {
      sequence_mode: sequenceMode.value,
      loop_closure: isClosure,
      left_index: isClosure ? Math.max(0, list.length - 1) : Number(left.value || 0),
      right_index: isClosure ? 0 : Number(right.value || 1),
      mode: mode.value,
      count: Math.max(1, Number(count.value || 1)),
      max_frames: Math.max(1, Number(maxFrames.value || 1)),
      min_benefit_score: Math.max(0, Math.min(100, Number(minBenefit.value || 0))),
      auto_retries: Math.max(0, Number(retries.value || 0)),
      max_spend_usd: Math.max(0, Number(maxSpend.value || 0)),
      whole_sequence_context: wholeContext.checked,
      user_instruction: instruction.value.trim(),
      planner_model: plannerModel.value.trim() || "gpt-5.6-luna",
      planner_effort: plannerEffort.value,
      image_model: imageModel.value.trim() || "gpt-image-2",
      image_quality: imageQuality.value,
      auditor_model: auditorModel.value.trim() || "gpt-5.6-luna",
      auditor_effort: auditorEffort.value,
    };
  }

  async function firstFrameDimensions() {
    const list = currentFrames();
    if (!list.length) return { width: 1024, height: 1024 };
    return await new Promise(resolve => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth || 1024, height: image.naturalHeight || 1024 });
      image.onerror = () => resolve({ width: 1024, height: 1024 });
      image.src = list[0].url;
    });
  }

  async function refreshEstimate() {
    const list = currentFrames();
    if (!openaiReady || list.length < 2) {
      costBox.textContent = list.length < 2
        ? "Cost estimate will appear when at least two frames are loaded."
        : "OpenAI bridge is not ready.";
      return;
    }
    try {
      const dimensions = await firstFrameDimensions();
      const payload = { ...requestSettings(), frame_count: list.length, ...dimensions };
      const estimate = await jsonRequest("/openai/estimate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const one = Number(estimate.one_attempt || 0);
      const maximum = Number(estimate.maximum || 0);
      const framesExpected = Number(estimate.frames || 1);
      costBox.textContent = [
        `Estimated paid attempt: ~$${one.toFixed(4)} per generated-frame attempt`,
        `Configured generation: ${framesExpected} frame${framesExpected === 1 ? "" : "s"} · up to ${estimate.attempts_per_frame || 1} attempt${Number(estimate.attempts_per_frame || 1) === 1 ? "" : "s"}/frame`,
        `Estimated worst case for this configuration: ~$${maximum.toFixed(4)}`,
        "Estimate includes planning + image generation/edit + audit and is approximate; actual API usage is recorded after each attempt.",
      ].join("\n");
    } catch (error) {
      costBox.textContent = `Could not estimate cost: ${error.message}`;
    }
  }

  function scheduleEstimate() {
    clearTimeout(estimateTimer);
    estimateTimer = setTimeout(() => refreshEstimate(), 250);
  }

  async function checkStatus() {
    try {
      const value = await jsonRequest("/openai-status");
      openaiReady = Boolean(value.key_configured && value.schema_built && value.runtime);
      if (openaiReady) {
        statusBox.textContent = `Ready · API key loaded from server environment/.env · runtime: ${value.runtime}${value.bridge_running ? " · bridge running" : " · bridge starts on demand"}`;
      } else {
        const missing = [];
        if (!value.key_configured) missing.push("OPENAI_API_KEY in .env");
        if (!value.schema_built) missing.push("built vendored openai-schema (run python setup_openai.py)");
        if (!value.runtime) missing.push("Node.js or Bun");
        statusBox.textContent = `Not ready: ${missing.join("; ")}`;
      }
    } catch (error) {
      openaiReady = false;
      statusBox.textContent = `OpenAI Interrogator status failed: ${error.message}`;
    }
    updatePairOptions();
  }

  function uploadOpenAIJob(data) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/openai/interpolate");
      xhr.upload.addEventListener("progress", event => {
        if (event.lengthComputable) {
          const pct = Math.round(event.loaded / event.total * 100);
          statusBox.textContent = `Uploading OpenAI job source frames… ${pct}%\nKeep this page open until the server returns a Job ID.`;
        }
      });
      xhr.addEventListener("load", () => {
        let value;
        try { value = JSON.parse(xhr.responseText); } catch { value = { error: xhr.responseText || `HTTP ${xhr.status}` }; }
        if (xhr.status < 200 || xhr.status >= 300 || value?.error) {
          reject(new Error(value?.error || `Upload failed (${xhr.status})`));
          return;
        }
        resolve(value);
      });
      xhr.addEventListener("error", () => reject(new Error("Network error while uploading the OpenAI interpolation job.")));
      xhr.send(data);
    });
  }

  function formatAudit(audit) {
    if (!audit) return "";
    const score = Number(audit.overall_score ?? 0);
    const state = audit.acceptable ? "accepted" : "rejected";
    const issues = Array.isArray(audit.violations) && audit.violations.length
      ? audit.violations.map(item => `• ${item.description || item.type}`).join("\n")
      : "• no material violations reported";
    return `${state.toUpperCase()} · match assessment ${score}/100\n${issues}`;
  }

  function renderJob(job) {
    currentJob = job;
    rememberJob(job);
    jobBox.hidden = false;
    const attempts = Array.isArray(job.attempts) ? job.attempts : [];
    const latest = attempts[attempts.length - 1];
    const accepted = Array.isArray(job.accepted) ? job.accepted : [];
    const lines = [
      `Job ID: ${job.id}`,
      `Status: ${job.status}${job.stage ? ` · ${job.stage}` : ""}`,
      `Progress: ${Math.round(Number(job.progress || 0))}%`,
      `Spent so far: $${Number(job.spent_usd || 0).toFixed(4)}`,
      Number(job.estimated_next_usd || 0) > 0 ? `Next attempt estimate: ~$${Number(job.estimated_next_usd).toFixed(4)}` : "",
      job.message || "",
      `Accepted generated frames: ${accepted.length}`,
    ].filter(Boolean);

    if (latest?.audit) lines.push("", formatAudit(latest.audit));
    jobBox.textContent = lines.join("\n");

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "small-button";
    copy.textContent = "Copy Job ID";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(job.id); }
      catch { restoreId.value = job.id; restoreId.focus(); restoreId.select(); }
    });
    jobBox.appendChild(document.createElement("br"));
    jobBox.appendChild(copy);

    if (job.status === "done" && accepted.length) {
      const insert = document.createElement("button");
      insert.type = "button";
      insert.className = "small-button";
      insert.style.marginLeft = "8px";
      insert.textContent = "Insert accepted frames into animation";
      insert.addEventListener("click", () => insertResults(job));
      jobBox.appendChild(insert);
    }

    reviewBox.hidden = !["needs_review", "interrupted", "budget_wait"].includes(job.status);
    retryButton.textContent = job.status === "interrupted" ? "Resume job" : "Learn from this attempt and try again";

    if (["done", "error", "cancelled"].includes(job.status)) clearActive(job.id);
  }

  async function insertResults(job) {
    const accepted = [...(job.accepted || [])].sort((a, b) => Number(a.target_fraction) - Number(b.target_fraction));
    if (!accepted.length) return;
    const list = currentFrames();
    const request = job.request || {};
    const newItems = [];
    for (let index = 0; index < accepted.length; index += 1) {
      const result = accepted[index];
      const response = await fetch(result.result_url, { cache: "no-store" });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const name = `openai-between-${Number(result.target_fraction).toFixed(3)}-${job.id.slice(0, 8)}.png`;
      const file = new File([blob], name, { type: "image/png" });
      newItems.push(makeFrameItem(file, name));
    }

    if (request.loop_closure) {
      list.push(...newItems);
    } else {
      const insertion = Math.max(0, Math.min(list.length, Number(request.left_index || 0) + 1));
      list.splice(insertion, 0, ...newItems);
    }
    renderFrames();
    status.textContent = `Inserted ${newItems.length} accepted OpenAI frame${newItems.length === 1 ? "" : "s"}. You can now use RIFE above for free final smoothing or generate the WebP directly.`;
    scheduleEstimate();
  }

  async function monitor(jobId, resumed = false) {
    const token = ++monitorToken;
    if (resumed) statusBox.textContent = `Restoring OpenAI job ${jobId}…`;
    while (token === monitorToken) {
      if (document.hidden) {
        await oiWaitVisible();
        if (token !== monitorToken) return;
      }
      try {
        const job = await jsonRequest(`/openai/job?id=${encodeURIComponent(jobId)}`);
        renderJob(job);
        if (["done", "error", "needs_review", "budget_wait", "interrupted", "cancelled"].includes(job.status)) return;
      } catch (error) {
        statusBox.textContent = `The server-side job should continue. Reconnecting: ${error.message}`;
      }
      await oiSleep(1200);
    }
  }

  async function startJob() {
    const list = currentFrames();
    if (list.length < 2) {
      statusBox.textContent = "Load at least two frames first.";
      return;
    }
    if (!openaiReady) {
      statusBox.textContent = "OpenAI Interrogator is not ready. Check the status above.";
      return;
    }
    const settings = requestSettings();
    if (!settings.loop_closure && settings.left_index >= settings.right_index) {
      statusBox.textContent = "The earlier anchor must come before the later anchor. Use Loop closure for last → first.";
      return;
    }

    startButton.disabled = true;
    try {
      const data = new FormData();
      data.append("openai_request", JSON.stringify(settings));
      list.forEach(item => data.append("frames", item.file, item.file.name));
      const job = await uploadOpenAIJob(data);
      rememberJob(job);
      restoreId.value = job.id;
      statusBox.textContent = `Upload complete. Job ${job.id} is owned by the server now. You may leave the tab or close the browser.`;
      renderJob(job);
      monitor(job.id);
    } catch (error) {
      statusBox.textContent = `OpenAI interpolation failed to start: ${error.message}`;
    } finally {
      startButton.disabled = !openaiReady || currentFrames().length < 2;
    }
  }

  async function retryJob() {
    if (!currentJob?.id) return;
    retryButton.disabled = true;
    try {
      const job = await jsonRequest("/openai/retry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: currentJob.id,
          feedback: feedback.value.trim(),
          max_spend_usd: Math.max(0, Number(maxSpend.value || 0)),
        }),
      });
      feedback.value = "";
      renderJob(job);
      monitor(job.id, true);
    } catch (error) {
      statusBox.textContent = `Could not resume job: ${error.message}`;
    } finally {
      retryButton.disabled = false;
    }
  }

  async function restoreJob() {
    const id = restoreId.value.trim() || activeJobId();
    if (!id) {
      statusBox.textContent = "Enter a Job ID to restore.";
      return;
    }
    ++monitorToken;
    try {
      const job = await jsonRequest(`/openai/job?id=${encodeURIComponent(id)}`);
      renderJob(job);
      if (!["done", "error", "needs_review", "budget_wait", "interrupted", "cancelled"].includes(job.status)) monitor(job.id, true);
    } catch (error) {
      statusBox.textContent = `Could not restore job: ${error.message}`;
    }
  }

  sequenceMode.addEventListener("change", () => { applyLoopMode(); scheduleEstimate(); });
  loopClosure.addEventListener("change", () => { applyLoopMode(); scheduleEstimate(); });
  mode.addEventListener("change", applyMode);
  [left, right, count, maxFrames, minBenefit, retries, maxSpend, wholeContext, plannerModel, plannerEffort, imageModel, imageQuality, auditorModel, auditorEffort]
    .forEach(control => control.addEventListener("change", scheduleEstimate));
  instruction.addEventListener("input", scheduleEstimate);
  estimateButton.addEventListener("click", refreshEstimate);
  startButton.addEventListener("click", startJob);
  retryButton.addEventListener("click", retryJob);
  restoreButton.addEventListener("click", restoreJob);

  const strip = document.getElementById("frameStrip");
  if (strip) new MutationObserver(updatePairOptions).observe(strip, { childList: true });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && currentJob?.id && !["done", "error", "needs_review", "budget_wait", "interrupted", "cancelled"].includes(currentJob.status)) {
      monitor(currentJob.id, true);
    }
  });

  applyMode();
  updatePairOptions();
  checkStatus().then(() => {
    const saved = activeJobId();
    if (saved) {
      restoreId.value = saved;
      restoreJob();
    }
  });
})();
