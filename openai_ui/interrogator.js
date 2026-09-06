(() => {
  const root = document.getElementById("openaiInterrogator");
  if (!root || root.dataset.initialised === "1") return;
  root.dataset.initialised = "1";

  const $ = id => document.getElementById(id);
  const sequenceMode = $("oiSequenceMode");
  const scope = $("oiScope");
  const mode = $("oiMode");
  const left = $("oiLeft");
  const right = $("oiRight");
  const leftWrap = $("oiLeftWrap");
  const rightWrap = $("oiRightWrap");
  const leftLabel = $("oiLeftLabel");
  const rightLabel = $("oiRightLabel");
  const loopClosure = $("oiLoopClosure");
  const loopClosureWrap = $("oiLoopClosureWrap");
  const loopClosureText = $("oiLoopClosureText");
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

  const JOBS_KEY = "webp-animator-openai-jobs-v2";
  const ACTIVE_KEY = "webp-animator-openai-active-v2";
  const STOPPED = new Set(["done", "error", "needs_review", "budget_wait", "interrupted", "cancelled"]);
  let openaiReady = false;
  let currentJob = null;
  let monitorToken = 0;
  let estimateTimer = null;

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  function waitVisible() {
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

  function currentFrames() {
    try { return Array.isArray(frames) ? frames : []; }
    catch { return []; }
  }

  function readJobs() {
    try {
      const value = JSON.parse(localStorage.getItem(JOBS_KEY) || "[]");
      return Array.isArray(value) ? value : [];
    } catch { return []; }
  }

  function rememberJob(job) {
    const jobs = readJobs().filter(item => item?.id !== job.id);
    jobs.unshift({ id: job.id, kind: job.kind || "single", created: job.created || Date.now() / 1000 });
    try {
      localStorage.setItem(JOBS_KEY, JSON.stringify(jobs.slice(0, 30)));
      if (!STOPPED.has(job.status)) localStorage.setItem(ACTIVE_KEY, job.id);
    } catch {}
  }

  function clearActive(jobId) {
    try {
      if (!jobId || localStorage.getItem(ACTIVE_KEY) === jobId) localStorage.removeItem(ACTIVE_KEY);
    } catch {}
  }

  function activeJobId() {
    try { return localStorage.getItem(ACTIVE_KEY) || ""; }
    catch { return ""; }
  }

  async function jsonRequest(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const text = await response.text();
    let value;
    try { value = JSON.parse(text); }
    catch { value = { error: text || `HTTP ${response.status}` }; }
    // Successful job-status objects are allowed to contain an `error` field when
    // status=error. HTTP status, not the presence of that field, determines whether
    // transport itself failed.
    if (!response.ok) throw new Error(value?.error || `HTTP ${response.status}`);
    return value;
  }

  function updateFrameOptions() {
    const list = currentFrames();
    const previousLeft = Number(left.value || 0);
    const previousRight = Number(right.value || Math.max(1, list.length - 1));
    left.innerHTML = "";
    right.innerHTML = "";
    list.forEach((item, index) => {
      const label = `Frame ${index + 1} · ${item.displayName || item.file?.name || "image"}`;
      const optionA = document.createElement("option");
      optionA.value = String(index);
      optionA.textContent = label;
      left.appendChild(optionA);
      right.appendChild(optionA.cloneNode(true));
    });
    if (list.length >= 2) {
      left.value = String(Math.min(previousLeft, list.length - 2));
      right.value = String(Math.max(Number(left.value) + 1, Math.min(previousRight, list.length - 1)));
    }
    applyScope();
    startButton.disabled = !openaiReady || list.length < 2;
    scheduleEstimate();
  }

  function applyScope() {
    const list = currentFrames();
    const value = scope.value;
    const loop = sequenceMode.value === "loop";

    leftWrap.hidden = value === "all";
    rightWrap.hidden = value === "all";
    leftLabel.textContent = value === "range" ? "First frame in range" : "Earlier anchor";
    rightLabel.textContent = value === "range" ? "Last frame in range" : "Later anchor";

    if (value === "all") {
      left.disabled = true;
      right.disabled = true;
      if (list.length >= 2) {
        left.value = "0";
        right.value = String(list.length - 1);
      }
      loopClosureWrap.hidden = !loop;
      loopClosureText.textContent = "Also interpolate the loop closure gap (last frame → first frame)";
      if (!loop) loopClosure.checked = false;
    } else if (value === "range") {
      left.disabled = false;
      right.disabled = false;
      loopClosureWrap.hidden = true;
      loopClosure.checked = false;
      if (list.length >= 2 && Number(left.value) >= Number(right.value)) {
        left.value = "0";
        right.value = String(list.length - 1);
      }
    } else {
      loopClosureWrap.hidden = !loop;
      loopClosureText.textContent = "Use the loop closure gap (last frame → first frame) instead of the selected pair";
      if (!loop) loopClosure.checked = false;
      if (loop && loopClosure.checked && list.length >= 2) {
        left.value = String(list.length - 1);
        right.value = "0";
        left.disabled = true;
        right.disabled = true;
      } else {
        left.disabled = false;
        right.disabled = false;
        if (list.length >= 2 && Number(left.value) >= Number(right.value)) {
          left.value = "0";
          right.value = "1";
        }
      }
    }
    scheduleEstimate();
  }

  function applyMode() {
    countWrap.hidden = mode.value !== "fixed";
    maxFramesWrap.hidden = mode.value !== "auto";
    benefitWrap.hidden = mode.value !== "auto";
    scheduleEstimate();
  }

  function requestSettings() {
    const list = currentFrames();
    const selectedScope = scope.value;
    const pairClosure = selectedScope === "pair" && sequenceMode.value === "loop" && loopClosure.checked;
    return {
      sequence_mode: sequenceMode.value,
      scope: selectedScope,
      loop_closure: pairClosure,
      include_loop_closure: selectedScope === "all" && sequenceMode.value === "loop" && loopClosure.checked,
      left_index: pairClosure ? Math.max(0, list.length - 1) : Number(left.value || 0),
      right_index: pairClosure ? 0 : Number(right.value || Math.max(1, list.length - 1)),
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
      costBox.textContent = list.length < 2 ? "Cost estimate will appear when at least two frames are loaded." : "OpenAI bridge is not ready.";
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
      const segments = Number(estimate.segments || 1);
      costBox.textContent = [
        `Estimated paid attempt: ~$${one.toFixed(4)} per generated-frame attempt`,
        `Selected gaps: ${segments}`,
        `Configured generation: up to ${framesExpected} generated frame${framesExpected === 1 ? "" : "s"} total · up to ${estimate.attempts_per_frame || 1} attempt${Number(estimate.attempts_per_frame || 1) === 1 ? "" : "s"}/frame`,
        `Estimated worst case for this configuration: ~$${maximum.toFixed(4)}`,
        wholeContext.checked ? "Full-animation context is enabled." : "Only the immediate gap context is enabled.",
        "The estimate is approximate; actual API usage is recorded after each completed attempt.",
      ].join("\n");
    } catch (error) {
      costBox.textContent = `Could not estimate cost: ${error.message}`;
    }
  }

  function scheduleEstimate() {
    clearTimeout(estimateTimer);
    estimateTimer = setTimeout(refreshEstimate, 250);
  }

  async function checkStatus() {
    try {
      const value = await jsonRequest("/openai-status");
      openaiReady = Boolean(value.key_configured && value.schema_built && value.runtime);
      if (openaiReady) {
        statusBox.textContent = `Ready · API key loaded server-side · runtime: ${value.runtime}${value.bridge_running ? " · bridge running" : " · bridge starts on demand"}`;
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
    updateFrameOptions();
  }

  function uploadJob(data) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/openai/interpolate");
      xhr.upload.addEventListener("progress", event => {
        if (event.lengthComputable) {
          const pct = Math.round(event.loaded / event.total * 100);
          statusBox.textContent = `Uploading OpenAI job source frames… ${pct}%\nKeep this page open only until the server returns a Job ID.`;
        }
      });
      xhr.addEventListener("load", () => {
        let value;
        try { value = JSON.parse(xhr.responseText); }
        catch { value = { error: xhr.responseText || `HTTP ${xhr.status}` }; }
        if (xhr.status < 200 || xhr.status >= 300) return reject(new Error(value?.error || `Upload failed (${xhr.status})`));
        resolve(value);
      });
      xhr.addEventListener("error", () => reject(new Error("Network error while uploading the OpenAI interpolation job.")));
      xhr.send(data);
    });
  }

  function formatAudit(audit) {
    if (!audit) return "";
    const state = audit.acceptable ? "ACCEPTED" : "REJECTED";
    const issues = Array.isArray(audit.violations) && audit.violations.length
      ? audit.violations.map(item => `• ${item.description || item.type}`).join("\n")
      : "• no material violations reported";
    return `${state} · match assessment ${Number(audit.overall_score || 0)}/100\n${issues}`;
  }

  function latestAttempt(job) {
    const attempts = Array.isArray(job.attempts) ? job.attempts : [];
    return attempts.length ? attempts[attempts.length - 1] : null;
  }

  function attemptUrl(job, attempt) {
    const owner = attempt.job_id || job.id;
    return `/openai/attempt?id=${encodeURIComponent(owner)}&fraction=${encodeURIComponent(attempt.target_fraction)}&attempt=${encodeURIComponent(attempt.attempt)}`;
  }

  function renderJob(job) {
    currentJob = job;
    rememberJob(job);
    jobBox.hidden = false;
    jobBox.innerHTML = "";
    const latest = latestAttempt(job);
    const accepted = Array.isArray(job.accepted) ? job.accepted : [];
    const segments = Array.isArray(job.segments) ? job.segments : [];
    const completedSegments = segments.filter(item => item.status === "done").length;
    const lines = [
      `Job ID: ${job.id}`,
      job.kind === "batch" ? `Batch gaps: ${completedSegments}/${segments.length}` : "",
      `Status: ${job.status}${job.stage ? ` · ${job.stage}` : ""}`,
      `Progress: ${Math.round(Number(job.progress || 0))}%`,
      `Spent so far: $${Number(job.spent_usd || 0).toFixed(4)}`,
      Number(job.estimated_next_usd || 0) > 0 ? `Next attempt estimate: ~$${Number(job.estimated_next_usd).toFixed(4)}` : "",
      job.message || "",
      job.error ? `Error: ${job.error}` : "",
      `Accepted generated frames: ${accepted.length}`,
    ].filter(Boolean);

    if (latest?.audit) {
      const segmentText = latest.left_index != null ? ` · gap ${Number(latest.left_index) + 1}→${Number(latest.right_index) + 1}` : "";
      lines.push(
        "",
        `Latest attempt${segmentText}: target t=${Number(latest.target_fraction).toFixed(3)} · attempt ${latest.attempt}`,
        `Attempt cost: $${Number(latest.actual_cost_usd || latest.estimated_cost_usd || 0).toFixed(4)}`,
        formatAudit(latest.audit),
      );
    }

    const text = document.createElement("div");
    text.textContent = lines.join("\n");
    jobBox.appendChild(text);

    if (latest) {
      const preview = document.createElement("img");
      preview.src = attemptUrl(job, latest);
      preview.alt = `OpenAI interpolation attempt ${latest.attempt}`;
      preview.style.display = "block";
      preview.style.width = "min(360px, 100%)";
      preview.style.maxHeight = "420px";
      preview.style.objectFit = "contain";
      preview.style.marginTop = "10px";
      preview.style.borderRadius = "10px";
      preview.style.background = "repeating-conic-gradient(color-mix(in srgb, CanvasText 8%, Canvas) 0 25%, Canvas 0 50%) 0/18px 18px";
      jobBox.appendChild(preview);
    }

    const actions = document.createElement("div");
    actions.className = "openai-actions";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy Job ID";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(job.id); }
      catch { restoreId.value = job.id; restoreId.focus(); restoreId.select(); }
    });
    actions.appendChild(copy);

    if (job.status === "done" && accepted.length) {
      const insert = document.createElement("button");
      insert.type = "button";
      insert.textContent = "Insert accepted frames into animation";
      insert.addEventListener("click", () => insertResults(job).catch(error => {
        statusBox.textContent = `Could not insert generated frames: ${error.message}`;
      }));
      actions.appendChild(insert);
    }
    jobBox.appendChild(actions);

    const resumable = ["needs_review", "budget_wait", "interrupted"].includes(job.status);
    const canRejectAccepted = job.status === "done" && latest?.audit?.acceptable;
    reviewBox.hidden = !(resumable || canRejectAccepted);
    if (!reviewBox.hidden) {
      if (job.status === "interrupted") retryButton.textContent = "Resume job";
      else if (job.status === "budget_wait") retryButton.textContent = "Resume with current spend limit";
      else if (canRejectAccepted) retryButton.textContent = "Reject this result, add feedback and try again";
      else retryButton.textContent = "Learn from this attempt and try again";
    }
    if (STOPPED.has(job.status)) clearActive(job.id);
  }

  async function insertResults(job) {
    const accepted = [...(job.accepted || [])];
    if (!accepted.length) return;
    const list = currentFrames();
    const request = job.request || {};
    const prepared = [];

    for (const result of accepted) {
      const response = await fetch(result.result_url, { cache: "no-store" });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const leftIndex = result.left_index != null ? Number(result.left_index) : Number(request.left_index || 0);
      const rightIndex = result.right_index != null ? Number(result.right_index) : Number(request.right_index || 1);
      const closure = result.loop_closure != null ? Boolean(result.loop_closure) : Boolean(request.loop_closure);
      const fraction = Number(result.target_fraction || 0.5);
      const name = `openai-${leftIndex + 1}-${rightIndex + 1}-${fraction.toFixed(3)}-${job.id.slice(0, 8)}.png`;
      prepared.push({ leftIndex, rightIndex, closure, fraction, item: makeFrameItem(new File([blob], name, { type: "image/png" }), name) });
    }

    const normalGroups = new Map();
    const closureItems = [];
    for (const entry of prepared) {
      if (entry.closure) {
        closureItems.push(entry);
        continue;
      }
      if (!normalGroups.has(entry.leftIndex)) normalGroups.set(entry.leftIndex, []);
      normalGroups.get(entry.leftIndex).push(entry);
    }

    // Work backwards through source indexes so inserting frames into later gaps cannot
    // shift the insertion positions for earlier gaps.
    [...normalGroups.entries()]
      .sort((a, b) => b[0] - a[0])
      .forEach(([leftIndex, entries]) => {
        entries.sort((a, b) => a.fraction - b.fraction);
        list.splice(leftIndex + 1, 0, ...entries.map(entry => entry.item));
      });

    closureItems.sort((a, b) => a.fraction - b.fraction);
    list.push(...closureItems.map(entry => entry.item));
    renderFrames();
    status.textContent = `Inserted ${prepared.length} accepted OpenAI frame${prepared.length === 1 ? "" : "s"}. The RIFE controls below can now add the free final smoothing pass.`;
    scheduleEstimate();
  }

  async function monitor(jobId, resumed = false) {
    const token = ++monitorToken;
    if (resumed) statusBox.textContent = `Restoring OpenAI job ${jobId}…`;
    while (token === monitorToken) {
      if (document.hidden) {
        await waitVisible();
        if (token !== monitorToken) return;
      }
      try {
        const job = await jsonRequest(`/openai/job?id=${encodeURIComponent(jobId)}`);
        renderJob(job);
        if (STOPPED.has(job.status)) return;
      } catch (error) {
        statusBox.textContent = `The server-side job should continue. Reconnecting: ${error.message}`;
      }
      await sleep(1200);
    }
  }

  function validateSettings(settings) {
    const list = currentFrames();
    if (settings.scope === "pair" && !settings.loop_closure && settings.left_index >= settings.right_index) {
      throw new Error("The earlier anchor must come before the later anchor. Use loop closure for last → first.");
    }
    if (settings.scope === "range" && settings.left_index >= settings.right_index) {
      throw new Error("The first frame in a range must come before the last frame.");
    }
    if (settings.scope === "all" && list.length < 2) throw new Error("Load at least two frames first.");
  }

  async function startJob() {
    const list = currentFrames();
    if (list.length < 2) return void (statusBox.textContent = "Load at least two frames first.");
    if (!openaiReady) return void (statusBox.textContent = "OpenAI Interrogator is not ready.");
    const settings = requestSettings();
    try { validateSettings(settings); }
    catch (error) { return void (statusBox.textContent = error.message); }

    startButton.disabled = true;
    try {
      const data = new FormData();
      data.append("openai_request", JSON.stringify(settings));
      list.forEach(item => data.append("frames", item.file, item.file.name));
      const job = await uploadJob(data);
      rememberJob(job);
      restoreId.value = job.id;
      statusBox.textContent = `Upload complete. Job ${job.id} is owned by the server now. You may leave the tab, close Safari or lose the client connection.`;
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
    const latest = latestAttempt(currentJob);
    retryButton.disabled = true;
    try {
      const job = await jsonRequest("/openai/retry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: currentJob.id,
          target_fraction: latest?.target_fraction ?? currentJob.pending_target ?? null,
          feedback: feedback.value.trim(),
          max_spend_usd: Math.max(0, Number(maxSpend.value || 0)),
        }),
      });
      feedback.value = "";
      renderJob(job);
      monitor(job.id, true);
    } catch (error) {
      statusBox.textContent = `Could not retry job: ${error.message}`;
    } finally {
      retryButton.disabled = false;
    }
  }

  async function restoreJob() {
    const id = restoreId.value.trim() || activeJobId();
    if (!id) return void (statusBox.textContent = "Enter a Job ID to restore.");
    ++monitorToken;
    try {
      const job = await jsonRequest(`/openai/job?id=${encodeURIComponent(id)}`);
      renderJob(job);
      if (!STOPPED.has(job.status)) monitor(job.id, true);
    } catch (error) {
      statusBox.textContent = `Could not restore job: ${error.message}`;
    }
  }

  sequenceMode.addEventListener("change", applyScope);
  scope.addEventListener("change", applyScope);
  loopClosure.addEventListener("change", applyScope);
  mode.addEventListener("change", applyMode);
  [left, right, count, maxFrames, minBenefit, retries, maxSpend, wholeContext, plannerModel, plannerEffort, imageModel, imageQuality, auditorModel, auditorEffort]
    .forEach(control => control.addEventListener("change", scheduleEstimate));
  instruction.addEventListener("input", scheduleEstimate);
  estimateButton.addEventListener("click", refreshEstimate);
  startButton.addEventListener("click", startJob);
  retryButton.addEventListener("click", retryJob);
  restoreButton.addEventListener("click", restoreJob);

  const strip = document.getElementById("frameStrip");
  if (strip) new MutationObserver(updateFrameOptions).observe(strip, { childList: true });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && currentJob?.id && !STOPPED.has(currentJob.status)) monitor(currentJob.id, true);
  });

  applyMode();
  updateFrameOptions();
  checkStatus().then(() => {
    const saved = activeJobId();
    if (saved) {
      restoreId.value = saved;
      restoreJob();
    }
  });
})();
