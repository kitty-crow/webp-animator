(() => {
  const panel = document.getElementById("openaiInterrogator");
  const jobBox = document.getElementById("oiJob");
  if (!panel || !jobBox || document.getElementById("oiStop")) return;

  const actions = document.createElement("div");
  actions.className = "openai-actions";
  actions.id = "oiStopActions";
  actions.hidden = true;

  const button = document.createElement("button");
  button.type = "button";
  button.id = "oiStop";
  button.textContent = "Stop OpenAI job now";
  button.style.background = "#8b0000";
  button.style.color = "white";
  button.style.borderColor = "#8b0000";
  actions.appendChild(button);
  jobBox.insertAdjacentElement("afterend", actions);

  function jobId() {
    const match = jobBox.textContent.match(/Job ID:\s*([0-9a-f]{32})/i);
    return match ? match[1].toLowerCase() : "";
  }

  function updateVisibility() {
    const text = jobBox.textContent.toLowerCase();
    const id = jobId();
    const stopped = ["status: done", "status: error", "status: needs_review", "status: budget_wait", "status: interrupted", "status: cancelled"].some(value => text.includes(value));
    actions.hidden = !id || stopped || jobBox.hidden;
    if (!actions.hidden) {
      button.disabled = false;
      button.textContent = "Stop OpenAI job now";
    }
  }

  button.addEventListener("click", async () => {
    const id = jobId();
    if (!id) return;
    button.disabled = true;
    button.textContent = "Stopping…";
    try {
      const response = await fetch("/openai/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: id }),
        cache: "no-store",
      });
      const text = await response.text();
      let value;
      try { value = JSON.parse(text); } catch { value = { error: text }; }
      if (!response.ok) throw new Error(value?.error || `HTTP ${response.status}`);
      jobBox.textContent = `${jobBox.textContent}\n\nSTOP REQUESTED: no further OpenAI planner/generator/auditor stages will be started.`;
      button.textContent = "Stopped";
      actions.hidden = true;
    } catch (error) {
      button.disabled = false;
      button.textContent = "Stop OpenAI job now";
      window.alert(`Could not stop OpenAI job: ${error.message}`);
    }
  });

  new MutationObserver(updateVisibility).observe(jobBox, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    attributeFilter: ["hidden"],
  });
  updateVisibility();
})();
