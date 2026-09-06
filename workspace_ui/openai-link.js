(() => {
  const linked = new Set();

  async function linkOpenAIJob(openaiJobId) {
    const workspace = window.webpAnimatorWorkspace;
    const globalId = workspace?.currentId?.();
    if (!globalId || !/^[0-9a-f]{32}$/i.test(openaiJobId || "")) return;
    const key = `${globalId}:${openaiJobId}`;
    if (linked.has(key)) return;
    try {
      const response = await fetch("/job/link-openai", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: globalId, openai_job_id: openaiJobId }),
      });
      if (response.ok) linked.add(key);
    } catch {}
  }

  function scanOpenAIJobBox() {
    const box = document.getElementById("oiJob");
    if (!box) return;
    const match = box.textContent.match(/Job ID:\s*([0-9a-f]{32})/i);
    if (match) linkOpenAIJob(match[1].toLowerCase());
  }

  const box = document.getElementById("oiJob");
  if (box) {
    new MutationObserver(scanOpenAIJobBox).observe(box, { childList: true, subtree: true, characterData: true });
    scanOpenAIJobBox();
  }

  document.addEventListener("webp-global-job-restored", event => {
    const openaiId = event.detail?.lastOpenAIJobId;
    if (!openaiId) return;
    const input = document.getElementById("oiRestoreId");
    const button = document.getElementById("oiRestore");
    if (!input || !button) return;
    input.value = openaiId;
    button.click();
  });
})();