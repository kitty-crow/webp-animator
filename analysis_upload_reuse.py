from __future__ import annotations

import sys


UI_PATCH = r'''
/* analysis-upload-reuse-ui-v1 */
(() => {
  "use strict";

  function hex(bytes) {
    return [...new Uint8Array(bytes)].map(value => value.toString(16).padStart(2, "0")).join("");
  }

  async function frameHashes() {
    const list = (() => { try { return Array.isArray(frames) ? frames : []; } catch { return []; } })();
    const hashes = [];
    for (const item of list) {
      if (!(item?.file instanceof Blob)) return null;
      hashes.push(hex(await crypto.subtle.digest("SHA-256", await item.file.arrayBuffer())));
    }
    return hashes;
  }

  async function matchingAnalysis(jobId) {
    if (!jobId || !crypto?.subtle) return null;
    try {
      const response = await fetch(`/analysis?id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
      if (!response.ok) return null;
      const state = await response.json();
      if (state.status !== "done" || !state.result) return null;
      const expected = Array.isArray(state.result.source_sha256) ? state.result.source_sha256.map(String) : [];
      if (!expected.length) return null;
      const actual = await frameHashes();
      if (!actual || actual.length !== expected.length) return null;
      if (!actual.every((value, index) => value.toLowerCase() === expected[index].toLowerCase())) return null;
      return expected;
    } catch {
      return null;
    }
  }

  async function sendReuse(data, hashes) {
    data.delete("frames");
    data.set("reuse_analysis_sources", "on");
    data.set("analysis_source_sha256", hashes.join(","));
    try { window.webpUploadProgress?.setRatio?.(0, 0, "Using analysed frames already on server"); } catch {}
    const response = await fetch("/generate", { method: "POST", body: data, cache: "no-store" });
    const text = await response.text();
    if (!response.ok) throw new Error(text || `Generation request failed (${response.status})`);
    let value;
    try { value = JSON.parse(text); }
    catch { throw new Error("Server returned invalid generation JSON."); }
    try { window.webpUploadProgress?.complete?.(0, "Reused analysed frames · no upload"); } catch {}
    return value;
  }

  function install(attempt = 0) {
    const original = window.uploadJob;
    if (typeof original !== "function") {
      if (attempt < 60) setTimeout(() => install(attempt + 1), 50);
      return;
    }
    if (original.__analysisReuseWrapped) return;

    const wrapped = async data => {
      const jobId = String(data.get("global_job_id") || window.webpAnimatorWorkspace?.currentId?.() || "").trim().toLowerCase();
      const hashes = await matchingAnalysis(jobId);
      if (hashes) return await sendReuse(data, hashes);
      return await original(data);
    };
    wrapped.__analysisReuseWrapped = true;
    window.uploadJob = wrapped;
    try { uploadJob = wrapped; } catch {}
  }

  // advanced-ui.js defines uploadJob earlier in the same script. Appending this
  // patch to ADVANCED_SCRIPT means install can normally wrap immediately.
  install();
})();
'''.strip()


def install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* analysis-upload-reuse-ui-v1 */"
    current = bytes(app_all.ADVANCED_SCRIPT)
    if marker in current:
        return
    app_all.ADVANCED_SCRIPT = current + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"
