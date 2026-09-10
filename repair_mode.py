from __future__ import annotations

import sys


MANUAL_TOKEN = "repair-mode:manual"
AUTO_TOKEN = "repair-mode:auto"


UI_PATCH = r'''
/* openai-repair-mode-ui-v2 */
(() => {
  "use strict";

  const MANUAL = "repair-mode:manual";
  const AUTO = "repair-mode:auto";
  let selectedSource = "editor";
  let selectedTimelineIndex = -1;

  function parts(value) {
    return String(value || "").split(",").map(part => part.trim()).filter(Boolean);
  }

  function currentJobId() {
    try {
      return String(
        window.webpAnimatorWorkspace?.currentId?.()
        || localStorage.getItem("webp-animator-active-job")
        || ""
      ).trim().toLowerCase();
    } catch {
      return String(window.webpAnimatorWorkspace?.currentId?.() || "").trim().toLowerCase();
    }
  }

  function install(attempt = 0) {
    const control = document.getElementById("temporalRepairControl");
    const engine = document.getElementById("temporalRepairEngine");
    const form = document.getElementById("form");
    const target = document.getElementById("targetGaps");
    if (!control || !engine || !form || !target) {
      if (attempt < 40) setTimeout(() => install(attempt + 1), 50);
      return;
    }
    if (document.getElementById("openaiRepairMode")) return;

    const wrap = document.createElement("span");
    wrap.id = "openaiRepairModeWrap";
    wrap.className = "openai-repair-mode-wrap";
    wrap.innerHTML = `<span>OpenAI repair mode</span><select id="openaiRepairMode" aria-label="OpenAI repair mode"><option value="auto">Auto · repair generated frames during the run</option><option value="manual">Manual · wait for painted defect selections</option></select><span class="hint" id="openaiRepairModeHint"></span>`;
    const mode = wrap.querySelector("#openaiRepairMode");
    const hint = wrap.querySelector("#openaiRepairModeHint");
    control.insertBefore(wrap, document.getElementById("openaiRepairStatus") || null);

    const saved = parts(target.value).map(value => value.toLowerCase());
    mode.value = saved.includes(MANUAL) ? "manual" : "auto";

    function refresh() {
      const active = engine.value === "openai";
      wrap.hidden = !active;
      hint.textContent = mode.value === "manual"
        ? "OpenAI does not run automatically. Finish the deterministic run, open a generated frame, paint the defect, then repair that marked area. The finished WebP is rebuilt after each accepted manual repair."
        : "OpenAI repairs generated frames automatically. Broad redraws are rejected and the deterministic candidate is kept instead.";
    }

    engine.addEventListener("change", refresh);
    mode.addEventListener("change", refresh);
    target.addEventListener("change", () => {
      const values = parts(target.value).map(value => value.toLowerCase());
      if (values.includes(MANUAL)) mode.value = "manual";
      else if (values.includes(AUTO)) mode.value = "auto";
      refresh();
    });

    form.addEventListener("formdata", event => {
      const data = event.formData;
      const values = parts(data.get("target_gaps")).filter(value => !value.toLowerCase().startsWith("repair-mode:"));
      if (engine.value === "openai") values.push(mode.value === "manual" ? MANUAL : AUTO);
      data.set("target_gaps", values.join(","));
    });

    // Remember which modal source was opened. This allows the generic repair HTTP
    // request to persist repairs made against the durable live timeline without
    // coupling the modal module to the job store implementation.
    document.addEventListener("click", event => {
      const liveImage = event.target.closest?.("#liveFrameGrid .live-frame-card img");
      if (liveImage) {
        const cards = [...document.querySelectorAll("#liveFrameGrid .live-frame-card")];
        selectedSource = "live";
        selectedTimelineIndex = cards.indexOf(liveImage.closest(".live-frame-card"));
        return;
      }
      if (event.target.closest?.("#frameStrip .frame-card .thumb")) {
        selectedSource = "editor";
        selectedTimelineIndex = -1;
      }
    }, true);

    // Add mode/persistence metadata to the repair call. Other fetches are untouched.
    if (!window.fetch.__openaiRepairModeWrapped) {
      const nativeFetch = window.fetch.bind(window);
      const wrappedFetch = (input, init = {}) => {
        const url = typeof input === "string" ? input : String(input?.url || "");
        const body = init?.body;
        if (url.split("?", 1)[0].endsWith("/openai-repair") && body instanceof FormData) {
          body.set("repair_mode", mode.value);
          if (selectedSource === "live" && selectedTimelineIndex >= 0) {
            const jobId = currentJobId();
            if (jobId) {
              body.set("job_id", jobId);
              body.set("timeline_index", String(selectedTimelineIndex));
            }
          }
        }
        return nativeFetch(input, init);
      };
      wrappedFetch.__openaiRepairModeWrapped = true;
      window.fetch = wrappedFetch;
    }

    // Manual really means manual: a painted region is mandatory for a direct repair.
    document.addEventListener("click", event => {
      const button = event.target.closest?.(".frame-detail-modal [data-repair]");
      if (!button || engine.value !== "openai" || mode.value !== "manual") return;
      const canvas = document.querySelector(".frame-detail-modal [data-mask]");
      if (!canvas?.width || !canvas?.height) return;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let marked = false;
      for (let i = 3; i < pixels.length; i += 4) {
        if (pixels[i] > 0) { marked = true; break; }
      }
      if (marked) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      const status = document.querySelector(".frame-detail-modal [data-detail-status]");
      if (status) status.textContent = "Manual mode: paint the defect area first. OpenAI will only be allowed to alter the marked region.";
    }, true);

    refresh();
  }

  // The canvas owns the available middle row. Explanatory text is deliberately
  // compact so it cannot push a tall portrait frame out of view on a phone.
  const css = document.createElement("style");
  css.textContent = `
    .openai-repair-mode-wrap { display:grid; gap:6px; }
    .openai-repair-mode-wrap[hidden] { display:none !important; }
    .frame-detail-modal { padding:0 !important; }
    .frame-detail-dialog { width:100% !important; height:100svh !important; max-height:100svh !important; grid-template-rows:auto minmax(0,1fr) auto !important; border-radius:0 !important; }
    .frame-detail-viewport { min-width:0 !important; min-height:0 !important; overflow:hidden !important; padding:6px !important; }
    .frame-detail-stage { max-width:none !important; max-height:none !important; }
    .frame-detail-image { display:block !important; max-width:none !important; max-height:none !important; width:100% !important; height:100% !important; object-fit:contain !important; }
    .frame-detail-actions { padding:6px 10px calc(6px + env(safe-area-inset-bottom)) !important; gap:6px !important; }
    .frame-detail-status { flex:1 1 100% !important; max-height:2.55em; overflow:auto; font-size:.7rem !important; line-height:1.25 !important; }
    .frame-detail-actions .primary { flex:1 1 auto; }
    @media (min-width:621px) {
      .frame-detail-modal { padding:12px !important; }
      .frame-detail-dialog { width:min(1180px,100%) !important; height:min(900px,calc(100svh - 24px)) !important; border-radius:16px !important; }
      .frame-detail-status { flex:1 1 260px !important; max-height:4em; }
    }
  `;
  document.head.appendChild(css);

  function installFitter(attempt = 0) {
    const modal = document.querySelector(".frame-detail-modal");
    const viewport = modal?.querySelector(".frame-detail-viewport");
    const stage = modal?.querySelector("[data-stage]");
    const image = modal?.querySelector("[data-detail-image]");
    if (!modal || !viewport || !stage || !image) {
      if (attempt < 60) setTimeout(() => installFitter(attempt + 1), 50);
      return;
    }
    if (image.dataset.fitInstalled === "1") return;
    image.dataset.fitInstalled = "1";

    function fit() {
      if (modal.hidden || !image.naturalWidth || !image.naturalHeight) return;
      const viewportStyle = getComputedStyle(viewport);
      const availableWidth = Math.max(1, viewport.clientWidth - parseFloat(viewportStyle.paddingLeft || 0) - parseFloat(viewportStyle.paddingRight || 0));
      const availableHeight = Math.max(1, viewport.clientHeight - parseFloat(viewportStyle.paddingTop || 0) - parseFloat(viewportStyle.paddingBottom || 0));
      const fitScale = Math.min(1, availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
      stage.style.width = `${Math.max(1, Math.floor(image.naturalWidth * fitScale))}px`;
      stage.style.height = `${Math.max(1, Math.floor(image.naturalHeight * fitScale))}px`;
      image.style.width = "100%";
      image.style.height = "100%";
    }

    image.addEventListener("load", () => requestAnimationFrame(fit));
    if (typeof ResizeObserver === "function") new ResizeObserver(() => requestAnimationFrame(fit)).observe(viewport);
    new MutationObserver(() => { if (!modal.hidden) requestAnimationFrame(fit); }).observe(modal, { attributes:true, attributeFilter:["hidden"] });
    window.addEventListener("resize", () => requestAnimationFrame(fit));
  }

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => { install(); installFitter(); });
  } else {
    install();
    installFitter();
  }
})();
'''.strip()


def repair_mode(settings: dict | None) -> str:
    if not isinstance(settings, dict):
        return "auto"
    explicit = str(settings.get("openai_repair_mode", "")).strip().lower()
    if explicit in {"auto", "manual"}:
        return explicit
    parts = {
        part.strip().lower()
        for part in str(settings.get("target_gaps", "")).replace(";", ",").split(",")
        if part.strip()
    }
    return "manual" if MANUAL_TOKEN in parts else "auto"


def _install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "WORKSPACE_SCRIPT"):
        return
    current = bytes(app_all.WORKSPACE_SCRIPT)
    if b"/* openai-repair-mode-ui-v2 */" in current:
        return
    app_all.WORKSPACE_SCRIPT = current + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"


def install(operation_pipeline_module) -> None:
    _install_ui_patch()
    if getattr(operation_pipeline_module, "_repair_mode_installed", False):
        return
    operation_pipeline_module._repair_mode_installed = True
    original = operation_pipeline_module._repair_pass

    def repair_pass(records, stage_dir, *, settings, source_count, pass_number, pass_total, progress):
        if (
            operation_pipeline_module.repair_engine(settings) == "openai"
            and repair_mode(settings) == "manual"
        ):
            progress(
                1.0,
                f"Pass {pass_number}/{pass_total} · OpenAI manual repair deferred until you mark defects",
            )
            return {
                "engine": "openai",
                "mode": "manual",
                "deferred": True,
                "audited": 0,
                "repaired": 0,
                "skipped": [],
            }
        result = original(
            records,
            stage_dir,
            settings=settings,
            source_count=source_count,
            pass_number=pass_number,
            pass_total=pass_total,
            progress=progress,
        )
        if isinstance(result, dict) and operation_pipeline_module.repair_engine(settings) == "openai":
            result.setdefault("mode", "auto")
        return result

    operation_pipeline_module._repair_pass = repair_pass
