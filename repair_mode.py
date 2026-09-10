from __future__ import annotations

import sys


MANUAL_TOKEN = "repair-mode:manual"
AUTO_TOKEN = "repair-mode:auto"


UI_PATCH = r'''
/* openai-repair-mode-ui-v1 */
(() => {
  "use strict";

  const MANUAL = "repair-mode:manual";
  const AUTO = "repair-mode:auto";

  function parts(value) {
    return String(value || "").split(",").map(part => part.trim()).filter(Boolean);
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
    if (saved.includes(MANUAL)) mode.value = "manual";
    else mode.value = "auto";

    function refresh() {
      const active = engine.value === "openai";
      wrap.hidden = !active;
      hint.textContent = mode.value === "manual"
        ? "OpenAI will not run automatically in Repair passes. Open a generated frame, paint the defect, then press Repair marked frame with OpenAI."
        : "OpenAI starts automatically in Repair passes. Broad repaints are rejected; the deterministic candidate remains authoritative if a repair is unsafe.";
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

  // Keep the image canvas dominant on phones. The stage is explicitly fitted to
  // the remaining viewport after the header, toolbar and compact action row.
  const css = document.createElement("style");
  css.textContent = `
    .openai-repair-mode-wrap { display:grid; gap:6px; }
    .openai-repair-mode-wrap[hidden] { display:none !important; }
    .frame-detail-modal { padding:0 !important; }
    .frame-detail-dialog { height:100svh !important; max-height:100svh !important; grid-template-rows:auto minmax(0,1fr) auto !important; }
    .frame-detail-viewport { min-height:0 !important; overflow:hidden !important; padding:6px !important; }
    .frame-detail-stage { max-width:none !important; max-height:none !important; }
    .frame-detail-image { max-width:none !important; max-height:none !important; width:100% !important; height:100% !important; object-fit:contain !important; }
    .frame-detail-actions { padding:7px 10px !important; gap:7px !important; }
    .frame-detail-status { flex:1 1 100% !important; max-height:2.7em; overflow:auto; font-size:.72rem !important; line-height:1.3 !important; }
    .frame-detail-actions .primary { flex:1 1 auto; }
    @media (min-width:621px) {
      .frame-detail-modal { padding:12px !important; }
      .frame-detail-dialog { height:min(900px,calc(100svh - 24px)) !important; }
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
      const style = getComputedStyle(viewport);
      const availableWidth = Math.max(1, viewport.clientWidth - parseFloat(style.paddingLeft || 0) - parseFloat(style.paddingRight || 0));
      const availableHeight = Math.max(1, viewport.clientHeight - parseFloat(style.paddingTop || 0) - parseFloat(style.paddingBottom || 0));
      const scale = Math.min(1, availableWidth / image.naturalWidth, availableHeight / image.naturalHeight);
      stage.style.width = `${Math.max(1, Math.floor(image.naturalWidth * scale))}px`;
      stage.style.height = `${Math.max(1, Math.floor(image.naturalHeight * scale))}px`;
      image.style.width = "100%";
      image.style.height = "100%";
    }

    image.addEventListener("load", () => requestAnimationFrame(fit));
    new ResizeObserver(() => requestAnimationFrame(fit)).observe(viewport);
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
    marker = b"/* openai-repair-mode-ui-v1 */"
    current = bytes(app_all.WORKSPACE_SCRIPT)
    if marker in current:
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
