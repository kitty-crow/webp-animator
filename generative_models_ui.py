from __future__ import annotations

import sys
from pathlib import Path


UI_PATCH = r'''
/* engine-catalog-ui-v4 */
(() => {
  "use strict";

  let engineState = null;
  let attempts = 0;

  function enginesFor(role) {
    if (!engineState || !Array.isArray(engineState.engines)) return [];
    return engineState.engines
      .filter(engine => engine && engine.ready === true && engine.role === role)
      .sort((left, right) =>
        Number(left.order || 0) - Number(right.order || 0) ||
        String(left.label || left.id || "").localeCompare(String(right.label || right.id || ""))
      );
  }

  function displayLabel(engine) {
    const prefix = engine.generative ? "Generative · " : "";
    return `${prefix}${engine.label || engine.id}`;
  }

  function populate(select, role) {
    const previous = select.value;
    const engines = enginesFor(role);
    select.replaceChildren();

    const none = document.createElement("option");
    none.value = "none";
    none.textContent = "None";
    select.append(none);

    for (const engine of engines) {
      const option = document.createElement("option");
      option.value = String(engine.id);
      option.textContent = displayLabel(engine);
      select.append(option);
    }

    if ([...select.options].some(option => option.value === previous)) {
      select.value = previous;
    }
  }

  function selectedEngine(select) {
    if (!select || !engineState || !Array.isArray(engineState.engines)) return null;
    return engineState.engines.find(engine => engine && engine.id === select.value) || null;
  }

  function bindHint(select, fallback) {
    const label = select.closest("label.option");
    let hint = label?.querySelector(".hint");
    if (!hint && label) {
      hint = document.createElement("span");
      hint.className = "hint";
      label.append(hint);
    }

    const update = () => {
      if (!hint) return;
      const engine = selectedEngine(select);
      hint.textContent = engine?.hint || fallback;
      const warningGb = Number(engine?.low_vram_warning_gb || 0);
      const vramBytes = Number(engineState?.acceleration?.total_vram?.[0] || 0);
      if (warningGb > 0 && vramBytes > 0 && vramBytes < warningGb * 1024 ** 3) {
        hint.textContent += ` This GPU has under ${warningGb} GB VRAM, so the model may fail even after reduced-resolution/offload fallbacks.`;
      }
    };

    if (select.dataset.engineCatalogBound !== "1") {
      select.dataset.engineCatalogBound = "1";
      select.addEventListener("change", update);
    }
    update();
  }

  function updateStatus() {
    const status = document.getElementById("rifeStatus");
    if (!status || !engineState || !Array.isArray(engineState.engines)) return;
    const available = engineState.engines.filter(engine => engine?.ready === true);
    status.textContent = available.length
      ? `Available engines: ${available.map(engine => engine.label || engine.id).join(" · ")}`
      : "No optional engines detected";
  }

  function install() {
    if (!engineState) return false;
    const interpolator = document.getElementById("interpolator");
    const generator = document.getElementById("frameGenerator");
    if (!interpolator || !generator) return false;

    populate(interpolator, "interpolator");
    populate(generator, "generator");
    bindHint(
      interpolator,
      "Select an installed interpolation engine for all gaps, manual targets, automatic missing gaps, or the loop seam.",
    );
    bindHint(
      generator,
      "Select an installed structural frame generator. It can be combined with an interpolator for further filling.",
    );
    updateStatus();

    document.dispatchEvent(new CustomEvent("webp-engine-catalog-ready", { detail: engineState }));
    return true;
  }

  function retryInstall() {
    attempts += 1;
    if (install() || attempts >= 200) return;
    setTimeout(retryInstall, 50);
  }

  fetch("/engine-status", { cache: "no-store" })
    .then(response => {
      if (!response.ok) throw new Error(`engine status failed (${response.status})`);
      return response.json();
    })
    .then(state => {
      engineState = state;
      retryInstall();
    })
    .catch(error => {
      console.error("Could not load WebP Animator engine catalog", error);
    });
})();
'''.strip()


def _app_all_module():
    module = sys.modules.get("app_all")
    if module is not None:
        return module
    module = sys.modules.get("__main__")
    filename = Path(str(getattr(module, "__file__", ""))).name.lower() if module else ""
    return module if filename == "app_all.py" else None


def install_ui_patch() -> None:
    app_all = _app_all_module()
    if app_all is None:
        return

    marker = b"/* engine-catalog-ui-v4 */"

    # app_all serves this script directly. Appending the catalog bootstrap here also
    # keeps direct /advanced-ui.js consumers in sync with the runtime engine registry.
    if hasattr(app_all, "ADVANCED_SCRIPT") and marker not in bytes(app_all.ADVANCED_SCRIPT):
        app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"

    # app_all's HTML gets a cache-busted advanced script plus an inline bootstrap so
    # selector discovery is independent of script execution order.
    if hasattr(app_all, "INDEX_HTML"):
        html = bytes(app_all.INDEX_HTML)
        html = html.replace(
            b'/advanced-ui.js"',
            b'/advanced-ui.js?v=engine-catalog-v4"',
        )
        if marker not in html:
            inline = b"\n<script>\n" + UI_PATCH.encode("utf-8") + b"\n</script>\n"
            html = html.replace(b"</body>", inline + b"</body>", 1)
        app_all.INDEX_HTML = html
