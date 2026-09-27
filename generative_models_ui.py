from __future__ import annotations

import sys


UI_PATCH = r'''
/* generative-models-ui-v3 */
(() => {
  "use strict";

  const models = [
    ["resshift", "Generative · Multi-Input ResShift Diffusion"],
    ["mog_ani", "Generative · MoG Animation"],
    ["mog_real", "Generative · MoG Real-world"],
    ["tooncrafter", "Generative · ToonCrafter (cartoon/anime)"],
  ];

  const hints = {
    resshift: "Endpoint-constrained residual diffusion. More expensive than RIFE/AMT, but better suited to difficult occlusion, articulation and missing-content transitions.",
    mog_ani: "Motion-aware generative interpolation tuned for animation. Very heavy; uses low-VRAM retries and may still exceed a 4 GB GPU.",
    mog_real: "Motion-aware generative interpolation tuned for photographic/real-world frames. Very heavy; uses low-VRAM retries and may still exceed a 4 GB GPU.",
    tooncrafter: "Generative cartoon interpolation from the two endpoint frames. Produces a full transition and selects the requested in-betweens from it. Extremely memory hungry on old GPUs.",
  };
  const conventionalHint = "Conventional temporal interpolation. Can work on all gaps, manual targets, automatic missing gaps, and the loop seam.";
  let engineState = null;

  function install() {
    const select = document.getElementById("interpolator");
    if (!select) return false;

    for (const [value, text] of models) {
      let node = select.querySelector(`option[value="${value}"]`);
      if (!node) {
        node = document.createElement("option");
        node.value = value;
        select.append(node);
      }
      node.textContent = text;
    }

    const label = select.closest("label.option");
    let hint = label?.querySelector(".hint");
    if (!hint && label) {
      hint = document.createElement("span");
      hint.className = "hint";
      label.append(hint);
    }

    function updateHint() {
      if (!hint) return;
      const value = select.value;
      hint.textContent = hints[value] || conventionalHint;
      const vram = Number(engineState?.acceleration?.total_vram?.[0] || 0);
      if (vram > 0 && vram < 6 * 1024 ** 3 && ["mog_ani", "mog_real", "tooncrafter"].includes(value)) {
        hint.textContent += " This GPU has under 6 GB VRAM, so the model may fail even after reduced-resolution/offload fallbacks.";
      }
    }

    if (select.dataset.generativeModelsBound !== "1") {
      select.dataset.generativeModelsBound = "1";
      select.addEventListener("change", updateHint);
    }
    updateHint();

    fetch("/engine-status", { cache: "no-store" })
      .then(response => response.ok ? response.json() : null)
      .then(state => {
        if (!state) return;
        engineState = state;
        updateHint();
        const status = document.getElementById("rifeStatus");
        if (!status) return;
        const extra = [
          `ResShift ${state.resshift?.ready ? "✓" : "missing"}`,
          `MoG Ani ${state.mog_ani?.ready ? "✓" : "missing"}`,
          `MoG Real ${state.mog_real?.ready ? "✓" : "missing"}`,
          `ToonCrafter ${state.tooncrafter?.ready ? "✓" : "missing"}`,
        ];
        const base = status.textContent.split(" · ResShift ")[0];
        status.textContent = `${base} · ${extra.join(" · ")}`;
      })
      .catch(() => {});

    return true;
  }

  if (install()) return;

  // advanced-ui.js creates the Interpolator selector dynamically. Do not depend on
  // script execution order: wait for it to exist and then install the generative
  // choices. This also makes the UI resilient to browser caching/reload timing.
  let attempts = 0;
  const timer = setInterval(() => {
    attempts += 1;
    if (install() || attempts >= 100) clearInterval(timer);
  }, 50);
})();
'''.strip()


def install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None:
        return

    marker = b"/* generative-models-ui-v3 */"

    # Keep a fallback copy in /advanced-ui.js for direct consumers of that script.
    if hasattr(app_all, "ADVANCED_SCRIPT") and marker not in bytes(app_all.ADVANCED_SCRIPT):
        app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"

    # More importantly, put the bootstrap directly in the served HTML. The base
    # advanced script is commonly cached by browsers on this local app, while the
    # HTML bootstrap retries until advanced-ui.js has created #interpolator.
    if hasattr(app_all, "INDEX_HTML"):
        html = bytes(app_all.INDEX_HTML)
        html = html.replace(
            b'/advanced-ui.js"',
            b'/advanced-ui.js?v=generative-models-v3"',
        )
        if marker not in html:
            inline = b"\n<script>\n" + UI_PATCH.encode("utf-8") + b"\n</script>\n"
            html = html.replace(b"</body>", inline + b"</body>", 1)
        app_all.INDEX_HTML = html
