from __future__ import annotations

import sys


UI_PATCH = r'''
/* generative-models-ui-v2 */
(() => {
  "use strict";
  const select = document.getElementById("interpolator");
  if (!select) return;

  const models = [
    ["resshift", "Generative · Multi-Input ResShift Diffusion"],
    ["mog_ani", "Generative · MoG Animation"],
    ["mog_real", "Generative · MoG Real-world"],
    ["tooncrafter", "Generative · ToonCrafter (cartoon/anime)"],
  ];
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
  const conventionalHint = "Conventional temporal interpolation. Can work on all gaps, manual targets, automatic missing gaps, and the loop seam.";
  const hints = {
    resshift: "Endpoint-constrained residual diffusion. More expensive than RIFE/AMT, but better suited to difficult occlusion, articulation and missing-content transitions.",
    mog_ani: "Motion-aware generative interpolation tuned for animation. Very heavy; uses low-VRAM retries and may still exceed a 4 GB GPU.",
    mog_real: "Motion-aware generative interpolation tuned for photographic/real-world frames. Very heavy; uses low-VRAM retries and may still exceed a 4 GB GPU.",
    tooncrafter: "Generative cartoon interpolation from the two endpoint frames. Produces a full 16-frame transition and the requested in-betweens are selected from it. Extremely memory hungry on old GPUs.",
  };

  let engineState = null;
  function updateHint() {
    if (!hint) return;
    const value = select.value;
    hint.textContent = hints[value] || conventionalHint;
    const vram = Number(engineState?.acceleration?.total_vram?.[0] || 0);
    if (vram > 0 && vram < 6 * 1024 ** 3 && ["mog_ani", "mog_real", "tooncrafter"].includes(value)) {
      hint.textContent += " This GPU has under 6 GB VRAM, so the model may fail even after reduced-resolution/offload fallbacks.";
    }
  }
  select.addEventListener("change", updateHint);
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
})();
'''.strip()


def install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* generative-models-ui-v2 */"
    if marker not in bytes(app_all.ADVANCED_SCRIPT):
        app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"

    # The browser may have cached /advanced-ui.js from a previous local deployment.
    # Query strings are ignored by the server route but force a fresh script fetch.
    if hasattr(app_all, "INDEX_HTML"):
        app_all.INDEX_HTML = bytes(app_all.INDEX_HTML).replace(
            b'/advanced-ui.js"',
            b'/advanced-ui.js?v=generative-models-v2"',
        )
