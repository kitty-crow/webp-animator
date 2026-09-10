from __future__ import annotations

import sys


LOOPED_TOKEN = "looped-animation"


UI_PATCH = r'''
/* looped-animation-ui-v1 */
(() => {
  "use strict";

  const TOKEN = "looped-animation";
  const tokenise = value => String(value || "")
    .split(",")
    .map(part => part.trim().toLowerCase())
    .filter(Boolean);

  function install(attempt = 0) {
    const form = document.getElementById("form");
    const target = document.getElementById("targetGaps");
    const select = document.getElementById("targetGapSelect");
    const loopAnalysis = document.getElementById("loopAnalysis");
    if (!form || !target || !select || !loopAnalysis) {
      if (attempt < 20) setTimeout(() => install(attempt + 1), 50);
      return;
    }
    if (document.getElementById("loopedAnimation")) return;

    const wrapper = document.createElement("label");
    wrapper.className = "wide";
    wrapper.style.display = "grid";
    wrapper.style.gap = "7px";
    wrapper.innerHTML = `
      <span class="checkbox" style="margin-top:0">
        <input id="loopedAnimation" type="checkbox">
        <strong>Looped animation</strong> · process Frame ${Math.max(1, select.options.length - 1)} ↔ Frame 1
      </span>
      <span class="hint">Always treat the last → first seam as a real temporal gap for gap-generation/interpolation passes. This is independent of Smart missing and its score threshold.</span>`;

    const checkbox = wrapper.querySelector("#loopedAnimation");
    const loopAnalysisLabel = loopAnalysis.closest("label");
    if (loopAnalysisLabel) {
      const hint = loopAnalysisLabel.querySelector(".hint");
      if (hint) hint.textContent = "Smart-missing scoring only. Use Looped animation below when the last → first seam must always be processed.";
      loopAnalysisLabel.insertAdjacentElement("afterend", wrapper);
    } else {
      select.closest("label")?.insertAdjacentElement("afterend", wrapper);
    }

    function ensureMarker() {
      let marker = [...select.options].find(option => option.value === TOKEN);
      if (!marker) {
        marker = document.createElement("option");
        marker.value = TOKEN;
        marker.textContent = "Looped animation marker";
        marker.hidden = true;
        select.appendChild(marker);
      }
      const enabled = tokenise(target.value).includes(TOKEN);
      marker.selected = enabled;
      checkbox.checked = enabled;
      const visible = [...select.options].filter(option => option.value !== TOKEN);
      const sourceCount = Math.max(1, visible.filter(option => option.value !== "loop").length + 1);
      const text = wrapper.querySelector("strong")?.parentElement;
      if (text) {
        text.lastChild.textContent = ` · process Frame ${sourceCount} ↔ Frame 1`;
      }
      return marker;
    }

    checkbox.addEventListener("change", () => {
      const marker = ensureMarker();
      marker.selected = checkbox.checked;
      // Reuse the advanced UI's existing selector synchronisation so the durable
      // job stores this alongside the other gap selections.
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    const observer = new MutationObserver(() => ensureMarker());
    observer.observe(select, { childList: true });
    ensureMarker();
  }

  if (document.readyState === "loading") {
    window.addEventListener("DOMContentLoaded", () => setTimeout(() => install(), 0));
  } else {
    setTimeout(() => install(), 0);
  }
})();
'''.strip()


def extract_looped_token(value) -> tuple[bool, str]:
    """Return explicit-loop state plus target-gap text with our UI marker removed."""
    if value is None:
        return False, ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        parts = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
    enabled = any(part.lower() == LOOPED_TOKEN for part in parts)
    kept = [part for part in parts if part.lower() != LOOPED_TOKEN]
    return enabled, ",".join(kept)


def _install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "WORKSPACE_SCRIPT"):
        return
    marker = b"/* looped-animation-ui-v1 */"
    current = bytes(app_all.WORKSPACE_SCRIPT)
    if marker in current:
        return
    app_all.WORKSPACE_SCRIPT = current + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"


def install(temporal_v2_module) -> None:
    """Make explicit loop processing independent of Smart-missing analysis.

    The advanced gap selector already persists arbitrary tokens in target_gaps. The
    dedicated UI marker is stripped before the established planner runs, so it does
    not accidentally turn the normal-gap planner into manual-only mode. We then add
    exactly one cyclic GapPlan. This gives conventional, smart and manually targeted
    jobs the same explicit last → first behaviour.
    """
    _install_ui_patch()
    if getattr(temporal_v2_module, "_looped_animation_installed", False):
        return
    temporal_v2_module._looped_animation_installed = True
    original = temporal_v2_module._plans_for_job

    def plans_for_job(records, settings, source_count):
        value = dict(settings or {})
        enabled, cleaned = extract_looped_token(value.get("target_gaps", ""))
        if not enabled:
            return original(records, value, source_count)

        value["target_gaps"] = cleaned
        plans, auto_scores, loop_score = original(records, value, source_count)
        plans = list(plans)
        if source_count >= 2 and len(records) >= 2 and not any(
            str(getattr(plan, "kind", "")).lower() == "loop" for plan in plans
        ):
            plans.append(
                temporal_v2_module.GapPlan(
                    "loop",
                    None,
                    max(0, int(source_count) - 1),
                    True,
                    loop_score,
                )
            )
        return plans, auto_scores, loop_score

    temporal_v2_module._plans_for_job = plans_for_job
