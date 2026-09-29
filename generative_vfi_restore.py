from __future__ import annotations

import sys
from pathlib import Path


UI_PATCH = r'''
/* generative-vfi-restore-ui-v3 */
(() => {
  "use strict";
  const markers = {
    "__vfi_resshift__": "resshift",
    "__vfi_mog_ani__": "mog_ani",
    "__vfi_mog_real__": "mog_real",
    "__vfi_tooncrafter__": "tooncrafter",
  };
  let pendingSettings = null;

  function restoreFromSettings(settings) {
    if (!settings) return false;
    const select = document.getElementById("interpolator");
    if (!select) return false;
    const targets = String(settings.target_gaps || "");
    for (const [marker, value] of Object.entries(markers)) {
      if (!targets.includes(marker)) continue;
      if (![...select.options].some(option => option.value === value)) return false;
      if (select.value !== value) {
        select.value = value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return true;
    }
    return true;
  }

  document.addEventListener("webp-global-job-restored", event => {
    pendingSettings = event.detail?.metadata?.settings || null;
    if (restoreFromSettings(pendingSettings)) pendingSettings = null;
  });

  document.addEventListener("webp-engine-catalog-ready", () => {
    if (pendingSettings && restoreFromSettings(pendingSettings)) pendingSettings = null;
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
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* generative-vfi-restore-ui-v3 */"
    if marker in bytes(app_all.ADVANCED_SCRIPT):
        return
    app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"
