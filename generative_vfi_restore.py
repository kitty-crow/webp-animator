from __future__ import annotations

import sys


UI_PATCH = r'''
/* generative-vfi-restore-ui-v2 */
(() => {
  "use strict";
  const markers = {
    "__vfi_resshift__": "resshift",
    "__vfi_mog_ani__": "mog_ani",
    "__vfi_mog_real__": "mog_real",
    "__vfi_tooncrafter__": "tooncrafter",
  };

  function restoreFromSettings(settings) {
    const select = document.getElementById("interpolator");
    if (!select || !settings) return;
    const targets = String(settings.target_gaps || "");
    for (const [marker, value] of Object.entries(markers)) {
      if (!targets.includes(marker)) continue;
      if (select.value !== value) {
        select.value = value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      return;
    }
  }

  document.addEventListener("webp-global-job-restored", event => {
    restoreFromSettings(event.detail?.metadata?.settings || null);
  });
})();
'''.strip()


def install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* generative-vfi-restore-ui-v2 */"
    if marker in bytes(app_all.ADVANCED_SCRIPT):
        return
    app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"
