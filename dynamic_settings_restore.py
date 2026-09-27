from __future__ import annotations

import sys


UI_PATCH = r'''
/* dynamic-settings-restore-ui-v1 */
(() => {
  "use strict";

  if (window.__webpDynamicSettingsRestoreV1) return;
  window.__webpDynamicSettingsRestoreV1 = true;

  const DB_NAME = "webp-animator-jobs";
  const DB_VERSION = 1;
  const STORE = "jobs";
  const LEGACY_GENERATIVE_MARKERS = {
    "__vfi_resshift__": "resshift",
    "__vfi_mog_ani__": "mog_ani",
    "__vfi_mog_real__": "mog_real",
    "__vfi_tooncrafter__": "tooncrafter",
  };

  function controlForKey(key) {
    const text = String(key || "");
    if (text.startsWith("id:")) return document.getElementById(text.slice(3));
    if (text.startsWith("name:")) {
      try { return document.querySelector(`[name="${CSS.escape(text.slice(5))}"]`); }
      catch { return null; }
    }
    try {
      return document.querySelector(`[name="${CSS.escape(text)}"]`) || document.getElementById(text);
    } catch {
      return document.getElementById(text);
    }
  }

  function setControl(control, value) {
    if (!control || control.type === "file" || control.id === "globalRestoreId") return;
    if (control.type === "checkbox" || control.type === "radio") {
      control.checked = Boolean(value);
      control.dispatchEvent(new Event("change", { bubbles: true }));
      return;
    }

    const wanted = String(value ?? "");
    if (control instanceof HTMLSelectElement && !control.multiple) {
      const exists = [...control.options].some(option => option.value === wanted);
      if (!exists && wanted) {
        control.dataset.pendingRestoreValue = wanted;
        return;
      }
      control.value = wanted;
      delete control.dataset.pendingRestoreValue;
      control.dispatchEvent(new Event("change", { bubbles: true }));
      return;
    }

    control.value = wanted;
    control.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function restoreTargetGapSelection() {
    const hidden = document.getElementById("targetGaps");
    const select = document.getElementById("targetGapSelect");
    if (!hidden || !select) return;
    const wanted = new Set(
      String(hidden.value || "")
        .split(",")
        .map(value => value.trim())
        .filter(Boolean),
    );
    [...select.options].forEach(option => {
      if (wanted.has(option.value)) option.selected = true;
    });
  }

  function applySettings(settings) {
    if (!settings || typeof settings !== "object") return false;
    for (const [key, value] of Object.entries(settings)) {
      setControl(controlForKey(key), value);
    }

    // Jobs saved while generative engines travelled through the historical AMT
    // compatibility token retain a marker in target_gaps. Recover the real selector
    // value after the runtime catalogue has populated its options.
    const target = document.getElementById("targetGaps");
    const interpolator = document.getElementById("interpolator");
    if (target && interpolator) {
      const text = String(target.value || "");
      for (const [marker, engine] of Object.entries(LEGACY_GENERATIVE_MARKERS)) {
        if (!text.includes(marker)) continue;
        if ([...interpolator.options].some(option => option.value === engine)) {
          interpolator.value = engine;
          delete interpolator.dataset.pendingRestoreValue;
          interpolator.dispatchEvent(new Event("change", { bubbles: true }));
        } else {
          interpolator.dataset.pendingRestoreValue = engine;
        }
        break;
      }
    }
    restoreTargetGapSelection();
    return true;
  }

  function openDb() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error("Could not open saved-job settings."));
    });
  }

  async function localSettings(id) {
    if (!id) return null;
    let db;
    try {
      db = await openDb();
      return await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readonly");
        const request = tx.objectStore(STORE).get(id);
        request.onsuccess = () => resolve(request.result?.settings || null);
        request.onerror = () => reject(request.error || new Error("Could not read saved-job settings."));
      });
    } catch {
      return null;
    } finally {
      try { db?.close(); } catch {}
    }
  }

  async function serverSettings(id) {
    if (!id) return null;
    try {
      const response = await fetch(`/job?id=${encodeURIComponent(id)}`, { cache: "no-store" });
      if (!response.ok) return null;
      const state = await response.json();
      return state?.settings && typeof state.settings === "object" ? state.settings : null;
    } catch {
      return null;
    }
  }

  let restoreGeneration = 0;
  async function restoreCurrent(preferred = null) {
    const generation = ++restoreGeneration;
    const id = String(window.webpAnimatorWorkspace?.currentId?.() || "").trim().toLowerCase();
    let settings = preferred;
    if (!settings) settings = await localSettings(id);
    if (!settings) settings = await serverSettings(id);
    if (generation !== restoreGeneration || !settings) return;
    applySettings(settings);
  }

  document.addEventListener("webp-global-job-restored", event => {
    const settings = event.detail?.metadata?.settings || null;
    restoreCurrent(settings);
  });

  document.addEventListener("webp-engine-catalog-ready", () => {
    restoreCurrent();
  });

  // workspace-ui.js is intentionally loaded before advanced-ui.js. Its automatic
  // restore can therefore finish before these dynamically-created controls exist.
  // Re-read the current saved job now that the controls do exist instead of relying
  // on event timing.
  setTimeout(() => restoreCurrent(), 0);
  setTimeout(() => restoreCurrent(), 300);
})();
'''.strip()


def install_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* dynamic-settings-restore-ui-v1 */"
    if marker in bytes(app_all.ADVANCED_SCRIPT):
        return
    app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"
