(() => {
  async function requestCachedRestore() {
    try {
      if (typeof frames !== "undefined" && frames.length) return;
      const id = window.webpAnimatorWorkspace?.currentId?.();
      if (!id) return;
      document.dispatchEvent(new CustomEvent("webp-global-job-restored", {
        detail: { id, metadata: null, cacheOnlyBootstrap: true },
      }));
    } catch {}
  }

  // workspace.js initialises asynchronously. Give it a moment to choose the
  // current Global Job ID, then ask the independent IndexedDB frame cache to
  // rehydrate the live frame array even when server metadata no longer exists.
  window.addEventListener("pageshow", () => setTimeout(requestCachedRestore, 800));
  setTimeout(requestCachedRestore, 1200);
})();
