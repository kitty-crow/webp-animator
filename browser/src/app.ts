import { AdaptiveAlignmentEngine, renderUnionFrames } from './alignment.js';
import { decodeInputFiles } from './frame-codec.js';
import { gpuLabel } from './hardware.js';
import { runPreflight } from './preflight.js';
import type { PreflightResult, ProgressUpdate, RegistrationSettings } from './types.js';
import { encodeAnimatedWebp } from './webp-muxer.js';

function element<T extends HTMLElement>(id: string, constructor: { new (): T }): T {
  const candidate = document.getElementById(id);
  if (!(candidate instanceof constructor)) throw new Error(`Missing or invalid #${id}.`);
  return candidate;
}

function numberInput(id: string): HTMLInputElement {
  return element(id, HTMLInputElement);
}

function selectInput(id: string): HTMLSelectElement {
  return element(id, HTMLSelectElement);
}

const fileInput = numberSafeFileInput('frames');
const runButton = element('run', HTMLButtonElement);
const progress = element('progress', HTMLProgressElement);
const progressText = element('progress-text', HTMLParagraphElement);
const status = element('status', HTMLParagraphElement);
const preview = element('preview', HTMLImageElement);
const download = element('download', HTMLAnchorElement);
const results = element('results', HTMLPreElement);
const preflightStatus = element('preflight-status', HTMLSpanElement);
const footerResources = element('resource-footer', HTMLDivElement);

let preflight: PreflightResult | null = null;
let resultUrl: string | null = null;

function numberSafeFileInput(id: string): HTMLInputElement {
  const input = element(id, HTMLInputElement);
  if (input.type !== 'file') throw new Error(`#${id} must be a file input.`);
  return input;
}

function settings(): RegistrationSettings {
  const axisRaw = selectInput('axis').value;
  const axis: RegistrationSettings['axis'] = axisRaw === 'x' || axisRaw === 'y' || axisRaw === 'none' ? axisRaw : 'xy';
  return {
    axis,
    maxShiftX: Math.max(0, Number(numberInput('max-shift-x').value) || 0),
    maxShiftY: Math.max(0, Number(numberInput('max-shift-y').value) || 0),
    sigma: Math.max(0.01, Number(numberInput('sigma').value) || 28),
    alphaThreshold: Math.max(0, Math.min(255, Math.round(Number(numberInput('alpha-threshold').value) || 0))),
    proxyMaxSide: Math.max(32, Math.round(Number(numberInput('proxy-max-side').value) || 256)),
  };
}

function setStatus(message: string, kind: 'neutral' | 'error' | 'success' = 'neutral'): void {
  status.textContent = message;
  status.dataset.kind = kind;
}

function setProgress(update: ProgressUpdate): void {
  const total = Math.max(1, update.total);
  progress.value = Math.min(1, update.current / total);
  const label = update.stage === 'decode'
    ? `Decoding${update.fileName ? ` ${update.fileName}` : ''}`
    : update.stage === 'align'
      ? 'Aligning frames'
      : 'Encoding WebP';
  progressText.textContent = `${label} · ${update.current}/${update.total}`;
}

function benchmarkLabel(result: PreflightResult): string {
  const successful = result.benchmarks
    .filter((entry) => entry.passed)
    .map((entry) => `${entry.backend} ${entry.milliseconds.toFixed(1)} ms`)
    .join(' · ');
  return successful || 'No accelerated benchmark passed';
}

function populateFooter(result: PreflightResult): void {
  const profile = result.profile;
  const ram = profile.deviceMemoryGiB !== null
    ? `~${profile.deviceMemoryGiB} GiB device memory`
    : profile.heapLimitMiB !== null
      ? `${profile.heapLimitMiB} MiB JS heap limit`
      : 'RAM estimate unavailable';
  const threading = profile.sharedMemory
    ? `${profile.workerCount} shared-memory workers`
    : profile.workerSupport
      ? `${profile.workerCount} copied-buffer workers`
      : 'single-threaded browser';
  const webgl = profile.webgl2.available ? `WebGL2 ${profile.webgl2.renderer ?? 'adapter'}` : 'WebGL2 unavailable';
  const simd = profile.wasmSimd ? 'WASM SIMD' : profile.wasm ? 'scalar WASM' : 'WASM unavailable';

  footerResources.replaceChildren();
  const selected = document.createElement('strong');
  selected.textContent = `Selected automatically: ${result.selectedBackend}`;
  const details = document.createElement('span');
  details.textContent = `${gpuLabel(profile)} · ${webgl} · ${profile.logicalCores} logical CPU cores · ${threading} · ${ram} · ${profile.memoryTier} memory tier · ${profile.memoryBudgetMiB} MiB working budget · ${simd}`;
  const benchmarks = document.createElement('span');
  benchmarks.textContent = `Pre-flight: ${benchmarkLabel(result)}`;
  footerResources.append(selected, details, benchmarks);
}

async function ensureIsolationServiceWorker(): Promise<void> {
  if (!('serviceWorker' in navigator) || !location.protocol.startsWith('http')) return;
  try {
    await navigator.serviceWorker.register('./coi-serviceworker.js', { scope: './' });
    if (!crossOriginIsolated && !navigator.serviceWorker.controller && sessionStorage.getItem('webp-coi-reload') !== '1') {
      sessionStorage.setItem('webp-coi-reload', '1');
      await navigator.serviceWorker.ready;
      location.reload();
    } else if (crossOriginIsolated) {
      sessionStorage.removeItem('webp-coi-reload');
    }
  } catch (error: unknown) {
    console.warn('Cross-origin isolation service worker could not be enabled.', error);
  }
}

async function initialise(): Promise<void> {
  preflightStatus.textContent = 'Running automatic hardware pre-flight…';
  await ensureIsolationServiceWorker();
  try {
    preflight = await runPreflight();
    populateFooter(preflight);
    preflightStatus.textContent = `Ready · ${preflight.selectedBackend}`;
    runButton.disabled = !fileInput.files?.length;
  } catch (error: unknown) {
    preflightStatus.textContent = 'Pre-flight failed; compatibility CPU path will be used.';
    setStatus(error instanceof Error ? error.message : String(error), 'error');
  }
}

fileInput.addEventListener('change', () => {
  runButton.disabled = !(fileInput.files?.length) || preflight === null;
});

runButton.addEventListener('click', () => {
  void (async (): Promise<void> => {
    const activePreflight = preflight;
    const files = fileInput.files ? Array.from(fileInput.files) : [];
    if (!activePreflight || files.length === 0) return;

    runButton.disabled = true;
    progress.value = 0;
    setStatus('Processing entirely on this device…');
    const engine = await AdaptiveAlignmentEngine.create(activePreflight.profile, activePreflight.selectedBackend);
    try {
      const decoded = await decodeInputFiles(files, setProgress);
      const registration = await engine.registerSequence(decoded.frames, settings(), (current, total) => {
        setProgress({ stage: 'align', current, total });
      });
      const rendered = renderUnionFrames(decoded.frames, registration.positions);
      const duration = Math.max(1, Math.round(Number(numberInput('duration').value) || 100));
      const loop = Math.max(0, Math.round(Number(numberInput('loop').value) || 0));
      const quality = Math.max(0.01, Math.min(1, (Number(numberInput('quality').value) || 95) / 100));
      const blob = await encodeAnimatedWebp(rendered, {
        duration,
        durations: decoded.sourceDurations,
        loop,
        quality,
      }, (current, total) => setProgress({ stage: 'encode', current, total }));

      if (resultUrl) URL.revokeObjectURL(resultUrl);
      resultUrl = URL.createObjectURL(blob);
      preview.src = resultUrl;
      preview.hidden = false;
      download.href = resultUrl;
      download.download = 'animation.webp';
      download.hidden = false;
      results.textContent = registration.pairwise
        .map((entry, index) => `frame ${index + 2}: dx=${entry.dx}, dy=${entry.dy}, score=${entry.score.toFixed(6)}`)
        .join('\n');
      progress.value = 1;
      progressText.textContent = 'Complete';
      setStatus(`Finished locally with ${activePreflight.selectedBackend}.`, 'success');
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : String(error), 'error');
    } finally {
      engine.close();
      runButton.disabled = false;
    }
  })();
});

void initialise();
