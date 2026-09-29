import { AdaptiveAlignmentEngine, renderUnionFrames } from './alignment.js';
import { decodeInputFiles } from './frame-codec.js';
import { gpuLabel } from './hardware.js';
import { InferenceWorkerClient, type RenderWorkerProgress } from './inference-worker-client.js';
import type { RenderInterpolationEngine } from './inference-worker-protocol.js';
import { interpolateAmtFrames } from './inference/amt-interpolation.js';
import { interpolateGenerativeFrames, type GenerativeInterpolationEngine } from './inference/generative-interpolation.js';
import { runPreflight } from './preflight.js';
import type { ComputeBackend, PreflightResult, ProgressUpdate, RegistrationSettings, ShiftResult } from './types.js';
import { encodeAnimatedWebp } from './webp-muxer.js';

type ManifestInterpolationEngine = Exclude<RenderInterpolationEngine, 'none' | 'rife'>;

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

function fileInputElement(id: string): HTMLInputElement {
  const input = element(id, HTMLInputElement);
  if (input.type !== 'file') throw new Error(`#${id} must be a file input.`);
  return input;
}

const fileInput = fileInputElement('frames');
const runButton = element('run', HTMLButtonElement);
const cancelButton = element('cancel', HTMLButtonElement);
const progress = element('progress', HTMLProgressElement);
const progressText = element('progress-text', HTMLParagraphElement);
const status = element('status', HTMLParagraphElement);
const preview = element('preview', HTMLImageElement);
const download = element('download', HTMLAnchorElement);
const results = element('results', HTMLPreElement);
const preflightStatus = element('preflight-status', HTMLSpanElement);
const footerResources = element('resource-footer', HTMLDivElement);
const manifestField = element('model-manifest-field', HTMLLabelElement);
const manifestInput = element('model-manifest-url', HTMLInputElement);
const engineInput = selectInput('interpolation-engine');
const inferenceWorker = new InferenceWorkerClient();

let preflight: PreflightResult | null = null;
let resultUrl: string | null = null;
let activeInferenceJobId: string | null = null;
let compatibilityJobActive = false;
let compatibilityCancelRequested = false;

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

function interpolationEngine(): RenderInterpolationEngine {
  const value = engineInput.value;
  if (value === 'rife' || value === 'amt' || value === 'resshift' || value === 'mog' || value === 'tooncrafter') return value;
  return 'none';
}

function isGenerative(engine: RenderInterpolationEngine): engine is GenerativeInterpolationEngine {
  return engine === 'resshift' || engine === 'mog' || engine === 'tooncrafter';
}

function requiresManifest(engine: RenderInterpolationEngine): engine is ManifestInterpolationEngine {
  return engine === 'amt' || isGenerative(engine);
}

function engineLabel(engine: RenderInterpolationEngine): string {
  if (engine === 'rife') return 'RIFE 4.25';
  if (engine === 'amt') return 'AMT-S';
  if (engine === 'resshift') return 'Multi-Input ResShift';
  if (engine === 'mog') return 'MoG';
  if (engine === 'tooncrafter') return 'ToonCrafter';
  return 'disabled';
}

function manifestStorageKey(engine: ManifestInterpolationEngine): string {
  return `webp-animator-model-manifest:${engine}`;
}

function syncManifestField(): void {
  const engine = interpolationEngine();
  manifestField.hidden = !requiresManifest(engine);
  if (!requiresManifest(engine)) {
    manifestInput.value = '';
    return;
  }
  manifestInput.value = localStorage.getItem(manifestStorageKey(engine)) ?? '';
}

function selectedManifestUrl(engine: RenderInterpolationEngine): string | null {
  if (!requiresManifest(engine)) return null;
  const value = manifestInput.value.trim();
  if (!value) throw new Error(`${engineLabel(engine)} requires the manifest.json produced by its browser exporter.`);
  return value;
}

function interpolationMultiplier(): 2 | 4 | 8 {
  const value = Number(selectInput('interpolation-multiplier').value);
  return value === 4 || value === 8 ? value : 2;
}

function outputDuration(): number {
  return Math.max(1, Math.round(Number(numberInput('duration').value) || 100));
}

function outputLoop(): number {
  return Math.max(0, Math.round(Number(numberInput('loop').value) || 0));
}

function outputQuality(): number {
  return Math.max(0.01, Math.min(1, (Number(numberInput('quality').value) || 95) / 100));
}

function setStatus(message: string, kind: 'neutral' | 'error' | 'success' = 'neutral'): void {
  status.textContent = message;
  status.dataset['kind'] = kind;
}

function setProgress(update: ProgressUpdate): void {
  const total = Math.max(1, update.total);
  progress.value = Math.min(1, update.current / total);
  const label = update.stage === 'decode'
    ? `Decoding${update.fileName ? ` ${update.fileName}` : ''}`
    : update.stage === 'align'
      ? 'Aligning frames'
      : update.stage === 'interpolate'
        ? 'Generating intermediate frames'
        : 'Encoding WebP';
  progressText.textContent = `${label} · ${update.current}/${update.total}`;
}

function setWorkerProgress(update: RenderWorkerProgress): void {
  if (update.stage === 'preflight') {
    const total = Math.max(1, update.total);
    progress.value = Math.min(1, update.current / total);
    progressText.textContent = `Worker hardware pre-flight · ${update.current}/${update.total}`;
  } else if (update.stage === 'decode') {
    setProgress({ stage: 'decode', current: update.current, total: update.total, fileName: update.detail });
  } else {
    setProgress({ stage: update.stage, current: update.current, total: update.total });
    if (update.stage === 'interpolate' && update.detail) progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
  }

  const backend = update.computeBackend === null ? '' : ` · ${update.computeBackend}`;
  const provider = update.provider === null ? '' : ` · model ${update.provider}`;
  const hidden = document.hidden ? ' · background tab' : '';
  setStatus(`Persistent render worker active${backend}${provider}${hidden}.`);
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
  const workerPipeline = profile.workerSupport && profile.offscreenCanvas
    ? 'full persistent render worker available'
    : 'compatibility coordinator required';

  footerResources.replaceChildren();
  const selected = document.createElement('strong');
  selected.textContent = `Selected automatically: ${result.selectedBackend}`;
  const details = document.createElement('span');
  details.textContent = `${gpuLabel(profile)} · ${webgl} · ${profile.logicalCores} logical CPU cores · ${threading} · ${ram} · ${profile.memoryTier} memory tier · ${profile.memoryBudgetMiB} MiB working budget · ${simd}`;
  const benchmarks = document.createElement('span');
  benchmarks.textContent = `Pre-flight: ${benchmarkLabel(result)}`;
  const models = document.createElement('span');
  models.textContent = `Execution: ${workerPipeline} · RIFE 4.25 built in · AMT/ResShift/MoG/ToonCrafter adapters ready for exported manifests · ProPainter licence-gated`;
  footerResources.append(selected, details, benchmarks, models);
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

interface InterpolationSequence {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: 'webgpu' | 'wasm' | null;
}

interface FinalRenderResult {
  readonly blob: Blob;
  readonly provider: 'webgpu' | 'wasm' | null;
  readonly computeBackend: ComputeBackend;
  readonly pairwise: readonly ShiftResult[];
  readonly mode: 'persistent-worker' | 'compatibility';
  readonly interpolation: RenderInterpolationEngine;
}

function passthroughSequence(
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
): InterpolationSequence {
  return { frames, durations, provider: null };
}

async function interpolateRife(
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  fallbackDuration: number,
  multiplier: 2 | 4 | 8,
): Promise<InterpolationSequence> {
  if (frames.length < 2) return passthroughSequence(frames, durations);
  setStatus('Dispatching RIFE 4.25 to the persistent inference worker…');
  const started = inferenceWorker.interpolateRife(
    frames,
    durations,
    fallbackDuration,
    multiplier,
    (workerProgress) => {
      setProgress({ stage: 'interpolate', current: workerProgress.current, total: workerProgress.total });
      if (workerProgress.provider !== null) {
        setStatus(`RIFE worker running via ${workerProgress.provider}${document.hidden ? ' while this tab is hidden' : ''}.`);
      }
    },
  );
  activeInferenceJobId = started.jobId;
  cancelButton.disabled = false;
  try {
    return await started.promise;
  } finally {
    activeInferenceJobId = null;
  }
}

function ensureCompatibilityActive(): void {
  if (compatibilityCancelRequested) throw new DOMException('Inference job cancelled.', 'AbortError');
}

async function interpolateCompatibilityAmt(
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  const result = await interpolateAmtFrames(
    manifestUrl,
    frames,
    durations,
    outputDuration(),
    interpolationMultiplier(),
    activePreflight.profile,
    (update) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'interpolate', current: update.current, total: update.total });
      progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
      const provider = update.provider ? ` via ${update.provider}` : '';
      setStatus(`AMT-S compatibility inference${provider}. Keep this tab active for best throughput.`);
    },
    ensureCompatibilityActive,
  );
  return result;
}

async function interpolateCompatibilityGenerative(
  engine: GenerativeInterpolationEngine,
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  const result = await interpolateGenerativeFrames(
    engine,
    manifestUrl,
    frames,
    durations,
    outputDuration(),
    interpolationMultiplier(),
    activePreflight.profile,
    (update) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'interpolate', current: update.current, total: update.total });
      progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
      const provider = update.provider ? ` via ${update.provider}` : '';
      setStatus(`${engineLabel(engine)} compatibility inference${provider}. Keep this tab active for best throughput.`);
    },
    ensureCompatibilityActive,
  );
  return result;
}

async function persistentRender(
  files: readonly File[],
  interpolation: RenderInterpolationEngine,
  manifestUrl: string | null,
): Promise<FinalRenderResult> {
  const started = inferenceWorker.renderFiles(
    files,
    settings(),
    interpolation,
    manifestUrl,
    interpolationMultiplier(),
    outputDuration(),
    outputLoop(),
    outputQuality(),
    setWorkerProgress,
  );
  activeInferenceJobId = started.jobId;
  cancelButton.disabled = false;
  try {
    const result = await started.promise;
    return {
      blob: result.blob,
      provider: result.provider,
      computeBackend: result.computeBackend,
      pairwise: result.pairwise,
      mode: 'persistent-worker',
      interpolation,
    };
  } finally {
    activeInferenceJobId = null;
  }
}

async function compatibilityRender(
  files: readonly File[],
  activePreflight: PreflightResult,
  interpolation: RenderInterpolationEngine,
  manifestUrl: string | null,
): Promise<FinalRenderResult> {
  compatibilityJobActive = true;
  compatibilityCancelRequested = false;
  cancelButton.disabled = false;
  const engine = await AdaptiveAlignmentEngine.create(activePreflight.profile, activePreflight.selectedBackend);
  try {
    ensureCompatibilityActive();
    const decoded = await decodeInputFiles(files, (update) => {
      ensureCompatibilityActive();
      setProgress(update);
    });
    ensureCompatibilityActive();
    const registration = await engine.registerSequence(decoded.frames, settings(), (current, total) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'align', current, total });
    });
    const rendered = renderUnionFrames(decoded.frames, registration.positions);
    const duration = outputDuration();
    let interpolated: InterpolationSequence;
    if (interpolation === 'rife') {
      interpolated = await interpolateRife(rendered, decoded.sourceDurations, duration, interpolationMultiplier());
    } else if (interpolation === 'amt') {
      if (manifestUrl === null) throw new Error('AMT-S manifest URL is missing.');
      interpolated = await interpolateCompatibilityAmt(manifestUrl, rendered, decoded.sourceDurations, activePreflight);
    } else if (isGenerative(interpolation)) {
      if (manifestUrl === null) throw new Error(`${engineLabel(interpolation)} manifest URL is missing.`);
      interpolated = await interpolateCompatibilityGenerative(interpolation, manifestUrl, rendered, decoded.sourceDurations, activePreflight);
    } else {
      interpolated = passthroughSequence(rendered, decoded.sourceDurations);
    }
    ensureCompatibilityActive();
    const blob = await encodeAnimatedWebp(interpolated.frames, {
      duration,
      durations: interpolated.durations,
      loop: outputLoop(),
      quality: outputQuality(),
    }, (current, total) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'encode', current, total });
    });
    return {
      blob,
      provider: interpolated.provider,
      computeBackend: activePreflight.selectedBackend,
      pairwise: registration.pairwise,
      mode: 'compatibility',
      interpolation,
    };
  } finally {
    compatibilityJobActive = false;
    engine.close();
  }
}

function publishResult(result: FinalRenderResult): void {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(result.blob);
  preview.src = resultUrl;
  preview.hidden = false;
  download.href = resultUrl;
  download.download = 'animation.webp';
  download.hidden = false;
  const alignmentReport = result.pairwise
    .map((entry, index) => `frame ${index + 2}: dx=${entry.dx}, dy=${entry.dy}, score=${entry.score.toFixed(6)}`)
    .join('\n');
  const modelReport = result.provider === null
    ? 'Interpolation: disabled'
    : `Interpolation: ${engineLabel(result.interpolation)} via ${result.provider}`;
  const modeReport = result.mode === 'persistent-worker'
    ? `Pipeline: persistent worker via ${result.computeBackend}`
    : `Pipeline: compatibility coordinator via ${result.computeBackend}`;
  results.textContent = `${modeReport}\n${modelReport}\n${alignmentReport}`;
  progress.value = 1;
  progressText.textContent = 'Complete';
  setStatus(`Finished entirely on this device with ${result.computeBackend}.`, 'success');
}

async function initialise(): Promise<void> {
  syncManifestField();
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

engineInput.addEventListener('change', syncManifestField);
manifestInput.addEventListener('change', () => {
  const engine = interpolationEngine();
  if (!requiresManifest(engine)) return;
  const value = manifestInput.value.trim();
  if (value) localStorage.setItem(manifestStorageKey(engine), value);
  else localStorage.removeItem(manifestStorageKey(engine));
});

cancelButton.addEventListener('click', () => {
  const jobId = activeInferenceJobId;
  if (jobId !== null) {
    cancelButton.disabled = true;
    setStatus('Cancelling after the current compute invocation returns…');
    inferenceWorker.cancel(jobId);
    return;
  }
  if (compatibilityJobActive) {
    compatibilityCancelRequested = true;
    cancelButton.disabled = true;
    setStatus('Cancelling after the current compatibility compute invocation returns…');
  }
});

document.addEventListener('visibilitychange', () => {
  if (activeInferenceJobId !== null) {
    if (document.hidden) setStatus('Tab hidden. The persistent render worker is continuing the active job.');
    else setStatus('Tab visible again. Persistent render worker remains active.');
  } else if (compatibilityJobActive && document.hidden) {
    setStatus('This browser is using the compatibility coordinator. Background throttling may pause this render.');
  }
});

runButton.addEventListener('click', () => {
  void (async (): Promise<void> => {
    const activePreflight = preflight;
    const files = fileInput.files ? Array.from(fileInput.files) : [];
    if (!activePreflight || files.length === 0) return;

    runButton.disabled = true;
    cancelButton.disabled = true;
    progress.value = 0;
    const interpolation = interpolationEngine();
    try {
      const manifestUrl = selectedManifestUrl(interpolation);
      setStatus(`Processing entirely on this device${interpolation === 'none' ? '' : ` with ${engineLabel(interpolation)}`}…`);
      const usePersistentWorker = activePreflight.profile.workerSupport && activePreflight.profile.offscreenCanvas;
      const result = usePersistentWorker
        ? await persistentRender(files, interpolation, manifestUrl)
        : await compatibilityRender(files, activePreflight, interpolation, manifestUrl);
      publishResult(result);
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        progressText.textContent = 'Cancelled';
        setStatus('Processing cancelled.', 'neutral');
      } else {
        setStatus(error instanceof Error ? error.message : String(error), 'error');
      }
    } finally {
      activeInferenceJobId = null;
      compatibilityJobActive = false;
      compatibilityCancelRequested = false;
      cancelButton.disabled = true;
      runButton.disabled = false;
    }
  })();
});

void initialise();
