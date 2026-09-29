import { AdaptiveAlignmentEngine, renderUnionFrames } from './alignment.js';
import { decodeInputFiles } from './frame-codec.js';
import { gpuLabel } from './hardware.js';
import { InferenceWorkerClient, type RenderWorkerProgress } from './inference-worker-client.js';
import type { FrameGeneratorEngine, RenderInterpolationEngine, TemporalRepairEngine } from './inference-worker-protocol.js';
import { interpolateAmtFrames } from './inference/amt-interpolation.js';
import { generateMidpointAnchors } from './inference/frame-generator.js';
import { interpolateGenerativeFrames, type GenerativeInterpolationEngine } from './inference/generative-interpolation.js';
import { repairProPainterFrames } from './inference/propainter.js';
import { runPreflight } from './preflight.js';
import type { ComputeBackend, PreflightResult, ProgressUpdate, RegistrationSettings, ShiftResult } from './types.js';
import { encodeAnimatedWebp } from './webp-muxer.js';

type ManifestInterpolationEngine = Exclude<RenderInterpolationEngine, 'none' | 'rife'>;
type ManifestGeneratorEngine = Exclude<FrameGeneratorEngine, 'none'>;
type ManifestRepairEngine = Exclude<TemporalRepairEngine, 'none'>;

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
const generatorManifestField = element('generator-manifest-field', HTMLLabelElement);
const generatorManifestInput = element('generator-manifest-url', HTMLInputElement);
const generatorInput = selectInput('frame-generator');
const repairManifestField = element('repair-manifest-field', HTMLLabelElement);
const repairManifestInput = element('repair-manifest-url', HTMLInputElement);
const repairInput = selectInput('repair-engine');
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

function frameGenerator(): FrameGeneratorEngine {
  const value = generatorInput.value;
  return value === 'eden' || value === 'speed' ? value : 'none';
}

function temporalRepair(): TemporalRepairEngine {
  return repairInput.value === 'propainter' ? 'propainter' : 'none';
}

function isGenerative(engine: RenderInterpolationEngine): engine is GenerativeInterpolationEngine {
  return engine === 'resshift' || engine === 'mog' || engine === 'tooncrafter';
}

function requiresManifest(engine: RenderInterpolationEngine): engine is ManifestInterpolationEngine {
  return engine === 'amt' || isGenerative(engine);
}

function requiresGeneratorManifest(generator: FrameGeneratorEngine): generator is ManifestGeneratorEngine {
  return generator === 'eden' || generator === 'speed';
}

function requiresRepairManifest(repair: TemporalRepairEngine): repair is ManifestRepairEngine {
  return repair === 'propainter';
}

function engineLabel(engine: RenderInterpolationEngine): string {
  if (engine === 'rife') return 'RIFE 4.25';
  if (engine === 'amt') return 'AMT-S';
  if (engine === 'resshift') return 'Multi-Input ResShift';
  if (engine === 'mog') return 'MoG';
  if (engine === 'tooncrafter') return 'ToonCrafter';
  return 'disabled';
}

function generatorLabel(generator: FrameGeneratorEngine): string {
  if (generator === 'eden') return 'EDEN';
  if (generator === 'speed') return 'SPEED';
  return 'disabled';
}

function repairLabel(repair: TemporalRepairEngine): string {
  return repair === 'propainter' ? 'ProPainter selective repair' : 'disabled';
}

function manifestStorageKey(engine: ManifestInterpolationEngine): string {
  return `webp-animator-model-manifest:${engine}`;
}

function generatorManifestStorageKey(generator: ManifestGeneratorEngine): string {
  return `webp-animator-generator-manifest:${generator}`;
}

function repairManifestStorageKey(repair: ManifestRepairEngine): string {
  return `webp-animator-repair-manifest:${repair}`;
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

function syncGeneratorManifestField(): void {
  const generator = frameGenerator();
  generatorManifestField.hidden = !requiresGeneratorManifest(generator);
  if (!requiresGeneratorManifest(generator)) {
    generatorManifestInput.value = '';
    return;
  }
  generatorManifestInput.value = localStorage.getItem(generatorManifestStorageKey(generator)) ?? '';
}

function syncRepairManifestField(): void {
  const repair = temporalRepair();
  repairManifestField.hidden = !requiresRepairManifest(repair);
  if (!requiresRepairManifest(repair)) {
    repairManifestInput.value = '';
    return;
  }
  repairManifestInput.value = localStorage.getItem(repairManifestStorageKey(repair)) ?? '';
}

function selectedManifestUrl(engine: RenderInterpolationEngine): string | null {
  if (!requiresManifest(engine)) return null;
  const value = manifestInput.value.trim();
  if (!value) throw new Error(`${engineLabel(engine)} requires the manifest.json produced by its browser exporter.`);
  return value;
}

function selectedGeneratorManifestUrl(generator: FrameGeneratorEngine): string | null {
  if (!requiresGeneratorManifest(generator)) return null;
  const value = generatorManifestInput.value.trim();
  if (!value) throw new Error(`${generatorLabel(generator)} requires the manifest.json produced by its browser exporter.`);
  return value;
}

function selectedRepairManifestUrl(repair: TemporalRepairEngine): string | null {
  if (!requiresRepairManifest(repair)) return null;
  const value = repairManifestInput.value.trim();
  if (!value) throw new Error(`${repairLabel(repair)} requires the manifest.json produced from licensed ProPainter weights.`);
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

function mergeProvider(
  current: 'webgpu' | 'wasm' | null,
  next: 'webgpu' | 'wasm' | null,
): 'webgpu' | 'wasm' | null {
  if (next === null) return current;
  if (current === null) return next;
  return current === 'wasm' || next === 'wasm' ? 'wasm' : 'webgpu';
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
      : update.stage === 'generate'
        ? 'Generating midpoint anchors'
        : update.stage === 'interpolate'
          ? 'Generating intermediate frames'
          : update.stage === 'repair'
            ? 'Repairing temporal alpha holes'
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
    if ((update.stage === 'generate' || update.stage === 'interpolate' || update.stage === 'repair') && update.detail) {
      progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
    }
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
  models.textContent = `Execution: ${workerPipeline} · RIFE 4.25 built in · EDEN/SPEED/AMT/ResShift/MoG/ToonCrafter manifest-ready · ProPainter manifest-ready with user-supplied licensed weights`;
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
  readonly generator: FrameGeneratorEngine;
  readonly interpolation: RenderInterpolationEngine;
  readonly repair: TemporalRepairEngine;
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

async function generateCompatibility(
  generator: ManifestGeneratorEngine,
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  return generateMidpointAnchors(
    generator,
    manifestUrl,
    frames,
    durations,
    outputDuration(),
    activePreflight.profile,
    (update) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'generate', current: update.current, total: update.total });
      progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
      const provider = update.provider ? ` via ${update.provider}` : '';
      setStatus(`${generatorLabel(generator)} compatibility generation${provider}. Keep this tab active for best throughput.`);
    },
    ensureCompatibilityActive,
  );
}

async function interpolateCompatibilityAmt(
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  return interpolateAmtFrames(
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
}

async function interpolateCompatibilityGenerative(
  engine: GenerativeInterpolationEngine,
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  return interpolateGenerativeFrames(
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
}

async function repairCompatibility(
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  activePreflight: PreflightResult,
): Promise<InterpolationSequence> {
  const repaired = await repairProPainterFrames(
    manifestUrl,
    frames,
    activePreflight.profile,
    (update) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'repair', current: update.current, total: update.total });
      progressText.textContent = `${update.detail} · ${update.current}/${update.total}`;
      const provider = update.provider ? ` via ${update.provider}` : '';
      setStatus(`ProPainter compatibility repair${provider}. Keep this tab active for best throughput.`);
    },
    ensureCompatibilityActive,
  );
  return { frames: repaired.frames, durations, provider: repaired.provider };
}

async function persistentRender(
  files: readonly File[],
  generator: FrameGeneratorEngine,
  generatorManifestUrl: string | null,
  interpolation: RenderInterpolationEngine,
  manifestUrl: string | null,
  repair: TemporalRepairEngine,
  repairManifestUrl: string | null,
): Promise<FinalRenderResult> {
  const started = inferenceWorker.renderFiles(
    files,
    settings(),
    generator,
    generatorManifestUrl,
    interpolation,
    manifestUrl,
    repair,
    repairManifestUrl,
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
      generator,
      interpolation,
      repair,
    };
  } finally {
    activeInferenceJobId = null;
  }
}

async function compatibilityRender(
  files: readonly File[],
  activePreflight: PreflightResult,
  generator: FrameGeneratorEngine,
  generatorManifestUrl: string | null,
  interpolation: RenderInterpolationEngine,
  manifestUrl: string | null,
  repair: TemporalRepairEngine,
  repairManifestUrl: string | null,
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

    let generated = passthroughSequence(rendered, decoded.sourceDurations);
    if (requiresGeneratorManifest(generator)) {
      if (generatorManifestUrl === null) throw new Error(`${generatorLabel(generator)} manifest URL is missing.`);
      generated = await generateCompatibility(generator, generatorManifestUrl, rendered, decoded.sourceDurations, activePreflight);
    }

    let interpolated: InterpolationSequence;
    if (interpolation === 'rife') {
      interpolated = await interpolateRife(generated.frames, generated.durations, duration, interpolationMultiplier());
    } else if (interpolation === 'amt') {
      if (manifestUrl === null) throw new Error('AMT-S manifest URL is missing.');
      interpolated = await interpolateCompatibilityAmt(manifestUrl, generated.frames, generated.durations, activePreflight);
    } else if (isGenerative(interpolation)) {
      if (manifestUrl === null) throw new Error(`${engineLabel(interpolation)} manifest URL is missing.`);
      interpolated = await interpolateCompatibilityGenerative(interpolation, manifestUrl, generated.frames, generated.durations, activePreflight);
    } else {
      interpolated = passthroughSequence(generated.frames, generated.durations);
    }

    let repaired = interpolated;
    if (requiresRepairManifest(repair)) {
      if (repairManifestUrl === null) throw new Error('ProPainter manifest URL is missing.');
      repaired = await repairCompatibility(repairManifestUrl, interpolated.frames, interpolated.durations, activePreflight);
    }

    ensureCompatibilityActive();
    const blob = await encodeAnimatedWebp(repaired.frames, {
      duration,
      durations: repaired.durations,
      loop: outputLoop(),
      quality: outputQuality(),
    }, (current, total) => {
      ensureCompatibilityActive();
      setProgress({ stage: 'encode', current, total });
    });
    return {
      blob,
      provider: mergeProvider(mergeProvider(generated.provider, interpolated.provider), repaired.provider),
      computeBackend: activePreflight.selectedBackend,
      pairwise: registration.pairwise,
      mode: 'compatibility',
      generator,
      interpolation,
      repair,
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
  const generatorReport = result.generator === 'none'
    ? 'Generator: disabled'
    : `Generator: ${generatorLabel(result.generator)}`;
  const modelReport = result.interpolation === 'none'
    ? 'Interpolation: disabled'
    : `Interpolation: ${engineLabel(result.interpolation)}`;
  const repairReport = result.repair === 'none'
    ? 'Temporal repair: disabled'
    : `Temporal repair: ${repairLabel(result.repair)}`;
  const providerReport = result.provider === null ? 'Model provider: none' : `Model provider: ${result.provider}`;
  const modeReport = result.mode === 'persistent-worker'
    ? `Pipeline: persistent worker via ${result.computeBackend}`
    : `Pipeline: compatibility coordinator via ${result.computeBackend}`;
  results.textContent = `${modeReport}\n${generatorReport}\n${modelReport}\n${repairReport}\n${providerReport}\n${alignmentReport}`;
  progress.value = 1;
  progressText.textContent = 'Complete';
  setStatus(`Finished entirely on this device with ${result.computeBackend}.`, 'success');
}

async function initialise(): Promise<void> {
  syncManifestField();
  syncGeneratorManifestField();
  syncRepairManifestField();
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
generatorInput.addEventListener('change', syncGeneratorManifestField);
repairInput.addEventListener('change', syncRepairManifestField);
manifestInput.addEventListener('change', () => {
  const engine = interpolationEngine();
  if (!requiresManifest(engine)) return;
  const value = manifestInput.value.trim();
  if (value) localStorage.setItem(manifestStorageKey(engine), value);
  else localStorage.removeItem(manifestStorageKey(engine));
});
generatorManifestInput.addEventListener('change', () => {
  const generator = frameGenerator();
  if (!requiresGeneratorManifest(generator)) return;
  const value = generatorManifestInput.value.trim();
  if (value) localStorage.setItem(generatorManifestStorageKey(generator), value);
  else localStorage.removeItem(generatorManifestStorageKey(generator));
});
repairManifestInput.addEventListener('change', () => {
  const repair = temporalRepair();
  if (!requiresRepairManifest(repair)) return;
  const value = repairManifestInput.value.trim();
  if (value) localStorage.setItem(repairManifestStorageKey(repair), value);
  else localStorage.removeItem(repairManifestStorageKey(repair));
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
    const generator = frameGenerator();
    const interpolation = interpolationEngine();
    const repair = temporalRepair();
    try {
      const generatorManifestUrl = selectedGeneratorManifestUrl(generator);
      const manifestUrl = selectedManifestUrl(interpolation);
      const repairManifestUrl = selectedRepairManifestUrl(repair);
      const generatorText = generator === 'none' ? '' : ` with ${generatorLabel(generator)}`;
      const interpolationText = interpolation === 'none' ? '' : `, ${engineLabel(interpolation)}`;
      const repairText = repair === 'none' ? '' : `, and ${repairLabel(repair)}`;
      setStatus(`Processing entirely on this device${generatorText}${interpolationText}${repairText}…`);
      const usePersistentWorker = activePreflight.profile.workerSupport && activePreflight.profile.offscreenCanvas;
      const result = usePersistentWorker
        ? await persistentRender(files, generator, generatorManifestUrl, interpolation, manifestUrl, repair, repairManifestUrl)
        : await compatibilityRender(files, activePreflight, generator, generatorManifestUrl, interpolation, manifestUrl, repair, repairManifestUrl);
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
