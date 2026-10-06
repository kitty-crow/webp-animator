import { AdaptiveAlignmentEngine, renderUnionFrames } from './alignment.js';
import { decodeInputFiles } from './frame-codec.js';
import { detectHardware } from './hardware.js';
import { writeInferenceCheckpoint, type InferenceCheckpointModel, type InferenceCheckpointStage } from './inference-checkpoint.js';
import {
  parseInferenceWorkerRequest,
  type InferenceWorkerResponse,
  type RenderProgressStage,
  type StartRenderRequest,
  type StartRifeRequest,
  type TransferFrame,
} from './inference-worker-protocol.js';
import { interpolateAmtFrames } from './inference/amt-interpolation.js';
import { generateMidpointAnchors } from './inference/frame-generator.js';
import { interpolateGenerativeFrames, type GenerativeInterpolationEngine } from './inference/generative-interpolation.js';
import { repairProPainterFrames } from './inference/propainter.js';
import { RifeOnnxAdapter } from './inference/rife.js';
import { runPreflight } from './preflight.js';
import type { ComputeBackend, ProgressUpdate } from './types.js';
import { encodeAnimatedWebp } from './webp-muxer.js';

interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: InferenceWorkerResponse, transfer?: Transferable[]): void;
}

interface RenderSequence {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: 'webgpu' | 'wasm' | null;
}

const scope = globalThis as unknown as WorkerScopeLike;
const cancelledJobs = new Set<string>();
let activeJobId: string | null = null;
let pageHidden = false;
const visibilityWaiters = new Set<() => void>();

function wakeVisibilityWaiters(): void {
  for (const resolve of visibilityWaiters) resolve();
  visibilityWaiters.clear();
}

function setPageHidden(hidden: boolean): void {
  pageHidden = hidden;
  if (!hidden) wakeVisibilityWaiters();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function imageFromTransfer(frame: TransferFrame): ImageData {
  return new ImageData(new Uint8ClampedArray(frame.buffer), frame.width, frame.height);
}

function transferableBuffer(data: Uint8ClampedArray): ArrayBuffer {
  if (data.buffer instanceof ArrayBuffer && data.byteOffset === 0 && data.byteLength === data.buffer.byteLength) {
    return data.buffer;
  }
  const copy = new Uint8ClampedArray(data.byteLength);
  copy.set(data);
  return copy.buffer;
}

function transferFromImage(frame: ImageData): TransferFrame {
  return {
    width: frame.width,
    height: frame.height,
    buffer: transferableBuffer(frame.data),
  };
}

function ensureNotCancelled(jobId: string): void {
  if (cancelledJobs.has(jobId)) throw new DOMException('Inference job cancelled.', 'AbortError');
}

async function waitUntilVisible(jobId: string): Promise<void> {
  while (pageHidden) {
    ensureNotCancelled(jobId);
    await new Promise<void>((resolve) => visibilityWaiters.add(resolve));
  }
  ensureNotCancelled(jobId);
}

async function withHiddenTabRecovery<T>(
  jobId: string,
  operation: () => Promise<T>,
  onPause: (message: string) => void,
  onResume?: () => void | Promise<void>,
): Promise<T> {
  while (true) {
    ensureNotCancelled(jobId);
    try {
      return await operation();
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === 'AbortError') throw error;
      if (!pageHidden) throw error;
      onPause(errorMessage(error));
      await waitUntilVisible(jobId);
      if (onResume) await onResume();
    }
  }
}

function mergeProvider(
  current: 'webgpu' | 'wasm' | null,
  next: 'webgpu' | 'wasm' | null,
): 'webgpu' | 'wasm' | null {
  if (next === null) return current;
  if (current === null) return next;
  return current === 'wasm' || next === 'wasm' ? 'wasm' : 'webgpu';
}

async function checkpoint(
  jobId: string,
  model: InferenceCheckpointModel,
  status: 'running' | 'completed' | 'cancelled' | 'failed',
  stage: InferenceCheckpointStage,
  current: number,
  total: number,
  provider: 'webgpu' | 'wasm' | null,
  error: string | null,
): Promise<void> {
  await writeInferenceCheckpoint({
    jobId,
    model,
    status,
    stage,
    current,
    total,
    provider,
    updatedAt: Date.now(),
    error,
  });
}

function progress(
  jobId: string,
  stage: 'initialising' | 'interpolating',
  current: number,
  total: number,
  provider: 'webgpu' | 'wasm' | null,
): void {
  scope.postMessage({ type: 'progress', jobId, stage, current, total, provider });
}

function renderProgress(
  jobId: string,
  stage: RenderProgressStage,
  current: number,
  total: number,
  provider: 'webgpu' | 'wasm' | null,
  computeBackend: ComputeBackend | null,
  detail: string | null = null,
): void {
  scope.postMessage({
    type: 'render-progress',
    jobId,
    stage,
    current,
    total,
    provider,
    computeBackend,
    detail,
  });
  void checkpoint(jobId, 'pipeline', 'running', stage, current, total, provider, null);
}

async function runRife(request: StartRifeRequest): Promise<void> {
  if (activeJobId !== null) throw new Error(`Inference worker is already processing ${activeJobId}.`);
  activeJobId = request.jobId;
  cancelledJobs.delete(request.jobId);

  const total = Math.max(0, (request.frames.length - 1) * (request.multiplier - 1));
  let current = 0;
  let provider: 'webgpu' | 'wasm' | null = null;
  let adapter: RifeOnnxAdapter | null = null;

  try {
    await checkpoint(request.jobId, 'rife', 'running', 'initialising', current, total, provider, null);
    progress(request.jobId, 'initialising', current, total, provider);
    ensureNotCancelled(request.jobId);

    const hardware = await detectHardware();
    ensureNotCancelled(request.jobId);
    adapter = await withHiddenTabRecovery(
      request.jobId,
      () => RifeOnnxAdapter.create(hardware),
      () => progress(request.jobId, 'initialising', current, total, provider),
    );
    provider = adapter.provider;
    await checkpoint(request.jobId, 'rife', 'running', 'interpolate', current, total, provider, null);
    progress(request.jobId, 'interpolating', current, total, provider);

    const frames = request.frames.map(imageFromTransfer);
    if (frames.length === 1) {
      const only = frames[0];
      if (!only) throw new Error('RIFE worker received an empty single-frame input.');
      const transferred = transferFromImage(only);
      await checkpoint(request.jobId, 'rife', 'completed', 'interpolate', 0, 0, provider, null);
      scope.postMessage({
        type: 'complete',
        jobId: request.jobId,
        frames: [transferred],
        durations: [request.durations[0] ?? request.fallbackDuration],
        provider,
      }, [transferred.buffer]);
      return;
    }

    const outputFrames: ImageData[] = [];
    const outputDurations: (number | null)[] = [];
    for (let pairIndex = 0; pairIndex < frames.length - 1; pairIndex += 1) {
      ensureNotCancelled(request.jobId);
      const first = frames[pairIndex];
      const second = frames[pairIndex + 1];
      if (!first || !second) throw new Error(`RIFE frame pair ${pairIndex} is incomplete.`);
      const sourceDuration = request.durations[pairIndex] ?? request.fallbackDuration;
      const subDuration = Math.max(1, Math.round(sourceDuration / request.multiplier));
      outputFrames.push(first);
      outputDurations.push(subDuration);

      for (let ordinal = 1; ordinal < request.multiplier; ordinal += 1) {
        ensureNotCancelled(request.jobId);
        const ratio = ordinal / request.multiplier;
        const result = await withHiddenTabRecovery(
          request.jobId,
          () => {
            if (!adapter) throw new Error('RIFE adapter is unavailable.');
            return adapter.interpolate(first, second, ratio);
          },
          () => progress(request.jobId, 'interpolating', current, total, provider),
          async () => {
            adapter?.close();
            adapter = await RifeOnnxAdapter.create(hardware);
            provider = adapter.provider;
          },
        );
        outputFrames.push(result.image);
        outputDurations.push(subDuration);
        current += 1;
        await checkpoint(request.jobId, 'rife', 'running', 'interpolate', current, total, provider, null);
        progress(request.jobId, 'interpolating', current, total, provider);
      }
    }

    const lastFrame = frames[frames.length - 1];
    if (!lastFrame) throw new Error('RIFE final frame is missing.');
    outputFrames.push(lastFrame);
    outputDurations.push(request.durations[request.durations.length - 1] ?? request.fallbackDuration);

    const transferred = outputFrames.map(transferFromImage);
    const transferList: Transferable[] = transferred.map((frame) => frame.buffer);
    await checkpoint(request.jobId, 'rife', 'completed', 'interpolate', total, total, provider, null);
    scope.postMessage({
      type: 'complete',
      jobId: request.jobId,
      frames: transferred,
      durations: outputDurations,
      provider,
    }, transferList);
  } catch (error: unknown) {
    const cancelled = error instanceof DOMException && error.name === 'AbortError';
    if (cancelled) {
      await checkpoint(request.jobId, 'rife', 'cancelled', 'interpolate', current, total, provider, null);
      scope.postMessage({ type: 'cancelled', jobId: request.jobId });
    } else {
      const message = errorMessage(error);
      await checkpoint(request.jobId, 'rife', 'failed', 'interpolate', current, total, provider, message);
      scope.postMessage({ type: 'error', jobId: request.jobId, message });
    }
  } finally {
    adapter?.close();
    cancelledJobs.delete(request.jobId);
    activeJobId = null;
  }
}

async function interpolateRifeRenderFrames(
  request: StartRenderRequest,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  hardware: Awaited<ReturnType<typeof detectHardware>>,
  computeBackend: ComputeBackend,
): Promise<{ readonly frames: readonly ImageData[]; readonly durations: readonly (number | null)[]; readonly provider: 'webgpu' | 'wasm' }> {
  const adapter = await RifeOnnxAdapter.create(hardware);
  const provider = adapter.provider;
  const total = (frames.length - 1) * (request.multiplier - 1);
  let current = 0;
  const outputFrames: ImageData[] = [];
  const outputDurations: (number | null)[] = [];
  renderProgress(request.jobId, 'interpolate', current, total, provider, computeBackend, `RIFE 4.25 via ${provider}`);
  try {
    for (let pairIndex = 0; pairIndex < frames.length - 1; pairIndex += 1) {
      ensureNotCancelled(request.jobId);
      const first = frames[pairIndex];
      const second = frames[pairIndex + 1];
      if (!first || !second) throw new Error(`Render interpolation pair ${pairIndex} is incomplete.`);
      const sourceDuration = durations[pairIndex] ?? request.duration;
      const subDuration = Math.max(1, Math.round(sourceDuration / request.multiplier));
      outputFrames.push(first);
      outputDurations.push(subDuration);
      for (let ordinal = 1; ordinal < request.multiplier; ordinal += 1) {
        ensureNotCancelled(request.jobId);
        const result = await adapter.interpolate(first, second, ordinal / request.multiplier);
        outputFrames.push(result.image);
        outputDurations.push(subDuration);
        current += 1;
        renderProgress(request.jobId, 'interpolate', current, total, provider, computeBackend, `RIFE 4.25 via ${provider}`);
      }
    }
    const last = frames[frames.length - 1];
    if (!last) throw new Error('Render interpolation final frame is missing.');
    outputFrames.push(last);
    outputDurations.push(durations[durations.length - 1] ?? request.duration);
    return { frames: outputFrames, durations: outputDurations, provider };
  } finally {
    adapter.close();
  }
}

async function generateRenderFrames(
  request: StartRenderRequest,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  hardware: Awaited<ReturnType<typeof detectHardware>>,
  computeBackend: ComputeBackend,
): Promise<RenderSequence> {
  if (request.generator === 'none' || frames.length < 2) return { frames, durations, provider: null };
  const manifestUrl = request.generatorManifestUrl;
  if (manifestUrl === null) throw new Error(`${request.generator.toUpperCase()} requires a generator manifest URL.`);
  const total = frames.length - 1;
  renderProgress(request.jobId, 'generate', 0, total, null, computeBackend, `Loading ${request.generator.toUpperCase()} generator manifest`);
  return generateMidpointAnchors(
    request.generator,
    manifestUrl,
    frames,
    durations,
    request.duration,
    hardware,
    (update) => {
      ensureNotCancelled(request.jobId);
      renderProgress(request.jobId, 'generate', update.current, update.total, update.provider, computeBackend, update.detail);
    },
    () => ensureNotCancelled(request.jobId),
  );
}

async function interpolateRenderFrames(
  request: StartRenderRequest,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  hardware: Awaited<ReturnType<typeof detectHardware>>,
  computeBackend: ComputeBackend,
): Promise<RenderSequence> {
  if (request.interpolation === 'none' || frames.length < 2) return { frames, durations, provider: null };
  if (request.interpolation === 'rife') return interpolateRifeRenderFrames(request, frames, durations, hardware, computeBackend);
  const manifestUrl = request.modelManifestUrl;
  if (manifestUrl === null) throw new Error(`${request.interpolation} requires a model manifest URL.`);
  const total = (frames.length - 1) * (request.multiplier - 1);
  if (request.interpolation === 'amt') {
    renderProgress(request.jobId, 'interpolate', 0, total, null, computeBackend, 'Loading AMT model manifest');
    return interpolateAmtFrames(
      manifestUrl,
      frames,
      durations,
      request.duration,
      request.multiplier,
      hardware,
      (update) => {
        ensureNotCancelled(request.jobId);
        renderProgress(request.jobId, 'interpolate', update.current, update.total, update.provider, computeBackend, update.detail);
      },
      () => ensureNotCancelled(request.jobId),
    );
  }
  const engine: GenerativeInterpolationEngine = request.interpolation;
  renderProgress(request.jobId, 'interpolate', 0, total, null, computeBackend, `Loading ${engine} model manifest`);
  return interpolateGenerativeFrames(
    engine,
    manifestUrl,
    frames,
    durations,
    request.duration,
    request.multiplier,
    hardware,
    (update) => {
      ensureNotCancelled(request.jobId);
      renderProgress(request.jobId, 'interpolate', update.current, update.total, update.provider, computeBackend, update.detail);
    },
    () => ensureNotCancelled(request.jobId),
  );
}

async function repairRenderFrames(
  request: StartRenderRequest,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  hardware: Awaited<ReturnType<typeof detectHardware>>,
  computeBackend: ComputeBackend,
): Promise<RenderSequence> {
  if (request.repair === 'none' || frames.length < 3) return { frames, durations, provider: null };
  const manifestUrl = request.repairManifestUrl;
  if (manifestUrl === null) throw new Error('ProPainter requires a repair manifest URL.');
  renderProgress(request.jobId, 'repair', 0, 1, null, computeBackend, 'Auditing temporal alpha holes');
  const result = await repairProPainterFrames(
    manifestUrl,
    frames,
    hardware,
    (update) => {
      ensureNotCancelled(request.jobId);
      renderProgress(request.jobId, 'repair', update.current, update.total, update.provider, computeBackend, update.detail);
    },
    () => ensureNotCancelled(request.jobId),
  );
  if (result.windows === 0) {
    renderProgress(request.jobId, 'repair', 1, 1, null, computeBackend, 'ProPainter · no repairable temporal alpha holes');
  } else {
    renderProgress(
      request.jobId,
      'repair',
      result.windows,
      result.windows,
      result.provider,
      computeBackend,
      `ProPainter · repaired ${result.repairedPixels} pixels`,
    );
  }
  return { frames: result.frames, durations, provider: result.provider };
}

async function runRender(request: StartRenderRequest): Promise<void> {
  if (activeJobId !== null) throw new Error(`Inference worker is already processing ${activeJobId}.`);
  activeJobId = request.jobId;
  cancelledJobs.delete(request.jobId);
  let stage: InferenceCheckpointStage = 'preflight';
  let current = 0;
  let total = 1;
  let provider: 'webgpu' | 'wasm' | null = null;
  let computeBackend: ComputeBackend | null = null;

  try {
    if (typeof OffscreenCanvas !== 'function') throw new Error('Persistent background rendering requires OffscreenCanvas in this browser.');
    renderProgress(request.jobId, 'preflight', 0, 1, provider, computeBackend, 'Benchmarking worker-local compute backends');
    const preflight = await withHiddenTabRecovery(
      request.jobId,
      () => runPreflight(),
      (message) => renderProgress(
        request.jobId,
        'preflight',
        0,
        1,
        provider,
        computeBackend,
        `Browser paused background compute while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'preflight', 0, 1, provider, computeBackend, 'Tab visible again; retrying pre-flight'),
    );
    const selectedBackend = preflight.selectedBackend;
    computeBackend = selectedBackend;
    ensureNotCancelled(request.jobId);
    renderProgress(request.jobId, 'preflight', 1, 1, provider, selectedBackend, `Selected ${selectedBackend}`);

    stage = 'decode';
    const decoded = await withHiddenTabRecovery(
      request.jobId,
      () => decodeInputFiles(request.files, (update: ProgressUpdate) => {
        if (update.stage !== 'decode') return;
        ensureNotCancelled(request.jobId);
        current = update.current;
        total = update.total;
        renderProgress(request.jobId, 'decode', update.current, update.total, provider, computeBackend, update.fileName);
      }),
      (message) => renderProgress(
        request.jobId,
        'decode',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background decoding while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'decode', current, total, provider, computeBackend, 'Tab visible again; retrying decode'),
    );
    ensureNotCancelled(request.jobId);

    stage = 'align';
    const registration = await withHiddenTabRecovery(
      request.jobId,
      async () => {
        const alignmentEngine = await AdaptiveAlignmentEngine.create(preflight.profile, selectedBackend);
        try {
          return await alignmentEngine.registerSequence(decoded.frames, request.registration, (completed, count) => {
            ensureNotCancelled(request.jobId);
            current = completed;
            total = count;
            renderProgress(request.jobId, 'align', completed, count, provider, selectedBackend);
          });
        } finally {
          alignmentEngine.close();
        }
      },
      (message) => renderProgress(
        request.jobId,
        'align',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background alignment while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'align', current, total, provider, computeBackend, 'Tab visible again; retrying alignment'),
    );
    const rendered = renderUnionFrames(decoded.frames, registration.positions);
    ensureNotCancelled(request.jobId);

    stage = 'generate';
    const generated = await withHiddenTabRecovery(
      request.jobId,
      () => generateRenderFrames(
        request,
        rendered,
        decoded.sourceDurations,
        preflight.profile,
        selectedBackend,
      ),
      (message) => renderProgress(
        request.jobId,
        'generate',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background generation while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'generate', current, total, provider, computeBackend, 'Tab visible again; retrying generation'),
    );
    provider = mergeProvider(provider, generated.provider);
    ensureNotCancelled(request.jobId);

    stage = 'interpolate';
    const interpolated = await withHiddenTabRecovery(
      request.jobId,
      () => interpolateRenderFrames(
        request,
        generated.frames,
        generated.durations,
        preflight.profile,
        selectedBackend,
      ),
      (message) => renderProgress(
        request.jobId,
        'interpolate',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background interpolation while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'interpolate', current, total, provider, computeBackend, 'Tab visible again; retrying interpolation'),
    );
    provider = mergeProvider(provider, interpolated.provider);
    ensureNotCancelled(request.jobId);

    stage = 'repair';
    const repaired = await withHiddenTabRecovery(
      request.jobId,
      () => repairRenderFrames(
        request,
        interpolated.frames,
        interpolated.durations,
        preflight.profile,
        selectedBackend,
      ),
      (message) => renderProgress(
        request.jobId,
        'repair',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background repair while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'repair', current, total, provider, computeBackend, 'Tab visible again; retrying repair'),
    );
    provider = mergeProvider(provider, repaired.provider);
    ensureNotCancelled(request.jobId);

    stage = 'encode';
    const blob = await withHiddenTabRecovery(
      request.jobId,
      () => encodeAnimatedWebp(repaired.frames, {
        duration: request.duration,
        durations: repaired.durations,
        loop: request.loop,
        quality: request.quality,
      }, (completed, count) => {
        ensureNotCancelled(request.jobId);
        current = completed;
        total = count;
        renderProgress(request.jobId, 'encode', completed, count, provider, computeBackend);
      }),
      (message) => renderProgress(
        request.jobId,
        'encode',
        current,
        total,
        provider,
        computeBackend,
        `Browser paused background encoding while hidden (${message}). Waiting for visibility…`,
      ),
      () => renderProgress(request.jobId, 'encode', current, total, provider, computeBackend, 'Tab visible again; retrying encode'),
    );
    const webp = await blob.arrayBuffer();
    await checkpoint(request.jobId, 'pipeline', 'completed', 'encode', total, total, provider, null);
    scope.postMessage({
      type: 'render-complete',
      jobId: request.jobId,
      webp,
      provider,
      computeBackend,
      pairwise: registration.pairwise,
    }, [webp]);
  } catch (error: unknown) {
    const cancelled = error instanceof DOMException && error.name === 'AbortError';
    if (cancelled) {
      await checkpoint(request.jobId, 'pipeline', 'cancelled', stage, current, total, provider, null);
      scope.postMessage({ type: 'cancelled', jobId: request.jobId });
    } else {
      const message = errorMessage(error);
      await checkpoint(request.jobId, 'pipeline', 'failed', stage, current, total, provider, message);
      scope.postMessage({ type: 'error', jobId: request.jobId, message });
    }
  } finally {
    cancelledJobs.delete(request.jobId);
    activeJobId = null;
  }
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  void (async (): Promise<void> => {
    let jobId = 'unknown';
    try {
      const request = parseInferenceWorkerRequest(event.data);
      if (request.type === 'visibility') {
        setPageHidden(request.hidden);
        return;
      }
      jobId = request.jobId;
      if (request.type === 'cancel') {
        cancelledJobs.add(request.jobId);
        wakeVisibilityWaiters();
        if (activeJobId !== request.jobId) scope.postMessage({ type: 'cancelled', jobId: request.jobId });
        return;
      }
      if (request.type === 'start-render') {
        await runRender(request);
        return;
      }
      await runRife(request);
    } catch (error: unknown) {
      scope.postMessage({ type: 'error', jobId, message: errorMessage(error) });
    }
  })();
});
