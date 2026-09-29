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
import { RifeOnnxAdapter } from './inference/rife.js';
import { runPreflight } from './preflight.js';
import type { ComputeBackend, ProgressUpdate } from './types.js';
import { encodeAnimatedWebp } from './webp-muxer.js';

interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: InferenceWorkerResponse, transfer?: Transferable[]): void;
}

const scope = globalThis as unknown as WorkerScopeLike;
const cancelledJobs = new Set<string>();
let activeJobId: string | null = null;

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
    adapter = await RifeOnnxAdapter.create(hardware);
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
        const result = await adapter.interpolate(first, second, ratio);
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

async function interpolateRenderFrames(
  request: StartRenderRequest,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  hardware: Awaited<ReturnType<typeof detectHardware>>,
  computeBackend: ComputeBackend,
): Promise<{ readonly frames: readonly ImageData[]; readonly durations: readonly (number | null)[]; readonly provider: 'webgpu' | 'wasm' | null }> {
  if (request.interpolation === 'none' || frames.length < 2) return { frames, durations, provider: null };
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

async function runRender(request: StartRenderRequest): Promise<void> {
  if (activeJobId !== null) throw new Error(`Inference worker is already processing ${activeJobId}.`);
  activeJobId = request.jobId;
  cancelledJobs.delete(request.jobId);
  let stage: InferenceCheckpointStage = 'preflight';
  let current = 0;
  let total = 1;
  let provider: 'webgpu' | 'wasm' | null = null;
  let computeBackend: ComputeBackend | null = null;
  let engine: AdaptiveAlignmentEngine | null = null;

  try {
    if (typeof OffscreenCanvas !== 'function') throw new Error('Persistent background rendering requires OffscreenCanvas in this browser.');
    renderProgress(request.jobId, 'preflight', 0, 1, provider, computeBackend, 'Benchmarking worker-local compute backends');
    const preflight = await runPreflight();
    computeBackend = preflight.selectedBackend;
    ensureNotCancelled(request.jobId);
    renderProgress(request.jobId, 'preflight', 1, 1, provider, computeBackend, `Selected ${computeBackend}`);

    stage = 'decode';
    const decoded = await decodeInputFiles(request.files, (update: ProgressUpdate) => {
      if (update.stage !== 'decode') return;
      ensureNotCancelled(request.jobId);
      current = update.current;
      total = update.total;
      renderProgress(request.jobId, 'decode', update.current, update.total, provider, computeBackend, update.fileName);
    });
    ensureNotCancelled(request.jobId);

    stage = 'align';
    engine = await AdaptiveAlignmentEngine.create(preflight.profile, computeBackend);
    const registration = await engine.registerSequence(decoded.frames, request.registration, (completed, count) => {
      ensureNotCancelled(request.jobId);
      current = completed;
      total = count;
      renderProgress(request.jobId, 'align', completed, count, provider, computeBackend);
    });
    const rendered = renderUnionFrames(decoded.frames, registration.positions);
    ensureNotCancelled(request.jobId);

    stage = 'interpolate';
    const interpolated = await interpolateRenderFrames(
      request,
      rendered,
      decoded.sourceDurations,
      preflight.profile,
      computeBackend,
    );
    provider = interpolated.provider;
    ensureNotCancelled(request.jobId);

    stage = 'encode';
    const blob = await encodeAnimatedWebp(interpolated.frames, {
      duration: request.duration,
      durations: interpolated.durations,
      loop: request.loop,
      quality: request.quality,
    }, (completed, count) => {
      ensureNotCancelled(request.jobId);
      current = completed;
      total = count;
      renderProgress(request.jobId, 'encode', completed, count, provider, computeBackend);
    });
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
    engine?.close();
    cancelledJobs.delete(request.jobId);
    activeJobId = null;
  }
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  void (async (): Promise<void> => {
    let jobId = 'unknown';
    try {
      const request = parseInferenceWorkerRequest(event.data);
      jobId = request.jobId;
      if (request.type === 'cancel') {
        cancelledJobs.add(request.jobId);
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
