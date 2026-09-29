import { detectHardware } from './hardware.js';
import { writeInferenceCheckpoint } from './inference-checkpoint.js';
import {
  parseInferenceWorkerRequest,
  type InferenceWorkerResponse,
  type StartRifeRequest,
  type TransferFrame,
} from './inference-worker-protocol.js';
import { RifeOnnxAdapter } from './inference/rife.js';

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
  status: 'running' | 'completed' | 'cancelled' | 'failed',
  current: number,
  total: number,
  provider: 'webgpu' | 'wasm' | null,
  error: string | null,
): Promise<void> {
  await writeInferenceCheckpoint({
    jobId,
    model: 'rife',
    status,
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

async function runRife(request: StartRifeRequest): Promise<void> {
  if (activeJobId !== null) throw new Error(`Inference worker is already processing ${activeJobId}.`);
  activeJobId = request.jobId;
  cancelledJobs.delete(request.jobId);

  const total = Math.max(0, (request.frames.length - 1) * (request.multiplier - 1));
  let current = 0;
  let provider: 'webgpu' | 'wasm' | null = null;
  let adapter: RifeOnnxAdapter | null = null;

  try {
    await checkpoint(request.jobId, 'running', current, total, provider, null);
    progress(request.jobId, 'initialising', current, total, provider);
    ensureNotCancelled(request.jobId);

    const hardware = await detectHardware();
    ensureNotCancelled(request.jobId);
    adapter = await RifeOnnxAdapter.create(hardware);
    provider = adapter.provider;
    await checkpoint(request.jobId, 'running', current, total, provider, null);
    progress(request.jobId, 'interpolating', current, total, provider);

    const frames = request.frames.map(imageFromTransfer);
    if (frames.length === 1) {
      const only = frames[0];
      if (!only) throw new Error('RIFE worker received an empty single-frame input.');
      const transferred = transferFromImage(only);
      await checkpoint(request.jobId, 'completed', 0, 0, provider, null);
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
        await checkpoint(request.jobId, 'running', current, total, provider, null);
        progress(request.jobId, 'interpolating', current, total, provider);
      }
    }

    const lastFrame = frames[frames.length - 1];
    if (!lastFrame) throw new Error('RIFE final frame is missing.');
    outputFrames.push(lastFrame);
    outputDurations.push(request.durations[request.durations.length - 1] ?? request.fallbackDuration);

    const transferred = outputFrames.map(transferFromImage);
    const transferList: Transferable[] = transferred.map((frame) => frame.buffer);
    await checkpoint(request.jobId, 'completed', total, total, provider, null);
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
      await checkpoint(request.jobId, 'cancelled', current, total, provider, null);
      scope.postMessage({ type: 'cancelled', jobId: request.jobId });
    } else {
      const message = errorMessage(error);
      await checkpoint(request.jobId, 'failed', current, total, provider, message);
      scope.postMessage({ type: 'error', jobId: request.jobId, message });
    }
  } finally {
    adapter?.close();
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
      await runRife(request);
    } catch (error: unknown) {
      scope.postMessage({ type: 'error', jobId, message: errorMessage(error) });
    }
  })();
});
