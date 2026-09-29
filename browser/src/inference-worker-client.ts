import type { ModelExecutionProvider } from './inference/ort-runtime.js';
import type {
  InferenceWorkerRequest,
  RenderInterpolationEngine,
  RenderProgressStage,
  RifeMultiplier,
  TransferFrame,
} from './inference-worker-protocol.js';
import type { ComputeBackend, RegistrationSettings, ShiftResult } from './types.js';

export interface RifeWorkerResult {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

export interface InferenceWorkerProgress {
  readonly stage: 'initialising' | 'interpolating';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface RenderWorkerProgress {
  readonly stage: RenderProgressStage;
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly computeBackend: ComputeBackend | null;
  readonly detail: string | null;
}

export interface RenderWorkerResult {
  readonly blob: Blob;
  readonly provider: ModelExecutionProvider | null;
  readonly computeBackend: ComputeBackend;
  readonly pairwise: readonly ShiftResult[];
}

interface PendingRifeJob {
  readonly kind: 'rife';
  readonly resolve: (result: RifeWorkerResult) => void;
  readonly reject: (error: Error) => void;
  readonly onProgress: (progress: InferenceWorkerProgress) => void;
}

interface PendingRenderJob {
  readonly kind: 'render';
  readonly resolve: (result: RenderWorkerResult) => void;
  readonly reject: (error: Error) => void;
  readonly onProgress: (progress: RenderWorkerProgress) => void;
}

type PendingJob = PendingRifeJob | PendingRenderJob;

function transferableBuffer(data: Uint8ClampedArray): ArrayBuffer {
  const copy = new Uint8ClampedArray(data.byteLength);
  copy.set(data);
  return copy.buffer;
}

function transferFrame(frame: ImageData): TransferFrame {
  return {
    width: frame.width,
    height: frame.height,
    buffer: transferableBuffer(frame.data),
  };
}

function imageFrame(frame: TransferFrame): ImageData {
  return new ImageData(new Uint8ClampedArray(frame.buffer), frame.width, frame.height);
}

function parseTransferFrame(value: unknown): TransferFrame {
  if (typeof value !== 'object' || value === null) throw new Error('Inference worker returned an invalid frame.');
  const record = value as Record<string, unknown>;
  const width = record['width'];
  const height = record['height'];
  const buffer = record['buffer'];
  if (typeof width !== 'number' || !Number.isInteger(width) || width <= 0) throw new Error('Inference worker frame width is invalid.');
  if (typeof height !== 'number' || !Number.isInteger(height) || height <= 0) throw new Error('Inference worker frame height is invalid.');
  if (!(buffer instanceof ArrayBuffer)) throw new Error('Inference worker frame buffer is invalid.');
  if (buffer.byteLength !== width * height * 4) throw new Error('Inference worker frame byte length is invalid.');
  return { width, height, buffer };
}

function parseDurations(value: unknown): readonly (number | null)[] {
  if (!Array.isArray(value)) throw new Error('Inference worker durations are invalid.');
  return value.map((entry: unknown): number | null => {
    if (entry === null) return null;
    if (typeof entry !== 'number' || !Number.isFinite(entry) || entry <= 0) throw new Error('Inference worker duration is invalid.');
    return entry;
  });
}

function parseProvider(value: unknown): ModelExecutionProvider | null {
  if (value === null) return null;
  if (value === 'webgpu' || value === 'wasm') return value;
  throw new Error('Inference worker provider is invalid.');
}

function parseComputeBackend(value: unknown): ComputeBackend {
  if (value === 'webgpu' || value === 'wasm-simd-workers' || value === 'wasm-workers' || value === 'js-workers' || value === 'main-thread-js') return value;
  throw new Error('Inference worker compute backend is invalid.');
}

function parseShiftResult(value: unknown): ShiftResult {
  if (typeof value !== 'object' || value === null) throw new Error('Inference worker shift result is invalid.');
  const record = value as Record<string, unknown>;
  const dx = record['dx'];
  const dy = record['dy'];
  const score = record['score'];
  if (typeof dx !== 'number' || typeof dy !== 'number' || typeof score !== 'number') throw new Error('Inference worker shift result fields are invalid.');
  return { dx, dy, score };
}

function parseRenderStage(value: unknown): RenderProgressStage {
  if (value === 'preflight' || value === 'decode' || value === 'align' || value === 'interpolate' || value === 'encode') return value;
  throw new Error('Inference worker render stage is invalid.');
}

function jobId(prefix: string, counter: number): string {
  return `${prefix}-${Date.now().toString(36)}-${counter.toString(36)}`;
}

export class InferenceWorkerClient {
  private readonly worker: Worker;
  private readonly pending = new Map<string, PendingJob>();
  private counter = 0;

  constructor() {
    this.worker = new Worker(new URL('./inference-worker.js', import.meta.url), { type: 'module', name: 'webp-animator-inference' });
    this.worker.addEventListener('message', (event: MessageEvent<unknown>) => this.handleMessage(event.data));
    this.worker.addEventListener('error', (event: ErrorEvent) => {
      const message = event.message || 'Inference worker failed.';
      for (const pending of this.pending.values()) pending.reject(new Error(message));
      this.pending.clear();
    });
  }

  private handleMessage(value: unknown): void {
    if (typeof value !== 'object' || value === null) return;
    const message = value as Record<string, unknown>;
    const type = message['type'];
    const id = message['jobId'];
    if (typeof type !== 'string' || typeof id !== 'string') return;
    const pending = this.pending.get(id);
    if (!pending) return;

    if (type === 'progress') {
      if (pending.kind !== 'rife') return;
      const stage = message['stage'];
      const current = message['current'];
      const total = message['total'];
      const provider = message['provider'];
      if (
        (stage === 'initialising' || stage === 'interpolating') &&
        typeof current === 'number' && Number.isFinite(current) &&
        typeof total === 'number' && Number.isFinite(total) &&
        (provider === 'webgpu' || provider === 'wasm' || provider === null)
      ) {
        pending.onProgress({ stage, current, total, provider });
      }
      return;
    }

    if (type === 'render-progress') {
      if (pending.kind !== 'render') return;
      try {
        const current = message['current'];
        const total = message['total'];
        const detail = message['detail'];
        const rawBackend = message['computeBackend'];
        if (typeof current !== 'number' || typeof total !== 'number') throw new Error('Inference worker render progress counters are invalid.');
        if (detail !== null && typeof detail !== 'string') throw new Error('Inference worker render progress detail is invalid.');
        pending.onProgress({
          stage: parseRenderStage(message['stage']),
          current,
          total,
          provider: parseProvider(message['provider']),
          computeBackend: rawBackend === null ? null : parseComputeBackend(rawBackend),
          detail,
        });
      } catch (error: unknown) {
        this.pending.delete(id);
        pending.reject(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }

    this.pending.delete(id);
    if (type === 'cancelled') {
      pending.reject(new DOMException('Inference job cancelled.', 'AbortError'));
      return;
    }
    if (type === 'error') {
      pending.reject(new Error(typeof message['message'] === 'string' ? message['message'] : 'Inference worker failed.'));
      return;
    }
    if (type === 'complete' && pending.kind === 'rife') {
      const rawFrames = message['frames'];
      const provider = message['provider'];
      if (!Array.isArray(rawFrames) || (provider !== 'webgpu' && provider !== 'wasm')) {
        pending.reject(new Error('Inference worker returned an invalid RIFE completion message.'));
        return;
      }
      try {
        const transferred = rawFrames.map(parseTransferFrame);
        const durations = parseDurations(message['durations']);
        if (durations.length !== transferred.length) throw new Error('Inference worker frame and duration counts differ.');
        pending.resolve({
          frames: transferred.map(imageFrame),
          durations,
          provider,
        });
      } catch (error: unknown) {
        pending.reject(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    if (type === 'render-complete' && pending.kind === 'render') {
      try {
        const webp = message['webp'];
        const rawPairwise = message['pairwise'];
        if (!(webp instanceof ArrayBuffer)) throw new Error('Inference worker WebP result is invalid.');
        if (!Array.isArray(rawPairwise)) throw new Error('Inference worker alignment report is invalid.');
        pending.resolve({
          blob: new Blob([webp], { type: 'image/webp' }),
          provider: parseProvider(message['provider']),
          computeBackend: parseComputeBackend(message['computeBackend']),
          pairwise: rawPairwise.map(parseShiftResult),
        });
      } catch (error: unknown) {
        pending.reject(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }
    pending.reject(new Error(`Inference worker returned unsupported message type ${type}.`));
  }

  interpolateRife(
    frames: readonly ImageData[],
    durations: readonly (number | null)[],
    fallbackDuration: number,
    multiplier: RifeMultiplier,
    onProgress: (progress: InferenceWorkerProgress) => void,
  ): { readonly jobId: string; readonly promise: Promise<RifeWorkerResult> } {
    if (frames.length === 0) throw new Error('RIFE worker requires at least one frame.');
    if (durations.length !== frames.length) throw new Error('RIFE worker frame and duration counts differ.');
    this.counter += 1;
    const id = jobId('rife', this.counter);
    const transferred = frames.map(transferFrame);
    const request: InferenceWorkerRequest = {
      type: 'start-rife',
      jobId: id,
      frames: transferred,
      durations,
      fallbackDuration,
      multiplier,
    };
    const promise = new Promise<RifeWorkerResult>((resolve, reject) => {
      this.pending.set(id, { kind: 'rife', resolve, reject, onProgress });
      this.worker.postMessage(request, transferred.map((frame) => frame.buffer));
    });
    return { jobId: id, promise };
  }

  renderFiles(
    files: readonly File[],
    registration: RegistrationSettings,
    interpolation: RenderInterpolationEngine,
    multiplier: RifeMultiplier,
    duration: number,
    loop: number,
    quality: number,
    onProgress: (progress: RenderWorkerProgress) => void,
  ): { readonly jobId: string; readonly promise: Promise<RenderWorkerResult> } {
    if (files.length === 0) throw new Error('Render worker requires at least one source file.');
    this.counter += 1;
    const id = jobId('render', this.counter);
    const request: InferenceWorkerRequest = {
      type: 'start-render',
      jobId: id,
      files,
      registration,
      interpolation,
      multiplier,
      duration,
      loop,
      quality,
    };
    const promise = new Promise<RenderWorkerResult>((resolve, reject) => {
      this.pending.set(id, { kind: 'render', resolve, reject, onProgress });
      this.worker.postMessage(request);
    });
    return { jobId: id, promise };
  }

  cancel(jobIdValue: string): void {
    const request: InferenceWorkerRequest = { type: 'cancel', jobId: jobIdValue };
    this.worker.postMessage(request);
  }

  close(): void {
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(new Error('Inference worker closed.'));
    this.pending.clear();
  }
}
