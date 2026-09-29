import type { ModelExecutionProvider } from './inference/ort-runtime.js';
import type {
  InferenceWorkerRequest,
  InferenceWorkerResponse,
  RifeMultiplier,
  TransferFrame,
} from './inference-worker-protocol.js';

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

interface PendingJob {
  readonly resolve: (result: RifeWorkerResult) => void;
  readonly reject: (error: Error) => void;
  readonly onProgress: (progress: InferenceWorkerProgress) => void;
}

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

function jobId(counter: number): string {
  return `rife-${Date.now().toString(36)}-${counter.toString(36)}`;
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
    const message = value as Partial<InferenceWorkerResponse>;
    if (typeof message.jobId !== 'string' || typeof message.type !== 'string') return;
    const pending = this.pending.get(message.jobId);
    if (!pending) return;

    if (message.type === 'progress') {
      if (
        (message.stage === 'initialising' || message.stage === 'interpolating') &&
        typeof message.current === 'number' &&
        typeof message.total === 'number' &&
        (message.provider === 'webgpu' || message.provider === 'wasm' || message.provider === null)
      ) {
        pending.onProgress({
          stage: message.stage,
          current: message.current,
          total: message.total,
          provider: message.provider,
        });
      }
      return;
    }

    this.pending.delete(message.jobId);
    if (message.type === 'cancelled') {
      pending.reject(new DOMException('Inference job cancelled.', 'AbortError'));
      return;
    }
    if (message.type === 'error') {
      pending.reject(new Error(typeof message.message === 'string' ? message.message : 'Inference worker failed.'));
      return;
    }
    if (
      message.type === 'complete' &&
      Array.isArray(message.frames) &&
      Array.isArray(message.durations) &&
      (message.provider === 'webgpu' || message.provider === 'wasm')
    ) {
      const frames = message.frames.map((frame: TransferFrame) => imageFrame(frame));
      pending.resolve({ frames, durations: message.durations, provider: message.provider });
      return;
    }
    pending.reject(new Error('Inference worker returned an invalid completion message.'));
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
    const id = jobId(this.counter);
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
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage(request, transferred.map((frame) => frame.buffer));
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
