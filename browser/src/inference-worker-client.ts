import type { ModelExecutionProvider } from './inference/ort-runtime.js';
import type {
  InferenceWorkerRequest,
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
    const message = value as Record<string, unknown>;
    const type = message['type'];
    const id = message['jobId'];
    if (typeof type !== 'string' || typeof id !== 'string') return;
    const pending = this.pending.get(id);
    if (!pending) return;

    if (type === 'progress') {
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

    this.pending.delete(id);
    if (type === 'cancelled') {
      pending.reject(new DOMException('Inference job cancelled.', 'AbortError'));
      return;
    }
    if (type === 'error') {
      pending.reject(new Error(typeof message['message'] === 'string' ? message['message'] : 'Inference worker failed.'));
      return;
    }
    if (type === 'complete') {
      const rawFrames = message['frames'];
      const provider = message['provider'];
      if (!Array.isArray(rawFrames) || (provider !== 'webgpu' && provider !== 'wasm')) {
        pending.reject(new Error('Inference worker returned an invalid completion message.'));
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
