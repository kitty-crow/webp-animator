import type { HardwareProfile, WorkerScoreRequest, WorkerScoreResponse } from './types.js';

interface WorkerPoolOptions {
  readonly workerCount?: number;
  readonly preferWasm?: boolean;
  readonly preferSimd?: boolean;
}

interface PendingRequest {
  readonly resolve: (response: WorkerScoreResponse) => void;
  readonly reject: (error: Error) => void;
}

export interface ScoreOptions {
  readonly sigma: number;
  readonly alphaThreshold: number;
  readonly normaliser: number;
}

function chunkCandidates(candidates: Int32Array, count: number): readonly Int32Array[] {
  const pairs = candidates.length / 2;
  const chunks: Int32Array[] = [];
  let cursor = 0;
  for (let index = 0; index < count && cursor < pairs; index += 1) {
    const remaining = pairs - cursor;
    const remainingWorkers = count - index;
    const take = Math.ceil(remaining / remainingWorkers);
    chunks.push(candidates.slice(cursor * 2, (cursor + take) * 2));
    cursor += take;
  }
  return chunks;
}

function toShared(bytes: Uint8ClampedArray): Uint8Array {
  if (bytes.buffer instanceof SharedArrayBuffer) return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const shared = new SharedArrayBuffer(bytes.byteLength);
  const view = new Uint8Array(shared);
  view.set(bytes);
  return view;
}

function parseWorkerResponse(value: unknown): WorkerScoreResponse {
  if (typeof value !== 'object' || value === null) throw new Error('Worker returned a non-object response.');
  const record = value as Record<string, unknown>;
  const id = record['id'];
  const error = record['error'];
  const scores = record['scores'];
  const backend = record['backend'];
  if (typeof id !== 'number') throw new Error('Worker response is missing request id.');
  if (typeof error === 'string') return { id, scores: null, backend: null, error };
  if (!(scores instanceof Float64Array)) throw new Error('Worker response is missing score data.');
  if (backend !== 'wasm-simd' && backend !== 'wasm' && backend !== 'js') throw new Error('Worker response has an unknown backend.');
  return { id, scores, backend, error: null };
}

export class AlignmentWorkerPool {
  private readonly profile: HardwareProfile;
  private readonly preferWasm: boolean;
  private readonly preferSimd: boolean;
  private readonly workers: Worker[] = [];
  private readonly pending = new Map<number, PendingRequest>();
  private nextId = 1;

  constructor(profile: HardwareProfile, options: WorkerPoolOptions = {}) {
    this.profile = profile;
    this.preferWasm = options.preferWasm ?? profile.wasm;
    this.preferSimd = options.preferSimd ?? profile.wasmSimd;
    const requestedCount = (options.workerCount ?? profile.workerCount) || 1;
    const count = Math.max(1, requestedCount);
    for (let index = 0; index < count; index += 1) {
      const worker = new Worker(new URL('./alignment-worker.js', import.meta.url), { type: 'module' });
      worker.addEventListener('message', (event: MessageEvent<unknown>) => {
        let response: WorkerScoreResponse;
        try {
          response = parseWorkerResponse(event.data);
        } catch (error: unknown) {
          console.error('Rejected malformed alignment worker response.', error);
          return;
        }
        const callback = this.pending.get(response.id);
        if (!callback) return;
        this.pending.delete(response.id);
        if (response.error !== null) callback.reject(new Error(response.error));
        else callback.resolve(response);
      });
      worker.addEventListener('error', (event: ErrorEvent) => console.error('Alignment worker error.', event.error ?? event.message));
      this.workers.push(worker);
    }
  }

  private run(worker: Worker, payload: Omit<WorkerScoreRequest, 'id'>): Promise<WorkerScoreResponse> {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise<WorkerScoreResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ ...payload, id } satisfies WorkerScoreRequest);
    });
  }

  async scoreCandidates(first: ImageData, second: ImageData, candidates: Int32Array, options: ScoreOptions): Promise<Float64Array> {
    if (candidates.length === 0) return new Float64Array();
    const candidateCount = candidates.length / 2;
    const workerCount = Math.min(this.workers.length, Math.max(1, candidateCount));
    const chunks = chunkCandidates(candidates, workerCount);
    const useShared = this.profile.sharedMemory;
    const firstBytes = useShared ? toShared(first.data) : new Uint8Array(first.data.buffer.slice(0));
    const secondBytes = useShared ? toShared(second.data) : new Uint8Array(second.data.buffer.slice(0));

    const jobs = chunks.map((chunk, index) => {
      const worker = this.workers[index];
      if (!worker) throw new Error(`Worker ${index} is unavailable.`);
      return this.run(worker, {
        aBuffer: firstBytes.buffer,
        bBuffer: secondBytes.buffer,
        aw: first.width,
        ah: first.height,
        bw: second.width,
        bh: second.height,
        candidates: chunk,
        sigma: options.sigma,
        alphaThreshold: options.alphaThreshold,
        normaliser: options.normaliser,
        preferWasm: this.preferWasm,
        preferSimd: this.preferSimd,
      });
    });

    const parts = await Promise.all(jobs);
    const output = new Float64Array(candidateCount);
    let offset = 0;
    for (const part of parts) {
      if (part.scores === null) throw new Error(part.error ?? 'Worker scoring failed.');
      output.set(part.scores, offset);
      offset += part.scores.length;
    }
    return output;
  }

  close(): void {
    for (const worker of this.workers) worker.terminate();
    this.workers.length = 0;
    this.pending.clear();
  }
}
