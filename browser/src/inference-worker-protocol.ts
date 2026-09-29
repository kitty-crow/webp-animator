import type { ModelExecutionProvider } from './inference/ort-runtime.js';

export type RifeMultiplier = 2 | 4 | 8;

export interface TransferFrame {
  readonly width: number;
  readonly height: number;
  readonly buffer: ArrayBuffer;
}

export interface StartRifeRequest {
  readonly type: 'start-rife';
  readonly jobId: string;
  readonly frames: readonly TransferFrame[];
  readonly durations: readonly (number | null)[];
  readonly fallbackDuration: number;
  readonly multiplier: RifeMultiplier;
}

export interface CancelInferenceRequest {
  readonly type: 'cancel';
  readonly jobId: string;
}

export type InferenceWorkerRequest = StartRifeRequest | CancelInferenceRequest;

export interface InferenceProgressResponse {
  readonly type: 'progress';
  readonly jobId: string;
  readonly stage: 'initialising' | 'interpolating';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface InferenceCompleteResponse {
  readonly type: 'complete';
  readonly jobId: string;
  readonly frames: readonly TransferFrame[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

export interface InferenceCancelledResponse {
  readonly type: 'cancelled';
  readonly jobId: string;
}

export interface InferenceErrorResponse {
  readonly type: 'error';
  readonly jobId: string;
  readonly message: string;
}

export type InferenceWorkerResponse =
  | InferenceProgressResponse
  | InferenceCompleteResponse
  | InferenceCancelledResponse
  | InferenceErrorResponse;

function recordOf(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null) throw new Error('Inference worker message must be an object.');
  return value as Record<string, unknown>;
}

function stringField(record: Readonly<Record<string, unknown>>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string' || value.length === 0) throw new Error(`Inference worker ${key} must be a non-empty string.`);
  return value;
}

function numericField(record: Readonly<Record<string, unknown>>, key: string): number {
  const value = record[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`Inference worker ${key} must be finite.`);
  return value;
}

function parseTransferFrame(value: unknown): TransferFrame {
  const record = recordOf(value);
  const width = numericField(record, 'width');
  const height = numericField(record, 'height');
  const buffer = record['buffer'];
  if (!(buffer instanceof ArrayBuffer)) throw new Error('Inference frame buffer must be an ArrayBuffer.');
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error('Inference frame dimensions must be positive integers.');
  }
  const expected = width * height * 4;
  if (buffer.byteLength !== expected) throw new Error(`Inference frame byte length mismatch: expected ${expected}, received ${buffer.byteLength}.`);
  return { width, height, buffer };
}

function parseDurations(value: unknown): readonly (number | null)[] {
  if (!Array.isArray(value)) throw new Error('Inference durations must be an array.');
  return value.map((entry: unknown): number | null => {
    if (entry === null) return null;
    if (typeof entry !== 'number' || !Number.isFinite(entry) || entry <= 0) {
      throw new Error('Inference durations must contain positive finite numbers or null.');
    }
    return entry;
  });
}

function parseMultiplier(value: unknown): RifeMultiplier {
  if (value === 2 || value === 4 || value === 8) return value;
  throw new Error('RIFE multiplier must be 2, 4 or 8.');
}

export function parseInferenceWorkerRequest(value: unknown): InferenceWorkerRequest {
  const record = recordOf(value);
  const type = stringField(record, 'type');
  const jobId = stringField(record, 'jobId');
  if (type === 'cancel') return { type, jobId };
  if (type !== 'start-rife') throw new Error(`Unsupported inference worker request: ${type}.`);

  const rawFrames = record['frames'];
  if (!Array.isArray(rawFrames) || rawFrames.length === 0) throw new Error('RIFE worker request must contain at least one frame.');
  const frames = rawFrames.map(parseTransferFrame);
  const durations = parseDurations(record['durations']);
  if (durations.length !== frames.length) throw new Error('RIFE frame and duration counts differ.');
  const fallbackDuration = numericField(record, 'fallbackDuration');
  if (fallbackDuration <= 0) throw new Error('RIFE fallback duration must be positive.');
  const multiplier = parseMultiplier(record['multiplier']);
  return { type, jobId, frames, durations, fallbackDuration, multiplier };
}
