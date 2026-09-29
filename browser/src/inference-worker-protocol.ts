import type { ModelExecutionProvider } from './inference/ort-runtime.js';
import type { ComputeBackend, RegistrationSettings, ShiftResult } from './types.js';

export type RifeMultiplier = 2 | 4 | 8;
export type RenderInterpolationEngine = 'none' | 'rife' | 'amt' | 'resshift' | 'mog' | 'tooncrafter';
export type RenderProgressStage = 'preflight' | 'decode' | 'align' | 'interpolate' | 'encode';

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

export interface StartRenderRequest {
  readonly type: 'start-render';
  readonly jobId: string;
  readonly files: readonly File[];
  readonly registration: RegistrationSettings;
  readonly interpolation: RenderInterpolationEngine;
  readonly modelManifestUrl: string | null;
  readonly multiplier: RifeMultiplier;
  readonly duration: number;
  readonly loop: number;
  readonly quality: number;
}

export interface CancelInferenceRequest {
  readonly type: 'cancel';
  readonly jobId: string;
}

export type InferenceWorkerRequest = StartRifeRequest | StartRenderRequest | CancelInferenceRequest;

export interface InferenceProgressResponse {
  readonly type: 'progress';
  readonly jobId: string;
  readonly stage: 'initialising' | 'interpolating';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface RenderProgressResponse {
  readonly type: 'render-progress';
  readonly jobId: string;
  readonly stage: RenderProgressStage;
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly computeBackend: ComputeBackend | null;
  readonly detail: string | null;
}

export interface InferenceCompleteResponse {
  readonly type: 'complete';
  readonly jobId: string;
  readonly frames: readonly TransferFrame[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

export interface RenderCompleteResponse {
  readonly type: 'render-complete';
  readonly jobId: string;
  readonly webp: ArrayBuffer;
  readonly provider: ModelExecutionProvider | null;
  readonly computeBackend: ComputeBackend;
  readonly pairwise: readonly ShiftResult[];
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
  | RenderProgressResponse
  | InferenceCompleteResponse
  | RenderCompleteResponse
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

function nonNegativeInteger(record: Readonly<Record<string, unknown>>, key: string): number {
  const value = numericField(record, key);
  if (!Number.isInteger(value) || value < 0) throw new Error(`Inference worker ${key} must be a non-negative integer.`);
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
  throw new Error('Interpolation multiplier must be 2, 4 or 8.');
}

function parseRegistration(value: unknown): RegistrationSettings {
  const record = recordOf(value);
  const axisValue = record['axis'];
  const axis: RegistrationSettings['axis'] = axisValue === 'x' || axisValue === 'y' || axisValue === 'none' || axisValue === 'xy'
    ? axisValue
    : (() => { throw new Error('Registration axis is invalid.'); })();
  const maxShiftX = numericField(record, 'maxShiftX');
  const maxShiftY = numericField(record, 'maxShiftY');
  const sigma = numericField(record, 'sigma');
  const alphaThreshold = numericField(record, 'alphaThreshold');
  const proxyMaxSide = numericField(record, 'proxyMaxSide');
  if (maxShiftX < 0 || maxShiftY < 0) throw new Error('Registration shifts must be non-negative.');
  if (sigma <= 0 || proxyMaxSide < 1) throw new Error('Registration sigma and proxy size must be positive.');
  if (!Number.isInteger(alphaThreshold) || alphaThreshold < 0 || alphaThreshold > 255) throw new Error('Registration alpha threshold is invalid.');
  return { axis, maxShiftX, maxShiftY, sigma, alphaThreshold, proxyMaxSide };
}

function parseFiles(value: unknown): readonly File[] {
  if (!Array.isArray(value) || value.length === 0) throw new Error('Render worker requires at least one source file.');
  return value.map((entry: unknown): File => {
    if (!(entry instanceof File)) throw new Error('Render worker source must be a File.');
    return entry;
  });
}

function parseInterpolationEngine(value: unknown): RenderInterpolationEngine {
  if (value === 'none' || value === 'rife' || value === 'amt' || value === 'resshift' || value === 'mog' || value === 'tooncrafter') return value;
  throw new Error('Render interpolation model is invalid.');
}

function parseManifestUrl(value: unknown, interpolation: RenderInterpolationEngine): string | null {
  if (interpolation === 'none' || interpolation === 'rife') {
    if (value === null || value === undefined || value === '') return null;
    if (typeof value !== 'string') throw new Error('Render model manifest URL is invalid.');
    return value;
  }
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${interpolation} requires an exported model manifest URL.`);
  }
  return value.trim();
}

function parseRender(record: Readonly<Record<string, unknown>>, jobId: string): StartRenderRequest {
  const interpolation = parseInterpolationEngine(record['interpolation']);
  const duration = numericField(record, 'duration');
  const quality = numericField(record, 'quality');
  if (duration <= 0) throw new Error('Render duration must be positive.');
  if (quality <= 0 || quality > 1) throw new Error('Render quality must be in (0, 1].');
  return {
    type: 'start-render',
    jobId,
    files: parseFiles(record['files']),
    registration: parseRegistration(record['registration']),
    interpolation,
    modelManifestUrl: parseManifestUrl(record['modelManifestUrl'], interpolation),
    multiplier: parseMultiplier(record['multiplier']),
    duration,
    loop: nonNegativeInteger(record, 'loop'),
    quality,
  };
}

export function parseInferenceWorkerRequest(value: unknown): InferenceWorkerRequest {
  const record = recordOf(value);
  const type = stringField(record, 'type');
  const jobId = stringField(record, 'jobId');
  if (type === 'cancel') return { type, jobId };
  if (type === 'start-render') return parseRender(record, jobId);
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
