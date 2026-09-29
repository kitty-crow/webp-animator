export type ComputeBackend = 'webgpu' | 'wasm-simd-workers' | 'wasm-workers' | 'js-workers' | 'main-thread-js';

export interface GpuAdapterSummary {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

export interface WebGpuCapability {
  readonly available: boolean;
  readonly adapter: GPUAdapter | null;
  readonly info: GpuAdapterSummary | null;
  readonly limits: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxComputeWorkgroupsPerDimension: number;
  } | null;
  readonly error: string | null;
}

export interface WebGlCapability {
  readonly available: boolean;
  readonly renderer: string | null;
  readonly maxTextureSize: number;
  readonly error: string | null;
}

export interface HardwareProfile {
  readonly logicalCores: number;
  readonly workerCount: number;
  readonly deviceMemoryGiB: number | null;
  readonly heapLimitMiB: number | null;
  readonly memoryTier: 'low' | 'medium' | 'high';
  readonly memoryBudgetMiB: number;
  readonly mobile: boolean;
  readonly sharedMemory: boolean;
  readonly crossOriginIsolated: boolean;
  readonly workerSupport: boolean;
  readonly wasm: boolean;
  readonly wasmSimd: boolean;
  readonly webgpu: WebGpuCapability;
  readonly webgl2: WebGlCapability;
  readonly offscreenCanvas: boolean;
  readonly imageDecoder: boolean;
}

export interface RegistrationSettings {
  readonly axis: 'x' | 'y' | 'xy' | 'none';
  readonly maxShiftX: number;
  readonly maxShiftY: number;
  readonly sigma: number;
  readonly alphaThreshold: number;
  readonly proxyMaxSide: number;
}

export interface ShiftResult {
  readonly dx: number;
  readonly dy: number;
  readonly score: number;
}

export interface Position {
  readonly x: number;
  readonly y: number;
}

export interface RegistrationResult {
  readonly positions: readonly Position[];
  readonly pairwise: readonly ShiftResult[];
}

export interface PreflightBenchmark {
  readonly backend: ComputeBackend;
  readonly milliseconds: number;
  readonly score: number;
  readonly passed: boolean;
  readonly error: string | null;
}

export interface PreflightResult {
  readonly profile: HardwareProfile;
  readonly selectedBackend: ComputeBackend;
  readonly benchmarks: readonly PreflightBenchmark[];
  readonly completedAt: number;
}

export interface DecodeProgress {
  readonly stage: 'decode';
  readonly current: number;
  readonly total: number;
  readonly fileName: string | null;
}

export interface OperationProgress {
  readonly stage: 'align' | 'encode';
  readonly current: number;
  readonly total: number;
}

export type ProgressUpdate = DecodeProgress | OperationProgress;

export interface DecodedInput {
  readonly frames: readonly ImageData[];
  readonly sourceDurations: readonly (number | null)[];
}

export interface WorkerScoreRequest {
  readonly id: number;
  readonly aBuffer: ArrayBufferLike;
  readonly bBuffer: ArrayBufferLike;
  readonly aw: number;
  readonly ah: number;
  readonly bw: number;
  readonly bh: number;
  readonly candidates: Int32Array;
  readonly sigma: number;
  readonly alphaThreshold: number;
  readonly normaliser: number;
  readonly preferWasm: boolean;
  readonly preferSimd: boolean;
}

export interface WorkerScoreSuccess {
  readonly id: number;
  readonly scores: Float64Array;
  readonly backend: 'wasm-simd' | 'wasm' | 'js';
  readonly error: null;
}

export interface WorkerScoreFailure {
  readonly id: number;
  readonly scores: null;
  readonly backend: null;
  readonly error: string;
}

export type WorkerScoreResponse = WorkerScoreSuccess | WorkerScoreFailure;

export interface WasmAlignmentModule {
  readonly HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(pointer: number): void;
  _score_translation(
    aPointer: number,
    aw: number,
    ah: number,
    bPointer: number,
    bw: number,
    bh: number,
    dx: number,
    dy: number,
    sigma: number,
    alphaThreshold: number,
    normaliser: number,
  ): number;
}

export interface ImageDecoderTrack {
  readonly frameCount: number;
}

export interface ImageDecoderTracks {
  readonly ready: Promise<void>;
  readonly selectedTrack: ImageDecoderTrack | null;
}

export interface ImageDecoderDecodeResult {
  readonly image: VideoFrame;
}

export interface BrowserImageDecoder {
  readonly tracks: ImageDecoderTracks;
  decode(options: { readonly frameIndex: number; readonly completeFramesOnly: boolean }): Promise<ImageDecoderDecodeResult>;
  close(): void;
}

export interface BrowserImageDecoderConstructor {
  new (options: { readonly data: ArrayBuffer; readonly type: string }): BrowserImageDecoder;
  isTypeSupported(type: string): Promise<boolean>;
}
