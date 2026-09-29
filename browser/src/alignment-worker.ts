import type { WasmAlignmentModule, WorkerScoreRequest, WorkerScoreResponse } from './types.js';

type WasmFactory = (options: { readonly locateFile: (path: string) => string }) => Promise<WasmAlignmentModule> | WasmAlignmentModule;

interface WorkerScopeLike {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: WorkerScoreResponse, transfer?: Transferable[]): void;
}

const scope = globalThis as unknown as WorkerScopeLike;
let wasmModulePromise: Promise<WasmAlignmentModule> | null = null;
let wasmMode: 'simd' | 'scalar' | null = null;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isArrayBufferLike(value: unknown): value is ArrayBufferLike {
  return value instanceof ArrayBuffer || (typeof SharedArrayBuffer === 'function' && value instanceof SharedArrayBuffer);
}

function parseRequest(value: unknown): WorkerScoreRequest {
  if (typeof value !== 'object' || value === null) throw new Error('Worker request must be an object.');
  const record = value as Record<string, unknown>;
  const id = record['id'];
  const aBuffer = record['aBuffer'];
  const bBuffer = record['bBuffer'];
  const candidates = record['candidates'];
  const aw = record['aw'];
  const ah = record['ah'];
  const bw = record['bw'];
  const bh = record['bh'];
  const sigma = record['sigma'];
  const alphaThreshold = record['alphaThreshold'];
  const normaliser = record['normaliser'];
  const preferWasm = record['preferWasm'];
  const preferSimd = record['preferSimd'];

  if (typeof id !== 'number') throw new Error('Worker request has no numeric id.');
  if (!isArrayBufferLike(aBuffer) || !isArrayBufferLike(bBuffer)) throw new Error('Worker request is missing frame buffers.');
  if (!(candidates instanceof Int32Array)) throw new Error('Worker request candidates must be Int32Array.');
  if (typeof aw !== 'number' || typeof ah !== 'number' || typeof bw !== 'number' || typeof bh !== 'number') throw new Error('Worker frame dimensions must be numeric.');
  if (typeof sigma !== 'number' || typeof alphaThreshold !== 'number' || typeof normaliser !== 'number') throw new Error('Worker score parameters must be numeric.');
  if (typeof preferWasm !== 'boolean' || typeof preferSimd !== 'boolean') throw new Error('Worker backend preferences must be boolean.');

  return {
    id,
    aBuffer,
    bBuffer,
    aw,
    ah,
    bw,
    bh,
    candidates,
    sigma,
    alphaThreshold,
    normaliser,
    preferWasm,
    preferSimd,
  };
}

function jsScoreTranslation(
  first: Uint8Array,
  aw: number,
  ah: number,
  second: Uint8Array,
  bw: number,
  bh: number,
  dx: number,
  dy: number,
  sigma: number,
  alphaThreshold: number,
  normaliser: number,
): number {
  const x0 = Math.max(0, dx);
  const y0 = Math.max(0, dy);
  const x1 = Math.min(aw, dx + bw);
  const y1 = Math.min(ah, dy + bh);
  if (x1 <= x0 || y1 <= y0) return Number.NEGATIVE_INFINITY;

  let total = 0;
  const inverseSigma = 1 / sigma;
  for (let y = y0; y < y1; y += 1) {
    const by = y - dy;
    for (let x = x0; x < x1; x += 1) {
      const bx = x - dx;
      const ai = (y * aw + x) * 4;
      const bi = (by * bw + bx) * 4;
      const aAlpha = first[ai + 3];
      const bAlpha = second[bi + 3];
      if (aAlpha === undefined || bAlpha === undefined) throw new Error('Frame buffer bounds mismatch.');
      if (aAlpha <= alphaThreshold && bAlpha <= alphaThreshold) continue;

      const ar = first[ai];
      const ag = first[ai + 1];
      const ab = first[ai + 2];
      const br = second[bi];
      const bg = second[bi + 1];
      const bb = second[bi + 2];
      if (ar === undefined || ag === undefined || ab === undefined || br === undefined || bg === undefined || bb === undefined) throw new Error('Frame buffer bounds mismatch.');
      const distance = (
        Math.abs(ar - br) +
        Math.abs(ag - bg) +
        Math.abs(ab - bb) +
        Math.abs(aAlpha - bAlpha)
      ) * 0.25;
      const z = distance * inverseSigma;
      total += Math.exp(-(z * z));
    }
  }
  return total / Math.max(normaliser, 1);
}

function isModuleNamespace(value: unknown): value is { readonly default: unknown } {
  return typeof value === 'object' && value !== null && 'default' in value;
}

function isWasmFactory(value: unknown): value is WasmFactory {
  return typeof value === 'function';
}

async function loadWasm(preferSimd: boolean): Promise<WasmAlignmentModule> {
  const wanted: 'simd' | 'scalar' = preferSimd ? 'simd' : 'scalar';
  if (wasmModulePromise && wasmMode === wanted) return wasmModulePromise;
  wasmMode = wanted;
  wasmModulePromise = (async (): Promise<WasmAlignmentModule> => {
    const moduleUrl = new URL(preferSimd ? '../assets/alignment-wasm-simd.js' : '../assets/alignment-wasm.js', import.meta.url);
    const imported: unknown = await import(moduleUrl.href);
    if (!isModuleNamespace(imported) || !isWasmFactory(imported.default)) throw new Error('Generated Emscripten module has an unexpected shape.');
    return imported.default({ locateFile: (path: string): string => new URL(`../assets/${path}`, import.meta.url).href });
  })();
  return wasmModulePromise;
}

async function scoreWithWasm(request: WorkerScoreRequest): Promise<Float64Array> {
  const module = await loadWasm(request.preferSimd);
  const first = new Uint8Array(request.aBuffer);
  const second = new Uint8Array(request.bBuffer);
  const firstPointer = module._malloc(first.byteLength);
  const secondPointer = module._malloc(second.byteLength);
  try {
    module.HEAPU8.set(first, firstPointer);
    module.HEAPU8.set(second, secondPointer);
    const scores = new Float64Array(request.candidates.length / 2);
    for (let index = 0; index < scores.length; index += 1) {
      const dx = request.candidates[index * 2];
      const dy = request.candidates[index * 2 + 1];
      if (dx === undefined || dy === undefined) throw new Error('Candidate pair is incomplete.');
      scores[index] = module._score_translation(firstPointer, request.aw, request.ah, secondPointer, request.bw, request.bh, dx, dy, request.sigma, request.alphaThreshold, request.normaliser);
    }
    return scores;
  } finally {
    module._free(firstPointer);
    module._free(secondPointer);
  }
}

function scoreWithJs(request: WorkerScoreRequest): Float64Array {
  const first = new Uint8Array(request.aBuffer);
  const second = new Uint8Array(request.bBuffer);
  const scores = new Float64Array(request.candidates.length / 2);
  for (let index = 0; index < scores.length; index += 1) {
    const dx = request.candidates[index * 2];
    const dy = request.candidates[index * 2 + 1];
    if (dx === undefined || dy === undefined) throw new Error('Candidate pair is incomplete.');
    scores[index] = jsScoreTranslation(first, request.aw, request.ah, second, request.bw, request.bh, dx, dy, request.sigma, request.alphaThreshold, request.normaliser);
  }
  return scores;
}

scope.addEventListener('message', (event: MessageEvent<unknown>) => {
  void (async (): Promise<void> => {
    let requestId = -1;
    try {
      const request = parseRequest(event.data);
      requestId = request.id;
      let scores: Float64Array;
      let backend: 'wasm-simd' | 'wasm' | 'js' = 'js';
      if (request.preferWasm) {
        try {
          scores = await scoreWithWasm(request);
          backend = request.preferSimd ? 'wasm-simd' : 'wasm';
        } catch (error: unknown) {
          console.warn('WASM worker path failed; using typed CPU fallback.', error);
          scores = scoreWithJs(request);
        }
      } else {
        scores = scoreWithJs(request);
      }
      const response: WorkerScoreResponse = { id: request.id, scores, backend, error: null };
      scope.postMessage(response, [scores.buffer]);
    } catch (error: unknown) {
      const response: WorkerScoreResponse = { id: requestId, scores: null, backend: null, error: errorMessage(error) };
      scope.postMessage(response);
    }
  })();
});
