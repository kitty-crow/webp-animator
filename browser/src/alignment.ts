import type { ComputeBackend, HardwareProfile, Position, RegistrationResult, RegistrationSettings, ShiftResult } from './types.js';
import type { ScoreOptions } from './worker-pool.js';
import { AlignmentWorkerPool } from './worker-pool.js';
import { WebGpuScorer } from './webgpu-score.js';

interface CandidateScorer {
  scoreCandidates(first: ImageData, second: ImageData, candidates: Int32Array, options: ScoreOptions): Promise<Float64Array>;
  close?(): void;
}

function makeCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function context2d(canvas: OffscreenCanvas | HTMLCanvasElement): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D is unavailable.');
  return context as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
}

function activePixelCount(image: ImageData, alphaThreshold: number): number {
  let active = 0;
  for (let index = 3; index < image.data.length; index += 4) {
    const alpha = image.data[index];
    if (alpha !== undefined && alpha > alphaThreshold) active += 1;
  }
  return active || image.width * image.height;
}

function values(start: number, end: number, step: number): readonly number[] {
  const output: number[] = [];
  const safeStep = Math.max(1, step);
  for (let value = start; value <= end; value += safeStep) output.push(value);
  if (output.length === 0 || output[output.length - 1] !== end) output.push(end);
  return output;
}

function candidateGrid(
  axis: RegistrationSettings['axis'],
  maxX: number,
  maxY: number,
  step: number,
  centreX = 0,
  centreY = 0,
  radiusX = maxX,
  radiusY = maxY,
): Int32Array {
  const xs = axis === 'y' || axis === 'none'
    ? [0]
    : values(Math.max(-maxX, centreX - radiusX), Math.min(maxX, centreX + radiusX), step);
  const ys = axis === 'x' || axis === 'none'
    ? [0]
    : values(Math.max(-maxY, centreY - radiusY), Math.min(maxY, centreY + radiusY), step);
  const pairs: number[] = [];
  for (const dy of ys) for (const dx of xs) pairs.push(dx, dy);
  return new Int32Array(pairs);
}

function best(candidates: Int32Array, scores: Float64Array): ShiftResult {
  let index = -1;
  let score = Number.NEGATIVE_INFINITY;
  for (let candidate = 0; candidate < scores.length; candidate += 1) {
    const value = scores[candidate];
    if (value !== undefined && value > score) {
      index = candidate;
      score = value;
    }
  }
  if (index < 0) throw new Error('Alignment produced no candidate score.');
  const dx = candidates[index * 2];
  const dy = candidates[index * 2 + 1];
  if (dx === undefined || dy === undefined) throw new Error('Alignment candidate is incomplete.');
  return { dx, dy, score };
}

async function resize(image: ImageData, scale: number): Promise<ImageData> {
  if (Math.abs(scale - 1) < 1e-9) return image;
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  const source = makeCanvas(image.width, image.height);
  context2d(source).putImageData(image, 0, 0);
  const output = makeCanvas(width, height);
  const context = context2d(output);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'medium';
  context.drawImage(source, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

async function makeScorer(profile: HardwareProfile, backend: ComputeBackend): Promise<CandidateScorer> {
  if (backend === 'webgpu' && profile.webgpu.adapter) return WebGpuScorer.create(profile.webgpu.adapter);
  if (backend === 'wasm-simd-workers') return new AlignmentWorkerPool(profile, { preferWasm: true, preferSimd: true });
  if (backend === 'wasm-workers') return new AlignmentWorkerPool(profile, { preferWasm: true, preferSimd: false });
  if (backend === 'js-workers') return new AlignmentWorkerPool(profile, { preferWasm: false, preferSimd: false });
  return {
    async scoreCandidates(first: ImageData, second: ImageData, candidates: Int32Array, options: ScoreOptions): Promise<Float64Array> {
      const pool = new AlignmentWorkerPool(profile, { preferWasm: false, preferSimd: false, workerCount: 1 });
      try {
        return await pool.scoreCandidates(first, second, candidates, options);
      } finally {
        pool.close();
      }
    },
  };
}

export class AdaptiveAlignmentEngine {
  static async create(profile: HardwareProfile, backend: ComputeBackend): Promise<AdaptiveAlignmentEngine> {
    return new AdaptiveAlignmentEngine(await makeScorer(profile, backend));
  }

  private constructor(private readonly scorer: CandidateScorer) {}

  async findBestTranslation(first: ImageData, second: ImageData, settings: RegistrationSettings): Promise<ShiftResult> {
    const maxX = settings.axis === 'y' || settings.axis === 'none' ? 0 : settings.maxShiftX;
    const maxY = settings.axis === 'x' || settings.axis === 'none' ? 0 : settings.maxShiftY;
    const fullNormaliser = Math.max(activePixelCount(first, settings.alphaThreshold), activePixelCount(second, settings.alphaThreshold), 1);
    if (settings.axis === 'none') {
      const candidates = new Int32Array([0, 0]);
      return best(candidates, await this.scorer.scoreCandidates(first, second, candidates, {
        sigma: settings.sigma,
        alphaThreshold: settings.alphaThreshold,
        normaliser: fullNormaliser,
      }));
    }

    const scale = Math.min(1, settings.proxyMaxSide / Math.max(first.width, first.height, second.width, second.height));
    const proxyFirst = await resize(first, scale);
    const proxySecond = await resize(second, scale);
    const proxyX = Math.max(0, Math.round(maxX * scale));
    const proxyY = Math.max(0, Math.round(maxY * scale));
    const coarseStep = Math.max(1, Math.ceil(Math.max(proxyX, proxyY) / 40));
    const proxyNormaliser = Math.max(activePixelCount(proxyFirst, settings.alphaThreshold), activePixelCount(proxySecond, settings.alphaThreshold), 1);

    let candidates = candidateGrid(settings.axis, proxyX, proxyY, coarseStep);
    let result = best(candidates, await this.scorer.scoreCandidates(proxyFirst, proxySecond, candidates, {
      sigma: settings.sigma,
      alphaThreshold: settings.alphaThreshold,
      normaliser: proxyNormaliser,
    }));

    if (coarseStep > 1) {
      candidates = candidateGrid(settings.axis, proxyX, proxyY, 1, result.dx, result.dy, coarseStep, coarseStep);
      result = best(candidates, await this.scorer.scoreCandidates(proxyFirst, proxySecond, candidates, {
        sigma: settings.sigma,
        alphaThreshold: settings.alphaThreshold,
        normaliser: proxyNormaliser,
      }));
    }

    const guessX = Math.round(result.dx / Math.max(scale, 1e-9));
    const guessY = Math.round(result.dy / Math.max(scale, 1e-9));
    const radius = Math.max(3, Math.ceil(1 / Math.max(scale, 1e-9)));
    candidates = candidateGrid(settings.axis, maxX, maxY, 1, guessX, guessY, radius, radius);
    return best(candidates, await this.scorer.scoreCandidates(first, second, candidates, {
      sigma: settings.sigma,
      alphaThreshold: settings.alphaThreshold,
      normaliser: fullNormaliser,
    }));
  }

  async registerSequence(frames: readonly ImageData[], settings: RegistrationSettings, onProgress: (current: number, total: number) => void): Promise<RegistrationResult> {
    if (frames.length === 0) return { positions: [], pairwise: [] };
    const positions: Position[] = [{ x: 0, y: 0 }];
    const pairwise: ShiftResult[] = [];
    for (let index = 1; index < frames.length; index += 1) {
      const previous = frames[index - 1];
      const current = frames[index];
      const priorPosition = positions[index - 1];
      if (!previous || !current || !priorPosition) throw new Error('Frame sequence indexing failed.');
      onProgress(index, frames.length - 1);
      const shift = await this.findBestTranslation(previous, current, settings);
      pairwise.push(shift);
      positions.push({ x: priorPosition.x + shift.dx, y: priorPosition.y + shift.dy });
    }
    return { positions, pairwise };
  }

  close(): void {
    this.scorer.close?.();
  }
}

export function renderUnionFrames(frames: readonly ImageData[], positions: readonly Position[]): readonly ImageData[] {
  if (frames.length === 0 || positions.length !== frames.length) throw new Error('Frame positions are incomplete.');
  const minX = Math.min(...positions.map((position) => position.x));
  const minY = Math.min(...positions.map((position) => position.y));
  const maxX = Math.max(...positions.map((position, index) => position.x + (frames[index]?.width ?? 0)));
  const maxY = Math.max(...positions.map((position, index) => position.y + (frames[index]?.height ?? 0)));
  const width = maxX - minX;
  const height = maxY - minY;
  const output: ImageData[] = [];
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    const position = positions[index];
    if (!frame || !position) throw new Error('Frame rendering index failed.');
    const canvas = makeCanvas(width, height);
    const context = context2d(canvas);
    context.clearRect(0, 0, width, height);
    context.putImageData(frame, position.x - minX, position.y - minY);
    output.push(context.getImageData(0, 0, width, height));
  }
  return output;
}
