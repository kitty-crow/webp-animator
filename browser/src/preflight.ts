import { detectHardware } from './hardware.js';
import type { ComputeBackend, HardwareProfile, PreflightBenchmark, PreflightResult } from './types.js';
import { AlignmentWorkerPool } from './worker-pool.js';
import { WebGpuScorer } from './webgpu-score.js';

interface Scorer {
  scoreCandidates(first: ImageData, second: ImageData, candidates: Int32Array, options: {
    readonly sigma: number;
    readonly alphaThreshold: number;
    readonly normaliser: number;
  }): Promise<Float64Array>;
  close?(): void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function syntheticFrame(width: number, height: number): ImageData {
  const image = new ImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const r = (x * 13 + y * 7) & 0xff;
      const g = (x * 3 + y * 17) & 0xff;
      const b = (x * 19 + y * 5) & 0xff;
      image.data[index] = r;
      image.data[index + 1] = g;
      image.data[index + 2] = b;
      image.data[index + 3] = 255;
    }
  }
  return image;
}

function candidateGrid(radius: number): Int32Array {
  const pairs: number[] = [];
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) pairs.push(dx, dy);
  }
  return new Int32Array(pairs);
}

function winningCandidate(candidates: Int32Array, scores: Float64Array): { readonly dx: number; readonly dy: number; readonly score: number } {
  let bestIndex = -1;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < scores.length; index += 1) {
    const score = scores[index];
    if (score !== undefined && score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  if (bestIndex < 0) throw new Error('Benchmark backend returned no finite score.');
  const dx = candidates[bestIndex * 2];
  const dy = candidates[bestIndex * 2 + 1];
  if (dx === undefined || dy === undefined) throw new Error('Benchmark candidate indexing failed.');
  return { dx, dy, score: bestScore };
}

async function benchmark(backend: ComputeBackend, scorer: Scorer): Promise<PreflightBenchmark> {
  const frame = syntheticFrame(96, 96);
  const candidates = candidateGrid(4);
  const options = { sigma: 28, alphaThreshold: 8, normaliser: frame.width * frame.height };
  try {
    await scorer.scoreCandidates(frame, frame, candidates, options);
    const started = performance.now();
    const scores = await scorer.scoreCandidates(frame, frame, candidates, options);
    const elapsed = performance.now() - started;
    const winner = winningCandidate(candidates, scores);
    const passed = winner.dx === 0 && winner.dy === 0 && Number.isFinite(winner.score);
    return {
      backend,
      milliseconds: elapsed,
      score: winner.score,
      passed,
      error: passed ? null : `Self-test selected ${winner.dx},${winner.dy} instead of 0,0.`,
    };
  } catch (error: unknown) {
    return { backend, milliseconds: Number.POSITIVE_INFINITY, score: Number.NEGATIVE_INFINITY, passed: false, error: errorMessage(error) };
  } finally {
    scorer.close?.();
  }
}

async function workerBenchmark(profile: HardwareProfile, backend: ComputeBackend): Promise<PreflightBenchmark> {
  const preferWasm = backend === 'wasm-simd-workers' || backend === 'wasm-workers';
  const preferSimd = backend === 'wasm-simd-workers';
  const pool = new AlignmentWorkerPool(profile, { preferWasm, preferSimd, workerCount: profile.workerCount });
  return benchmark(backend, pool);
}

export async function runPreflight(): Promise<PreflightResult> {
  const profile = await detectHardware();
  const tests: Promise<PreflightBenchmark>[] = [];

  if (profile.webgpu.available && profile.webgpu.adapter) {
    tests.push((async (): Promise<PreflightBenchmark> => {
      try {
        const scorer = await WebGpuScorer.create(profile.webgpu.adapter);
        return benchmark('webgpu', scorer);
      } catch (error: unknown) {
        return { backend: 'webgpu', milliseconds: Number.POSITIVE_INFINITY, score: Number.NEGATIVE_INFINITY, passed: false, error: errorMessage(error) };
      }
    })());
  }

  if (profile.workerSupport && profile.wasm && profile.wasmSimd) tests.push(workerBenchmark(profile, 'wasm-simd-workers'));
  if (profile.workerSupport && profile.wasm) tests.push(workerBenchmark(profile, 'wasm-workers'));
  if (profile.workerSupport) tests.push(workerBenchmark(profile, 'js-workers'));

  const benchmarks = await Promise.all(tests);
  const successful = benchmarks.filter((entry) => entry.passed).sort((left, right) => left.milliseconds - right.milliseconds);
  const selectedBackend: ComputeBackend = successful[0]?.backend ?? 'main-thread-js';
  return { profile, selectedBackend, benchmarks, completedAt: Date.now() };
}
