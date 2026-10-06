import { describe, expect, test } from 'bun:test';

import { parseInferenceWorkerRequest } from '../src/inference-worker-protocol.js';
import { BROWSER_MODEL_CATALOG, browserModelDefinition } from '../src/inference/catalog.js';
import { ddimStep, makeLegacyLdmUniformDdimSchedule, makeUniformDdimSchedule } from '../src/inference/ddim.js';
import { DEFAULT_MOG_SCHEDULE, makeMogDdimSchedule, mogDdimStep, mogDynamicScale, mogTimesteps, mogVPrediction } from '../src/inference/mog-scheduler.js';
import { dilatePlane } from '../src/inference/propainter-mask.js';
import { rifePaddedDimension } from '../src/inference/rife.js';
import { initialiseResShiftSample, makeResShiftSchedule, resShiftReverseStep } from '../src/inference/resshift-scheduler.js';

describe('browser ONNX runtime loading', () => {
  test('uses a worker-safe deployed module URL instead of a document import map', async () => {
    const runtimeSource = await Bun.file(new URL('../src/inference/ort-runtime.ts', import.meta.url)).text();
    const pageSource = await Bun.file(new URL('../index.html', import.meta.url)).text();
    expect(runtimeSource).toContain("new URL('../../vendor/ort.webgpu.bundle.min.mjs', import.meta.url)");
    expect(runtimeSource).not.toContain("import * as ort from 'onnxruntime-web/webgpu'");
    expect(pageSource).not.toContain('type="importmap"');
  });
});

describe('browser model catalogue', () => {
  test('model families are unique and use automatic WebGPU to WASM fallback', () => {
    const families = BROWSER_MODEL_CATALOG.map((definition) => definition.family);
    expect(new Set(families).size).toBe(families.length);
    for (const definition of BROWSER_MODEL_CATALOG) {
      expect(definition.preferredProvider).toBe('webgpu');
      expect(definition.fallbackProvider).toBe('wasm');
      expect(definition.components.length).toBeGreaterThan(0);
    }
  });

  test('RIFE 4.25 browser asset is pinned and integrity checked', () => {
    const rife = browserModelDefinition('rife');
    expect(rife.status).toBe('adapter-ready');
    expect(rife.assets).toHaveLength(1);
    const asset = rife.assets[0];
    expect(asset).toBeDefined();
    if (!asset) throw new Error('RIFE asset unexpectedly missing.');
    expect(asset.bytes).toBe(22_748_049);
    expect(asset.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(asset.sha256).toBe('65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584');
    expect(asset.licence).toBe('MIT');
  });

  test('ProPainter remains explicitly licence-gated', () => {
    expect(browserModelDefinition('propainter').status).toBe('licence-gated');
  });
});

describe('RIFE browser preprocessing', () => {
  test('pads arbitrary frame sizes to the native scale-1 model multiple', () => {
    expect(rifePaddedDimension(1)).toBe(128);
    expect(rifePaddedDimension(128)).toBe(128);
    expect(rifePaddedDimension(129)).toBe(256);
    expect(rifePaddedDimension(1080)).toBe(1152);
    expect(rifePaddedDimension(1920)).toBe(1920);
  });
});

describe('persistent inference worker protocol', () => {
  test('accepts page visibility updates without a job id', () => {
    expect(parseInferenceWorkerRequest({ type: 'visibility', hidden: true })).toEqual({
      type: 'visibility',
      hidden: true,
    });
    expect(parseInferenceWorkerRequest({ type: 'visibility', hidden: false })).toEqual({
      type: 'visibility',
      hidden: false,
    });
  });

  test('accepts a strictly shaped RIFE request', () => {
    const request = parseInferenceWorkerRequest({
      type: 'start-rife',
      jobId: 'job-1',
      frames: [{ width: 1, height: 1, buffer: new ArrayBuffer(4) }],
      durations: [100],
      fallbackDuration: 100,
      multiplier: 2,
    });
    expect(request.type).toBe('start-rife');
    if (request.type !== 'start-rife') throw new Error('Unexpected request type.');
    expect(request.frames).toHaveLength(1);
    expect(request.multiplier).toBe(2);
  });

  test('accepts cancellation without model payload', () => {
    expect(parseInferenceWorkerRequest({ type: 'cancel', jobId: 'job-2' })).toEqual({
      type: 'cancel',
      jobId: 'job-2',
    });
  });

  test('rejects malformed frame buffers and multipliers', () => {
    expect(() => parseInferenceWorkerRequest({
      type: 'start-rife',
      jobId: 'job-3',
      frames: [{ width: 2, height: 2, buffer: new ArrayBuffer(4) }],
      durations: [100],
      fallbackDuration: 100,
      multiplier: 3,
    })).toThrow();
  });
});

describe('DDIM browser scheduler', () => {
  test('uniform schedule is descending and ends at clean alpha', () => {
    const schedule = makeUniformDdimSchedule([1, 0.95, 0.8, 0.5, 0.2], 3);
    expect(schedule.map((step) => step.trainingTimestep)).toEqual([4, 2, 0]);
    expect(schedule.at(-1)?.alphaPrevious).toBe(1);
  });

  test('legacy LDM uniform schedule preserves the upstream +1 timestep convention', () => {
    const alphas = Array.from({ length: 1000 }, (_, index) => 1 - index / 2000);
    const schedule = makeLegacyLdmUniformDdimSchedule(alphas, 50);
    expect(schedule).toHaveLength(50);
    expect(schedule[0]?.trainingTimestep).toBe(981);
    expect(schedule.at(-1)?.trainingTimestep).toBe(1);
    expect(schedule.at(-1)?.alphaPrevious).toBe(alphas[0]);
  });

  test('deterministic DDIM step preserves shape and finite values', () => {
    const output = ddimStep(
      new Float32Array([0.3, -0.2, 0.8]),
      new Float32Array([0.1, -0.1, 0.2]),
      { alpha: 0.5, alphaPrevious: 0.8, eta: 0 },
    );
    expect(output).toHaveLength(3);
    for (const value of output) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('MoG browser v-prediction DDIM scheduler', () => {
  test('matches upstream uniform_trailing 50-of-1000 timestep selection', () => {
    const timesteps = Array.from(mogTimesteps(DEFAULT_MOG_SCHEDULE));
    expect(timesteps).toHaveLength(50);
    expect(timesteps[0]).toBe(19);
    expect(timesteps[1]).toBe(39);
    expect(timesteps[48]).toBe(979);
    expect(timesteps[49]).toBe(999);
  });

  test('matches upstream dynamic-rescale ramp and selected previous scale', () => {
    expect(mogDynamicScale(0)).toBeCloseTo(1, 10);
    expect(mogDynamicScale(399)).toBeCloseTo(0.7, 10);
    expect(mogDynamicScale(999)).toBeCloseTo(0.7, 10);
    const schedule = makeMogDdimSchedule(DEFAULT_MOG_SCHEDULE);
    const early = schedule.find((step) => step.trainingTimestep === 399);
    if (!early) throw new Error('MoG 399 timestep unexpectedly absent.');
    expect(early.dynamicScale).toBeCloseTo(0.7, 10);
    expect(early.dynamicScalePrevious).toBeGreaterThan(early.dynamicScale);
  });

  test('v prediction converts to x0 and epsilon with the standard identities', () => {
    const sample = new Float32Array([0.5, -0.25]);
    const velocity = new Float32Array([0.1, 0.3]);
    const alpha = 0.64;
    const prediction = mogVPrediction(sample, velocity, alpha);
    expect(prediction.predictedX0[0]).toBeCloseTo(0.34, 6);
    expect(prediction.epsilon[0]).toBeCloseTo(0.38, 6);
    expect(prediction.predictedX0[1]).toBeCloseTo(-0.38, 6);
    expect(prediction.epsilon[1]).toBeCloseTo(0.09, 6);
  });

  test('50-step eta-1 schedule remains finite after one stochastic step', () => {
    const schedule = makeMogDdimSchedule(DEFAULT_MOG_SCHEDULE);
    expect(schedule).toHaveLength(50);
    const latest = schedule.at(-1);
    if (!latest) throw new Error('MoG schedule unexpectedly empty.');
    const output = mogDdimStep(
      new Float32Array([0.2, -0.4, 0.7]),
      new Float32Array([0.1, 0.05, -0.2]),
      latest,
      new Float32Array([0.3, -0.1, 0.2]),
    );
    for (const value of output) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('ResShift browser scheduler', () => {
  const schedule = makeResShiftSchedule({
    timesteps: 20,
    kappa: 2,
    p: 0.3,
    minNoiseLevel: 0.04,
    etasEnd: 0.99,
  });

  test('matches upstream schedule endpoints', () => {
    expect(schedule.sqrtSumEta).toHaveLength(20);
    expect(schedule.sqrtSumEta[0]).toBeCloseTo(0.02, 10);
    expect(schedule.sqrtSumEta[19]).toBeCloseTo(0.99, 10);
    expect(schedule.sumPreviousEta[0]).toBe(0);
    expect(schedule.backwardStd[0]).toBe(0);
  });

  test('deterministic initial and reverse samples remain finite', () => {
    const endpoints = {
      first: new Float32Array([-1, 0, 1]),
      second: new Float32Array([1, 0.5, -1]),
      tau: 0.25,
    } as const;
    const zeroNoise = new Float32Array(3);
    const initial = initialiseResShiftSample(endpoints, schedule, 2, zeroNoise);
    const previous = resShiftReverseStep(
      initial,
      new Float32Array([0.2, 0.1, -0.3]),
      endpoints,
      schedule,
      19,
      zeroNoise,
    );
    expect(previous).toHaveLength(3);
    for (const value of previous) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('ProPainter browser mask primitives', () => {
  test('separable dilation matches a 3x3 max filter around one pixel', () => {
    const source = new Float32Array(25);
    source[12] = 1;
    const output = dilatePlane(source, 5, 5, 1);
    const selected: number[] = [];
    for (let index = 0; index < output.length; index += 1) {
      if ((output[index] ?? 0) > 0.5) selected.push(index);
    }
    expect(selected).toEqual([6, 7, 8, 11, 12, 13, 16, 17, 18]);
  });
});
