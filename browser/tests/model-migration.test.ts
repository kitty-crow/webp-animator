import { describe, expect, test } from 'bun:test';

import { BROWSER_MODEL_CATALOG, browserModelDefinition } from '../src/inference/catalog.js';
import { ddimStep, makeUniformDdimSchedule } from '../src/inference/ddim.js';
import { dilatePlane } from '../src/inference/propainter-mask.js';
import { initialiseResShiftSample, makeResShiftSchedule, resShiftReverseStep } from '../src/inference/resshift-scheduler.js';

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

describe('DDIM browser scheduler', () => {
  test('uniform schedule is descending and ends at clean alpha', () => {
    const schedule = makeUniformDdimSchedule([1, 0.95, 0.8, 0.5, 0.2], 3);
    expect(schedule.map((step) => step.trainingTimestep)).toEqual([4, 2, 0]);
    expect(schedule.at(-1)?.alphaPrevious).toBe(1);
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
