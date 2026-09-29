import { describe, expect, test } from 'bun:test';

import { euclideanDistanceTransformBinary } from '../src/inference/nedt.js';
import { appendMaskChannel, softmaxSplatCpu, type FlowTensor, type MetricTensor, type NchwTensor } from '../src/inference/softsplat.js';

function tensor(data: readonly number[], channels: number, height: number, width: number): NchwTensor {
  return {
    data: new Float32Array(data),
    batch: 1,
    channels,
    height,
    width,
  };
}

function zeroFlow(height: number, width: number): FlowTensor {
  return {
    data: new Float32Array(2 * height * width),
    batch: 1,
    height,
    width,
  };
}

function zeroMetric(height: number, width: number): MetricTensor {
  return {
    data: new Float32Array(height * width),
    batch: 1,
    height,
    width,
  };
}

describe('ResShift softmax splat browser parity', () => {
  test('zero flow is an identity transform within Float32 precision', () => {
    const input = tensor([1, 2, 3, 4], 1, 2, 2);
    const metric = zeroMetric(2, 2);
    metric.data.set([0.2, -0.4, 1.1, 0]);
    const output = softmaxSplatCpu(input, zeroFlow(2, 2), metric);
    const expected = [1, 2, 3, 4];
    for (let index = 0; index < expected.length; index += 1) {
      expect(output.data[index]).toBeCloseTo(expected[index] ?? 0, 6);
    }
  });

  test('integer x flow splats a source sample into the next pixel', () => {
    const input = tensor([4, 0, 0], 1, 1, 3);
    const flow = zeroFlow(1, 3);
    flow.data[0] = 1;
    const output = softmaxSplatCpu(input, flow, zeroMetric(1, 3));
    expect(Array.from(output.data)).toEqual([0, 2, 0]);
  });

  test('mask channel is appended per batch without altering source channels', () => {
    const input = tensor([0.25, 0.75], 1, 1, 2);
    const masked = appendMaskChannel(input);
    expect(masked.channels).toBe(2);
    expect(Array.from(masked.data)).toEqual([0.25, 0.75, 1, 1]);
  });
});

describe('ResShift NEDT browser parity', () => {
  test('binary EDT returns exact Euclidean distance to the selected centre pixel', () => {
    const binary = new Float32Array([
      0, 0, 0,
      0, 1, 0,
      0, 0, 0,
    ]);
    const output = euclideanDistanceTransformBinary(binary, 1, 3, 3);
    expect(output[4]).toBeCloseTo(0, 7);
    expect(output[1]).toBeCloseTo(1, 7);
    expect(output[0]).toBeCloseTo(Math.SQRT2, 7);
    expect(output[8]).toBeCloseTo(Math.SQRT2, 7);
  });

  test('all-selected binary plane has zero distance everywhere', () => {
    const output = euclideanDistanceTransformBinary(new Float32Array([1, 1, 1, 1]), 1, 2, 2);
    expect(Array.from(output)).toEqual([0, 0, 0, 0]);
  });
});
