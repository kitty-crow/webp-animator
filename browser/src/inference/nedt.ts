import type { NchwTensor } from './softsplat.js';

function validateRgbTensor(input: NchwTensor): void {
  if (!Number.isInteger(input.batch) || input.batch <= 0) throw new Error('NEDT batch must be positive.');
  if (input.channels !== 1 && input.channels !== 3 && input.channels !== 4) throw new Error('NEDT accepts 1, 3 or 4 channels.');
  if (!Number.isInteger(input.height) || input.height <= 0 || !Number.isInteger(input.width) || input.width <= 0) {
    throw new Error('NEDT dimensions must be positive integers.');
  }
  const expected = input.batch * input.channels * input.height * input.width;
  if (input.data.length !== expected) throw new Error(`NEDT input length mismatch: expected ${expected}, received ${input.data.length}.`);
}

function tensorIndex(
  batch: number,
  channel: number,
  y: number,
  x: number,
  channels: number,
  height: number,
  width: number,
): number {
  return (((batch * channels + channel) * height + y) * width) + x;
}

function planeIndex(batch: number, y: number, x: number, height: number, width: number): number {
  return ((batch * height + y) * width) + x;
}

function grayscale(input: NchwTensor): Float32Array {
  const output = new Float32Array(input.batch * input.height * input.width);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        const destination = planeIndex(batch, y, x, input.height, input.width);
        if (input.channels === 1) {
          output[destination] = input.data[tensorIndex(batch, 0, y, x, 1, input.height, input.width)] ?? 0;
          continue;
        }
        const r = input.data[tensorIndex(batch, 0, y, x, input.channels, input.height, input.width)] ?? 0;
        const g = input.data[tensorIndex(batch, 1, y, x, input.channels, input.height, input.width)] ?? 0;
        const b = input.data[tensorIndex(batch, 2, y, x, input.channels, input.height, input.width)] ?? 0;
        output[destination] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
    }
  }
  return output;
}

function gaussianKernel(sigma: number, kernelFactor: number): Float64Array {
  if (!Number.isFinite(sigma) || sigma <= 0) throw new Error('NEDT Gaussian sigma must be positive.');
  const radius = Math.max(Math.floor(sigma * kernelFactor), 1);
  const kernel = new Float64Array(radius * 2 + 1);
  let sum = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * sigma * sigma));
    kernel[offset + radius] = value;
    sum += value;
  }
  if (!(sum > 0)) throw new Error('NEDT Gaussian kernel normalisation failed.');
  for (let index = 0; index < kernel.length; index += 1) kernel[index] = (kernel[index] ?? 0) / sum;
  return kernel;
}

function clamped(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function gaussianBlur(
  input: Float32Array,
  batch: number,
  height: number,
  width: number,
  sigma: number,
  kernelFactor: number,
): Float32Array {
  const kernel = gaussianKernel(sigma, kernelFactor);
  const radius = (kernel.length - 1) / 2;
  const horizontal = new Float32Array(input.length);
  const output = new Float32Array(input.length);

  for (let b = 0; b < batch; b += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        for (let kernelIndex = 0; kernelIndex < kernel.length; kernelIndex += 1) {
          const sourceX = clamped(x + kernelIndex - radius, 0, width - 1);
          sum += (input[planeIndex(b, y, sourceX, height, width)] ?? 0) * (kernel[kernelIndex] ?? 0);
        }
        horizontal[planeIndex(b, y, x, height, width)] = sum;
      }
    }
  }

  for (let b = 0; b < batch; b += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        for (let kernelIndex = 0; kernelIndex < kernel.length; kernelIndex += 1) {
          const sourceY = clamped(y + kernelIndex - radius, 0, height - 1);
          sum += (horizontal[planeIndex(b, sourceY, x, height, width)] ?? 0) * (kernel[kernelIndex] ?? 0);
        }
        output[planeIndex(b, y, x, height, width)] = sum;
      }
    }
  }
  return output;
}

function squaredDistanceTransform1d(source: Float64Array): Float64Array {
  const length = source.length;
  if (length === 0) return new Float64Array();
  const sites = new Int32Array(length);
  const boundaries = new Float64Array(length + 1);
  const output = new Float64Array(length);
  let k = 0;
  sites[0] = 0;
  boundaries[0] = Number.NEGATIVE_INFINITY;
  boundaries[1] = Number.POSITIVE_INFINITY;

  for (let q = 1; q < length; q += 1) {
    let site = sites[k] ?? 0;
    let separation = ((source[q] ?? 0) + q * q - (source[site] ?? 0) - site * site) / (2 * (q - site));
    while (k > 0 && separation <= (boundaries[k] ?? Number.NEGATIVE_INFINITY)) {
      k -= 1;
      site = sites[k] ?? 0;
      separation = ((source[q] ?? 0) + q * q - (source[site] ?? 0) - site * site) / (2 * (q - site));
    }
    k += 1;
    sites[k] = q;
    boundaries[k] = separation;
    boundaries[k + 1] = Number.POSITIVE_INFINITY;
  }

  k = 0;
  for (let q = 0; q < length; q += 1) {
    while ((boundaries[k + 1] ?? Number.POSITIVE_INFINITY) < q) k += 1;
    const site = sites[k] ?? 0;
    const delta = q - site;
    output[q] = delta * delta + (source[site] ?? 0);
  }
  return output;
}

export function euclideanDistanceTransformBinary(
  binary: Float32Array,
  batch: number,
  height: number,
  width: number,
): Float32Array {
  const expected = batch * height * width;
  if (binary.length !== expected) throw new Error('NEDT binary plane length mismatch.');
  const diameterSquared = height * height + width * width;
  const rows = new Float64Array(expected);
  const sourceLine = new Float64Array(Math.max(width, height));

  for (let b = 0; b < batch; b += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const value = binary[planeIndex(b, y, x, height, width)] ?? 0;
        sourceLine[x] = (1 - value) * diameterSquared;
      }
      const transformed = squaredDistanceTransform1d(sourceLine.subarray(0, width));
      for (let x = 0; x < width; x += 1) rows[planeIndex(b, y, x, height, width)] = transformed[x] ?? 0;
    }
  }

  const output = new Float32Array(expected);
  for (let b = 0; b < batch; b += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let y = 0; y < height; y += 1) sourceLine[y] = rows[planeIndex(b, y, x, height, width)] ?? diameterSquared;
      const transformed = squaredDistanceTransform1d(sourceLine.subarray(0, height));
      for (let y = 0; y < height; y += 1) output[planeIndex(b, y, x, height, width)] = Math.sqrt(transformed[y] ?? 0);
    }
  }
  return output;
}

export interface NedtOptions {
  readonly t?: number;
  readonly sigmaFactor?: number;
  readonly k?: number;
  readonly epsilon?: number;
  readonly kernelFactor?: number;
  readonly expFactor?: number;
}

export function nedtCpu(input: NchwTensor, options: NedtOptions = {}): NchwTensor {
  validateRgbTensor(input);
  const t = options.t ?? 2;
  const sigmaFactor = options.sigmaFactor ?? 1 / 540;
  const k = options.k ?? 1.6;
  const epsilon = options.epsilon ?? 0.01;
  const kernelFactor = options.kernelFactor ?? 4;
  const expFactor = options.expFactor ?? 540 / 15;
  const sigma = input.height * sigmaFactor;
  const gray = grayscale(input);
  const gaussian0 = gaussianBlur(gray, input.batch, input.height, input.width, sigma, kernelFactor);
  const gaussian1 = gaussianBlur(gray, input.batch, input.height, input.width, sigma * k, kernelFactor);
  const binary = new Float32Array(gray.length);
  for (let index = 0; index < binary.length; index += 1) {
    const dog = 0.5 + t * ((gaussian1[index] ?? 0) - (gaussian0[index] ?? 0)) - epsilon;
    binary[index] = dog > 0.5 ? 1 : 0;
  }
  const edt = euclideanDistanceTransformBinary(binary, input.batch, input.height, input.width);
  const maximumSide = Math.max(input.height, input.width);
  const output = new Float32Array(edt.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = 1 - Math.exp(-(edt[index] ?? 0) * expFactor / maximumSide);
  }
  return {
    data: output,
    batch: input.batch,
    channels: 1,
    height: input.height,
    width: input.width,
  };
}
