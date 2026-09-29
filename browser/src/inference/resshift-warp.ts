import { appendMaskChannel, softmaxSplatCpu, type FlowTensor, type MetricTensor, type NchwTensor } from './softsplat.js';

export type SoftSplatFunction = (
  input: NchwTensor,
  flowXy: FlowTensor,
  metric: MetricTensor | null,
) => Promise<NchwTensor>;

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

function validateSameSpatial(first: NchwTensor, second: NchwTensor, label: string): void {
  if (first.batch !== second.batch || first.height !== second.height || first.width !== second.width) {
    throw new Error(`${label} spatial shapes differ.`);
  }
}

function validateFlow(flow: FlowTensor, reference: NchwTensor): void {
  if (flow.batch !== reference.batch || flow.height !== reference.height || flow.width !== reference.width) {
    throw new Error('ResShift flow spatial shape differs from its image tensor.');
  }
  if (flow.data.length !== flow.batch * 2 * flow.height * flow.width) throw new Error('ResShift flow data length is invalid.');
}

function srgbLinear(value: number): number {
  return value > 0.04045
    ? Math.pow((value + 0.055) / 1.055, 2.4)
    : value / 12.92;
}

function labPivot(value: number): number {
  return value > 0.008856
    ? Math.cbrt(value)
    : 7.787 * value + 16 / 116;
}

export function rgbToLabCpu(input: NchwTensor): NchwTensor {
  if (input.channels < 3) throw new Error('Lab conversion requires at least three RGB channels.');
  const output = new Float32Array(input.batch * 3 * input.height * input.width);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        const r = srgbLinear(input.data[tensorIndex(batch, 0, y, x, input.channels, input.height, input.width)] ?? 0);
        const g = srgbLinear(input.data[tensorIndex(batch, 1, y, x, input.channels, input.height, input.width)] ?? 0);
        const b = srgbLinear(input.data[tensorIndex(batch, 2, y, x, input.channels, input.height, input.width)] ?? 0);
        const xyzX = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.950456;
        const xyzY = 0.212671 * r + 0.71516 * g + 0.072169 * b;
        const xyzZ = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
        const fx = labPivot(xyzX);
        const fy = labPivot(xyzY);
        const fz = labPivot(xyzZ);
        output[tensorIndex(batch, 0, y, x, 3, input.height, input.width)] = 116 * fy - 16;
        output[tensorIndex(batch, 1, y, x, 3, input.height, input.width)] = 500 * (fx - fy);
        output[tensorIndex(batch, 2, y, x, 3, input.height, input.width)] = 200 * (fy - fz);
      }
    }
  }
  return { data: output, batch: input.batch, channels: 3, height: input.height, width: input.width };
}

function borderSample(
  input: NchwTensor,
  batch: number,
  channel: number,
  y: number,
  x: number,
): number {
  const clampedX = Math.max(0, Math.min(input.width - 1, x));
  const clampedY = Math.max(0, Math.min(input.height - 1, y));
  return input.data[tensorIndex(batch, channel, clampedY, clampedX, input.channels, input.height, input.width)] ?? 0;
}

/**
 * Matches the upstream HalfWarper.backward_wrapping grid convention.
 * Upstream flow channels are [dy, dx]. Its base grid uses linspace(-1, 1)
 * while grid_sample uses align_corners=false, so zero flow intentionally
 * preserves that slightly unusual coordinate convention.
 */
export function backwardWarpCpu(input: NchwTensor, flowDyDx: FlowTensor): NchwTensor {
  validateFlow(flowDyDx, input);
  const output = new Float32Array(input.data.length);
  const widthDenominator = Math.max(1, input.width - 1);
  const heightDenominator = Math.max(1, input.height - 1);

  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      const baseGridY = input.height === 1 ? 0 : -1 + 2 * y / heightDenominator;
      for (let x = 0; x < input.width; x += 1) {
        const baseGridX = input.width === 1 ? 0 : -1 + 2 * x / widthDenominator;
        const dy = flowDyDx.data[tensorIndex(batch, 0, y, x, 2, input.height, input.width)] ?? 0;
        const dx = flowDyDx.data[tensorIndex(batch, 1, y, x, 2, input.height, input.width)] ?? 0;
        const gridX = baseGridX + 2 * dx / input.width;
        const gridY = baseGridY + 2 * dy / input.height;
        const sourceX = ((gridX + 1) * input.width - 1) / 2;
        const sourceY = ((gridY + 1) * input.height - 1) / 2;
        const x0 = Math.floor(sourceX);
        const y0 = Math.floor(sourceY);
        const x1 = x0 + 1;
        const y1 = y0 + 1;
        const wx = sourceX - x0;
        const wy = sourceY - y0;

        for (let channel = 0; channel < input.channels; channel += 1) {
          const top = borderSample(input, batch, channel, y0, x0) * (1 - wx)
            + borderSample(input, batch, channel, y0, x1) * wx;
          const bottom = borderSample(input, batch, channel, y1, x0) * (1 - wx)
            + borderSample(input, batch, channel, y1, x1) * wx;
          output[tensorIndex(batch, channel, y, x, input.channels, input.height, input.width)] = top * (1 - wy) + bottom * wy;
        }
      }
    }
  }
  return { ...input, data: output };
}

export function zMetricCpu(
  first: NchwTensor,
  second: NchwTensor,
  flow0to1DyDx: FlowTensor,
  flow1to0DyDx: FlowTensor,
): { readonly z0to1: MetricTensor; readonly z1to0: MetricTensor } {
  validateSameSpatial(first, second, 'ResShift z-metric images');
  validateFlow(flow0to1DyDx, first);
  validateFlow(flow1to0DyDx, first);
  const lab0 = rgbToLabCpu(first);
  const lab1 = rgbToLabCpu(second);
  const lab0At1 = backwardWarpCpu(lab0, flow1to0DyDx);
  const lab1At0 = backwardWarpCpu(lab1, flow0to1DyDx);
  const z0to1 = new Float32Array(first.batch * first.height * first.width);
  const z1to0 = new Float32Array(z0to1.length);

  for (let batch = 0; batch < first.batch; batch += 1) {
    for (let y = 0; y < first.height; y += 1) {
      for (let x = 0; x < first.width; x += 1) {
        let sum0 = 0;
        let sum1 = 0;
        for (let channel = 0; channel < 3; channel += 1) {
          const index = tensorIndex(batch, channel, y, x, 3, first.height, first.width);
          const difference0 = (lab0.data[index] ?? 0) - (lab1At0.data[index] ?? 0);
          const difference1 = (lab1.data[index] ?? 0) - (lab0At1.data[index] ?? 0);
          sum0 += difference0 * difference0;
          sum1 += difference1 * difference1;
        }
        const destination = planeIndex(batch, y, x, first.height, first.width);
        z0to1[destination] = -0.1 * Math.sqrt(sum0);
        z1to0[destination] = -0.1 * Math.sqrt(sum1);
      }
    }
  }

  return {
    z0to1: { data: z0to1, batch: first.batch, height: first.height, width: first.width },
    z1to0: { data: z1to0, batch: first.batch, height: first.height, width: first.width },
  };
}

export function flowDyDxToXy(flow: FlowTensor): FlowTensor {
  const pixels = flow.height * flow.width;
  const output = new Float32Array(flow.data.length);
  for (let batch = 0; batch < flow.batch; batch += 1) {
    const dyStart = (batch * 2) * pixels;
    const dxStart = (batch * 2 + 1) * pixels;
    output.set(flow.data.subarray(dxStart, dxStart + pixels), dyStart);
    output.set(flow.data.subarray(dyStart, dyStart + pixels), dxStart);
  }
  return { ...flow, data: output };
}

export function scaleFlow(flow: FlowTensor, scaleDy: number, scaleDx: number): FlowTensor {
  const pixels = flow.height * flow.width;
  const output = new Float32Array(flow.data.length);
  for (let batch = 0; batch < flow.batch; batch += 1) {
    const dyStart = (batch * 2) * pixels;
    const dxStart = (batch * 2 + 1) * pixels;
    for (let index = 0; index < pixels; index += 1) {
      output[dyStart + index] = (flow.data[dyStart + index] ?? 0) * scaleDy;
      output[dxStart + index] = (flow.data[dxStart + index] ?? 0) * scaleDx;
    }
  }
  return { ...flow, data: output };
}

function extremumFilter(mask: MetricTensor, kernelSize: number, erosion: boolean): MetricTensor {
  if (!Number.isInteger(kernelSize) || kernelSize <= 0) throw new Error('Morphology kernel size must be a positive integer.');
  const radius = Math.floor(kernelSize / 2);
  const output = new Float32Array(mask.data.length);
  for (let batch = 0; batch < mask.batch; batch += 1) {
    for (let y = 0; y < mask.height; y += 1) {
      for (let x = 0; x < mask.width; x += 1) {
        let value = erosion ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
        let found = false;
        for (let ky = -radius; ky <= radius; ky += 1) {
          const sourceY = y + ky;
          if (sourceY < 0 || sourceY >= mask.height) continue;
          for (let kx = -radius; kx <= radius; kx += 1) {
            const sourceX = x + kx;
            if (sourceX < 0 || sourceX >= mask.width) continue;
            const sample = mask.data[planeIndex(batch, sourceY, sourceX, mask.height, mask.width)] ?? 0;
            value = erosion ? Math.min(value, sample) : Math.max(value, sample);
            found = true;
          }
        }
        output[planeIndex(batch, y, x, mask.height, mask.width)] = found ? value : 0;
      }
    }
  }
  return { ...mask, data: output };
}

export function morphOpenCpu(mask: MetricTensor, kernelSize: number): MetricTensor {
  if (kernelSize === 0) return mask;
  return extremumFilter(extremumFilter(mask, kernelSize, true), kernelSize, false);
}

function channelSlice(input: NchwTensor, channel: number): MetricTensor {
  if (channel < 0 || channel >= input.channels) throw new Error('Channel slice is out of range.');
  const output = new Float32Array(input.batch * input.height * input.width);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        output[planeIndex(batch, y, x, input.height, input.width)] = input.data[tensorIndex(batch, channel, y, x, input.channels, input.height, input.width)] ?? 0;
      }
    }
  }
  return { data: output, batch: input.batch, height: input.height, width: input.width };
}

function withoutLastChannel(input: NchwTensor): NchwTensor {
  if (input.channels <= 1) throw new Error('Cannot remove the only tensor channel.');
  const channels = input.channels - 1;
  const output = new Float32Array(input.batch * channels * input.height * input.width);
  const pixels = input.height * input.width;
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const source = (batch * input.channels + channel) * pixels;
      const destination = (batch * channels + channel) * pixels;
      output.set(input.data.subarray(source, source + pixels), destination);
    }
  }
  return { data: output, batch: input.batch, channels, height: input.height, width: input.width };
}

function blendByMask(primary: NchwTensor, secondary: NchwTensor, mask: MetricTensor): NchwTensor {
  validateSameSpatial(primary, secondary, 'ResShift warped tensors');
  if (primary.channels !== secondary.channels) throw new Error('ResShift warped channel counts differ.');
  const output = new Float32Array(primary.data.length);
  for (let batch = 0; batch < primary.batch; batch += 1) {
    for (let channel = 0; channel < primary.channels; channel += 1) {
      for (let y = 0; y < primary.height; y += 1) {
        for (let x = 0; x < primary.width; x += 1) {
          const index = tensorIndex(batch, channel, y, x, primary.channels, primary.height, primary.width);
          const weight = mask.data[planeIndex(batch, y, x, primary.height, primary.width)] ?? 0;
          output[index] = weight * (primary.data[index] ?? 0) + (1 - weight) * (secondary.data[index] ?? 0);
        }
      }
    }
  }
  return { ...primary, data: output };
}

function concatenateMask(input: NchwTensor, mask: MetricTensor): NchwTensor {
  const appended = appendMaskChannel(input);
  const pixels = input.height * input.width;
  for (let batch = 0; batch < input.batch; batch += 1) {
    const destination = (batch * appended.channels + input.channels) * pixels;
    const source = batch * pixels;
    appended.data.set(mask.data.subarray(source, source + pixels), destination);
  }
  return appended;
}

export async function halfWarp(
  first: NchwTensor,
  second: NchwTensor,
  flow0totDyDx: FlowTensor,
  flow1totDyDx: FlowTensor,
  z0to1: MetricTensor,
  z1to0: MetricTensor,
  splat: SoftSplatFunction,
  morphKernelSize = 5,
): Promise<{ readonly first: NchwTensor; readonly second: NchwTensor }> {
  validateSameSpatial(first, second, 'ResShift HalfWarper images');
  const firstWithMask = appendMaskChannel(first);
  const secondWithMask = appendMaskChannel(second);
  const forward0 = await splat(firstWithMask, flowDyDxToXy(flow0totDyDx), z0to1);
  const forward1 = await splat(secondWithMask, flowDyDxToXy(flow1totDyDx), z1to0);
  const wrapped0 = withoutLastChannel(forward0);
  const wrapped1 = withoutLastChannel(forward1);
  const mask0 = morphOpenCpu(channelSlice(forward0, forward0.channels - 1), morphKernelSize);
  const mask1 = morphOpenCpu(channelSlice(forward1, forward1.channels - 1), morphKernelSize);
  const base0 = blendByMask(wrapped0, wrapped1, mask0);
  const base1 = blendByMask(wrapped1, wrapped0, mask1);
  return {
    first: concatenateMask(base0, mask0),
    second: concatenateMask(base1, mask1),
  };
}

export const cpuSoftSplat: SoftSplatFunction = async (
  input: NchwTensor,
  flowXy: FlowTensor,
  metric: MetricTensor | null,
): Promise<NchwTensor> => softmaxSplatCpu(input, flowXy, metric);
