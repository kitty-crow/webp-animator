import type { FlowTensor, MetricTensor, NchwTensor } from './softsplat.js';

function sourceCoordinate(destination: number, inputSize: number, outputSize: number): number {
  return ((destination + 0.5) * inputSize / outputSize) - 0.5;
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

function sample(
  input: NchwTensor,
  batch: number,
  channel: number,
  y: number,
  x: number,
): number {
  const sourceX = Math.max(0, Math.min(input.width - 1, x));
  const sourceY = Math.max(0, Math.min(input.height - 1, y));
  return input.data[tensorIndex(batch, channel, sourceY, sourceX, input.channels, input.height, input.width)] ?? 0;
}

export function resizeNchwBilinear(input: NchwTensor, height: number, width: number): NchwTensor {
  if (!Number.isInteger(height) || height <= 0 || !Number.isInteger(width) || width <= 0) {
    throw new Error('Tensor resize dimensions must be positive integers.');
  }
  if (height === input.height && width === input.width) return input;
  const output = new Float32Array(input.batch * input.channels * height * width);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let channel = 0; channel < input.channels; channel += 1) {
      for (let y = 0; y < height; y += 1) {
        const sourceY = sourceCoordinate(y, input.height, height);
        const y0 = Math.floor(sourceY);
        const y1 = y0 + 1;
        const wy = sourceY - y0;
        for (let x = 0; x < width; x += 1) {
          const sourceX = sourceCoordinate(x, input.width, width);
          const x0 = Math.floor(sourceX);
          const x1 = x0 + 1;
          const wx = sourceX - x0;
          const top = sample(input, batch, channel, y0, x0) * (1 - wx)
            + sample(input, batch, channel, y0, x1) * wx;
          const bottom = sample(input, batch, channel, y1, x0) * (1 - wx)
            + sample(input, batch, channel, y1, x1) * wx;
          output[tensorIndex(batch, channel, y, x, input.channels, height, width)] = top * (1 - wy) + bottom * wy;
        }
      }
    }
  }
  return { data: output, batch: input.batch, channels: input.channels, height, width };
}

export function resizeFlowBilinear(flow: FlowTensor, height: number, width: number): FlowTensor {
  const resized = resizeNchwBilinear({
    data: flow.data,
    batch: flow.batch,
    channels: 2,
    height: flow.height,
    width: flow.width,
  }, height, width);
  return { data: resized.data, batch: resized.batch, height: resized.height, width: resized.width };
}

export function resizeMetricBilinear(metric: MetricTensor, height: number, width: number): MetricTensor {
  const resized = resizeNchwBilinear({
    data: metric.data,
    batch: metric.batch,
    channels: 1,
    height: metric.height,
    width: metric.width,
  }, height, width);
  return { data: resized.data, batch: resized.batch, height: resized.height, width: resized.width };
}

export function multiplyFlow(flow: FlowTensor, scalar: number): FlowTensor {
  if (!Number.isFinite(scalar)) throw new Error('Flow multiplier must be finite.');
  const data = new Float32Array(flow.data.length);
  for (let index = 0; index < data.length; index += 1) data[index] = (flow.data[index] ?? 0) * scalar;
  return { ...flow, data };
}

export function concatenateChannels(first: NchwTensor, second: NchwTensor): NchwTensor {
  if (first.batch !== second.batch || first.height !== second.height || first.width !== second.width) {
    throw new Error('Channel concatenation requires matching batch and spatial shapes.');
  }
  const pixels = first.height * first.width;
  const channels = first.channels + second.channels;
  const data = new Float32Array(first.batch * channels * pixels);
  for (let batch = 0; batch < first.batch; batch += 1) {
    for (let channel = 0; channel < first.channels; channel += 1) {
      const source = (batch * first.channels + channel) * pixels;
      const destination = (batch * channels + channel) * pixels;
      data.set(first.data.subarray(source, source + pixels), destination);
    }
    for (let channel = 0; channel < second.channels; channel += 1) {
      const source = (batch * second.channels + channel) * pixels;
      const destination = (batch * channels + first.channels + channel) * pixels;
      data.set(second.data.subarray(source, source + pixels), destination);
    }
  }
  return { data, batch: first.batch, channels, height: first.height, width: first.width };
}
