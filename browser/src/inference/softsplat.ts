export interface NchwTensor {
  readonly data: Float32Array;
  readonly batch: number;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
}

export interface FlowTensor {
  readonly data: Float32Array;
  readonly batch: number;
  readonly height: number;
  readonly width: number;
}

export interface MetricTensor {
  readonly data: Float32Array;
  readonly batch: number;
  readonly height: number;
  readonly width: number;
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer.`);
}

function validateTensor(tensor: NchwTensor): void {
  assertPositiveInteger(tensor.batch, 'Tensor batch');
  assertPositiveInteger(tensor.channels, 'Tensor channels');
  assertPositiveInteger(tensor.height, 'Tensor height');
  assertPositiveInteger(tensor.width, 'Tensor width');
  const expected = tensor.batch * tensor.channels * tensor.height * tensor.width;
  if (tensor.data.length !== expected) throw new Error(`Tensor data length mismatch: expected ${expected}, received ${tensor.data.length}.`);
}

function validateFlow(flow: FlowTensor, input: NchwTensor): void {
  if (flow.batch !== input.batch || flow.height !== input.height || flow.width !== input.width) {
    throw new Error('Softsplat flow shape does not match the input spatial shape.');
  }
  const expected = flow.batch * 2 * flow.height * flow.width;
  if (flow.data.length !== expected) throw new Error(`Softsplat flow length mismatch: expected ${expected}, received ${flow.data.length}.`);
}

function validateMetric(metric: MetricTensor | null, input: NchwTensor): void {
  if (metric === null) return;
  if (metric.batch !== input.batch || metric.height !== input.height || metric.width !== input.width) {
    throw new Error('Softsplat metric shape does not match the input spatial shape.');
  }
  const expected = metric.batch * metric.height * metric.width;
  if (metric.data.length !== expected) throw new Error(`Softsplat metric length mismatch: expected ${expected}, received ${metric.data.length}.`);
}

function nchwIndex(
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

function flowIndex(batch: number, channel: 0 | 1, y: number, x: number, height: number, width: number): number {
  return (((batch * 2 + channel) * height + y) * width) + x;
}

interface SplatTarget {
  readonly x: number;
  readonly y: number;
  readonly weight: number;
}

function splatTargets(outputX: number, outputY: number): readonly SplatTarget[] {
  const northwestX = Math.floor(outputX);
  const northwestY = Math.floor(outputY);
  const northeastX = northwestX + 1;
  const northeastY = northwestY;
  const southwestX = northwestX;
  const southwestY = northwestY + 1;
  const southeastX = northwestX + 1;
  const southeastY = northwestY + 1;
  return [
    {
      x: northwestX,
      y: northwestY,
      weight: (southeastX - outputX) * (southeastY - outputY),
    },
    {
      x: northeastX,
      y: northeastY,
      weight: (outputX - southwestX) * (southwestY - outputY),
    },
    {
      x: southwestX,
      y: southwestY,
      weight: (northeastX - outputX) * (outputY - northeastY),
    },
    {
      x: southeastX,
      y: southeastY,
      weight: (outputX - northwestX) * (outputY - northwestY),
    },
  ];
}

export function softmaxSplatCpu(
  input: NchwTensor,
  flow: FlowTensor,
  metric: MetricTensor | null,
): NchwTensor {
  validateTensor(input);
  validateFlow(flow, input);
  validateMetric(metric, input);

  const output = new Float32Array(input.data.length);
  const normaliser = new Float32Array(input.batch * input.height * input.width);

  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        const flowX = flow.data[flowIndex(batch, 0, y, x, input.height, input.width)] ?? 0;
        const flowY = flow.data[flowIndex(batch, 1, y, x, input.height, input.width)] ?? 0;
        const metricValue = metric === null
          ? 0
          : metric.data[planeIndex(batch, y, x, input.height, input.width)] ?? 0;
        const metricWeight = Math.exp(metricValue);
        const targets = splatTargets(x + flowX, y + flowY);

        for (const target of targets) {
          if (
            target.weight === 0 ||
            target.x < 0 || target.x >= input.width ||
            target.y < 0 || target.y >= input.height
          ) continue;

          const weightedMetric = target.weight * metricWeight;
          const normIndex = planeIndex(batch, target.y, target.x, input.height, input.width);
          normaliser[normIndex] = (normaliser[normIndex] ?? 0) + weightedMetric;
          for (let channel = 0; channel < input.channels; channel += 1) {
            const sourceIndex = nchwIndex(batch, channel, y, x, input.channels, input.height, input.width);
            const destinationIndex = nchwIndex(batch, channel, target.y, target.x, input.channels, input.height, input.width);
            output[destinationIndex] = (output[destinationIndex] ?? 0) + (input.data[sourceIndex] ?? 0) * weightedMetric;
          }
        }
      }
    }
  }

  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        const norm = normaliser[planeIndex(batch, y, x, input.height, input.width)] ?? 0;
        const denominator = norm === 0 ? 1 : norm;
        for (let channel = 0; channel < input.channels; channel += 1) {
          const index = nchwIndex(batch, channel, y, x, input.channels, input.height, input.width);
          output[index] = (output[index] ?? 0) / denominator;
        }
      }
    }
  }

  return {
    data: output,
    batch: input.batch,
    channels: input.channels,
    height: input.height,
    width: input.width,
  };
}

export function appendMaskChannel(input: NchwTensor): NchwTensor {
  validateTensor(input);
  const pixelsPerPlane = input.height * input.width;
  const outputChannels = input.channels + 1;
  const output = new Float32Array(input.batch * outputChannels * pixelsPerPlane);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let channel = 0; channel < input.channels; channel += 1) {
      const sourceStart = (batch * input.channels + channel) * pixelsPerPlane;
      const destinationStart = (batch * outputChannels + channel) * pixelsPerPlane;
      output.set(input.data.subarray(sourceStart, sourceStart + pixelsPerPlane), destinationStart);
    }
    const maskStart = (batch * outputChannels + input.channels) * pixelsPerPlane;
    output.fill(1, maskStart, maskStart + pixelsPerPlane);
  }
  return {
    data: output,
    batch: input.batch,
    channels: outputChannels,
    height: input.height,
    width: input.width,
  };
}
