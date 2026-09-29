import type { FlowTensor, MetricTensor, NchwTensor } from './softsplat.js';

const WORKGROUP_SIZE = 256;

const SPLAT_SHADER = `
struct Params {
  batch: u32,
  channels: u32,
  height: u32,
  width: u32,
  pixels_per_batch: u32,
  total_source_pixels: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> flow_data: array<f32>;
@group(0) @binding(2) var<storage, read> metric_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> normaliser: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

fn atomic_add_f32(target: ptr<storage, atomic<u32>, read_write>, value: f32) {
  if (value == 0.0) { return; }
  var previous = atomicLoad(target);
  loop {
    let previous_float = bitcast<f32>(previous);
    let next = bitcast<u32>(previous_float + value);
    let exchanged = atomicCompareExchangeWeak(target, previous, next);
    if (exchanged.exchanged) { break; }
    previous = exchanged.old_value;
  }
}

fn splat_one(
  batch_index: u32,
  source_spatial: u32,
  target_x: i32,
  target_y: i32,
  bilinear_weight: f32,
  metric_weight: f32,
) {
  if (bilinear_weight == 0.0) { return; }
  if (target_x < 0 || target_y < 0 || target_x >= i32(params.width) || target_y >= i32(params.height)) { return; }
  let target_spatial = u32(target_y) * params.width + u32(target_x);
  let norm_index = batch_index * params.pixels_per_batch + target_spatial;
  let weighted_metric = bilinear_weight * metric_weight;
  atomic_add_f32(&normaliser[norm_index], weighted_metric);

  for (var channel: u32 = 0u; channel < params.channels; channel = channel + 1u) {
    let source_index = (batch_index * params.channels + channel) * params.pixels_per_batch + source_spatial;
    let destination_index = (batch_index * params.channels + channel) * params.pixels_per_batch + target_spatial;
    atomic_add_f32(&accum[destination_index], input_data[source_index] * weighted_metric);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let flat = global_id.x;
  if (flat >= params.total_source_pixels) { return; }
  let batch_index = flat / params.pixels_per_batch;
  let source_spatial = flat % params.pixels_per_batch;
  let y = source_spatial / params.width;
  let x = source_spatial % params.width;
  let flow_x_index = (batch_index * 2u) * params.pixels_per_batch + source_spatial;
  let flow_y_index = (batch_index * 2u + 1u) * params.pixels_per_batch + source_spatial;
  let output_x = f32(x) + flow_data[flow_x_index];
  let output_y = f32(y) + flow_data[flow_y_index];
  let northwest_x = i32(floor(output_x));
  let northwest_y = i32(floor(output_y));
  let northeast_x = northwest_x + 1;
  let northeast_y = northwest_y;
  let southwest_x = northwest_x;
  let southwest_y = northwest_y + 1;
  let southeast_x = northwest_x + 1;
  let southeast_y = northwest_y + 1;
  let metric_weight = exp(metric_data[batch_index * params.pixels_per_batch + source_spatial]);

  splat_one(batch_index, source_spatial, northwest_x, northwest_y,
    (f32(southeast_x) - output_x) * (f32(southeast_y) - output_y), metric_weight);
  splat_one(batch_index, source_spatial, northeast_x, northeast_y,
    (output_x - f32(southwest_x)) * (f32(southwest_y) - output_y), metric_weight);
  splat_one(batch_index, source_spatial, southwest_x, southwest_y,
    (f32(northeast_x) - output_x) * (output_y - f32(northeast_y)), metric_weight);
  splat_one(batch_index, source_spatial, southeast_x, southeast_y,
    (output_x - f32(northwest_x)) * (output_y - f32(northwest_y)), metric_weight);
}
`;

const NORMALISE_SHADER = `
struct Params {
  batch: u32,
  channels: u32,
  height: u32,
  width: u32,
  pixels_per_batch: u32,
  total_source_pixels: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> normaliser: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  let total_values = params.batch * params.channels * params.pixels_per_batch;
  if (index >= total_values) { return; }
  let batch_channel = index / params.pixels_per_batch;
  let batch_index = batch_channel / params.channels;
  let spatial = index % params.pixels_per_batch;
  let norm_bits = atomicLoad(&normaliser[batch_index * params.pixels_per_batch + spatial]);
  let norm = bitcast<f32>(norm_bits);
  let denominator = select(norm, 1.0, norm == 0.0);
  let value = bitcast<f32>(atomicLoad(&accum[index]));
  output_data[index] = value / denominator;
}
`;

function validate(input: NchwTensor, flow: FlowTensor, metric: MetricTensor | null): void {
  const expectedInput = input.batch * input.channels * input.height * input.width;
  if (input.data.length !== expectedInput) throw new Error('WebGPU softsplat input length mismatch.');
  if (flow.batch !== input.batch || flow.height !== input.height || flow.width !== input.width) throw new Error('WebGPU softsplat flow shape mismatch.');
  if (flow.data.length !== input.batch * 2 * input.height * input.width) throw new Error('WebGPU softsplat flow length mismatch.');
  if (metric !== null) {
    if (metric.batch !== input.batch || metric.height !== input.height || metric.width !== input.width) throw new Error('WebGPU softsplat metric shape mismatch.');
    if (metric.data.length !== input.batch * input.height * input.width) throw new Error('WebGPU softsplat metric length mismatch.');
  }
}

function aligned4(bytes: number): number {
  return Math.max(4, (bytes + 3) & ~3);
}

function upload(device: GPUDevice, data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({
    size: aligned4(data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

function zeroBuffer(device: GPUDevice, bytes: number, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({
    size: aligned4(bytes),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).fill(0);
  buffer.unmap();
  return buffer;
}

export class WebGpuSoftSplat {
  static async create(adapter: GPUAdapter): Promise<WebGpuSoftSplat> {
    return new WebGpuSoftSplat(await adapter.requestDevice());
  }

  private readonly splatPipeline: GPUComputePipeline;
  private readonly normalisePipeline: GPUComputePipeline;

  private constructor(private readonly device: GPUDevice) {
    this.splatPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: SPLAT_SHADER }),
        entryPoint: 'main',
      },
    });
    this.normalisePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: NORMALISE_SHADER }),
        entryPoint: 'main',
      },
    });
  }

  async splat(input: NchwTensor, flow: FlowTensor, metric: MetricTensor | null): Promise<NchwTensor> {
    validate(input, flow, metric);
    const pixelsPerBatch = input.height * input.width;
    const totalSourcePixels = input.batch * pixelsPerBatch;
    const outputValues = input.data.length;
    const metricData = metric?.data ?? new Float32Array(totalSourcePixels);
    const params = new Uint32Array(8);
    params[0] = input.batch;
    params[1] = input.channels;
    params[2] = input.height;
    params[3] = input.width;
    params[4] = pixelsPerBatch;
    params[5] = totalSourcePixels;

    const storageLimit = Number(this.device.limits.maxStorageBufferBindingSize || 0);
    const largestBuffer = Math.max(input.data.byteLength, flow.data.byteLength, metricData.byteLength, outputValues * 4);
    if (storageLimit > 0 && largestBuffer > storageLimit) throw new Error('Softsplat workload exceeds this WebGPU storage-buffer limit.');

    const inputBuffer = upload(this.device, input.data, GPUBufferUsage.STORAGE);
    const flowBuffer = upload(this.device, flow.data, GPUBufferUsage.STORAGE);
    const metricBuffer = upload(this.device, metricData, GPUBufferUsage.STORAGE);
    const accumBuffer = zeroBuffer(this.device, outputValues * 4, GPUBufferUsage.STORAGE);
    const normaliserBuffer = zeroBuffer(this.device, totalSourcePixels * 4, GPUBufferUsage.STORAGE);
    const paramsBuffer = upload(this.device, params, GPUBufferUsage.UNIFORM);
    const outputBuffer = zeroBuffer(this.device, outputValues * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = zeroBuffer(this.device, outputValues * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    try {
      const splatGroup = this.device.createBindGroup({
        layout: this.splatPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: flowBuffer } },
          { binding: 2, resource: { buffer: metricBuffer } },
          { binding: 3, resource: { buffer: accumBuffer } },
          { binding: 4, resource: { buffer: normaliserBuffer } },
          { binding: 5, resource: { buffer: paramsBuffer } },
        ],
      });
      const normaliseGroup = this.device.createBindGroup({
        layout: this.normalisePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: accumBuffer } },
          { binding: 1, resource: { buffer: normaliserBuffer } },
          { binding: 2, resource: { buffer: outputBuffer } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const splatPass = encoder.beginComputePass();
      splatPass.setPipeline(this.splatPipeline);
      splatPass.setBindGroup(0, splatGroup);
      splatPass.dispatchWorkgroups(Math.ceil(totalSourcePixels / WORKGROUP_SIZE));
      splatPass.end();

      const normalisePass = encoder.beginComputePass();
      normalisePass.setPipeline(this.normalisePipeline);
      normalisePass.setBindGroup(0, normaliseGroup);
      normalisePass.dispatchWorkgroups(Math.ceil(outputValues / WORKGROUP_SIZE));
      normalisePass.end();
      encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outputValues * 4);
      this.device.queue.submit([encoder.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Float32Array(readback.getMappedRange());
      const data = new Float32Array(mapped.length);
      data.set(mapped);
      return {
        data,
        batch: input.batch,
        channels: input.channels,
        height: input.height,
        width: input.width,
      };
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      inputBuffer.destroy();
      flowBuffer.destroy();
      metricBuffer.destroy();
      accumBuffer.destroy();
      normaliserBuffer.destroy();
      paramsBuffer.destroy();
      outputBuffer.destroy();
      readback.destroy();
    }
  }

  close(): void {
    this.device.destroy();
  }
}
