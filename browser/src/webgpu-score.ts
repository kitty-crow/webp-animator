import type { ScoreOptions } from './worker-pool.js';

const WORKGROUP_SIZE = 256;

const SHADER = /* wgsl */ `
struct Params {
  aw: u32,
  ah: u32,
  bw: u32,
  bh: u32,
  sigma: f32,
  alpha_threshold: u32,
  groups_per_candidate: u32,
  candidate_count: u32,
}

@group(0) @binding(0) var<storage, read> a_pixels: array<u32>;
@group(0) @binding(1) var<storage, read> b_pixels: array<u32>;
@group(0) @binding(2) var<storage, read> candidates: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scratch: array<f32, ${WORKGROUP_SIZE}>;

fn channel(pixel: u32, shift: u32) -> f32 {
  return f32((pixel >> shift) & 255u);
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let candidate_index = group_id.y;
  let pixel_index = group_id.x * ${WORKGROUP_SIZE}u + local_id.x;
  var score = 0.0;

  if (candidate_index < params.candidate_count && pixel_index < params.aw * params.ah) {
    let shift = candidates[candidate_index];
    let x = i32(pixel_index % params.aw);
    let y = i32(pixel_index / params.aw);
    let bx = x - shift.x;
    let by = y - shift.y;

    if (bx >= 0 && by >= 0 && bx < i32(params.bw) && by < i32(params.bh)) {
      let a = a_pixels[pixel_index];
      let b_index = u32(by) * params.bw + u32(bx);
      let b = b_pixels[b_index];
      let aa = u32(channel(a, 24u));
      let ba = u32(channel(b, 24u));
      if (aa > params.alpha_threshold || ba > params.alpha_threshold) {
        let dist = (
          abs(channel(a, 0u) - channel(b, 0u)) +
          abs(channel(a, 8u) - channel(b, 8u)) +
          abs(channel(a, 16u) - channel(b, 16u)) +
          abs(channel(a, 24u) - channel(b, 24u))
        ) * 0.25;
        let z = dist / params.sigma;
        score = exp(-(z * z));
      }
    }
  }

  scratch[local_id.x] = score;
  workgroupBarrier();

  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (local_id.x < stride) {
      scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (local_id.x == 0u && candidate_index < params.candidate_count) {
    partials[candidate_index * params.groups_per_candidate + group_id.x] = scratch[0];
  }
}
`;

function pad4(byteLength: number): number {
  return (byteLength + 3) & ~3;
}

function createBuffer(
  device: GPUDevice,
  bytes: Uint8Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(4, pad4(bytes.byteLength)),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

function imageBytes(image: ImageData): Uint8Array {
  const copy = new Uint8Array(image.data.byteLength);
  copy.set(image.data);
  return copy;
}

function int32Bytes(values: Int32Array): Uint8Array {
  return new Uint8Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
}

export class WebGpuScorer {
  static async create(adapter: GPUAdapter): Promise<WebGpuScorer> {
    const device = await adapter.requestDevice();
    return new WebGpuScorer(device);
  }

  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;

  private constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: SHADER });
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }

  async scoreCandidates(
    first: ImageData,
    second: ImageData,
    candidates: Int32Array,
    options: ScoreOptions,
  ): Promise<Float64Array> {
    const candidateCount = candidates.length / 2;
    if (candidateCount === 0) return new Float64Array();

    const firstBytes = imageBytes(first);
    const secondBytes = imageBytes(second);
    const storageLimit = Number(this.device.limits.maxStorageBufferBindingSize);
    const largestFrameBytes = Math.max(firstBytes.byteLength, secondBytes.byteLength);
    if (storageLimit > 0 && largestFrameBytes > storageLimit) {
      throw new Error(`Frame buffer ${largestFrameBytes} exceeds WebGPU storage limit ${storageLimit}.`);
    }

    const groupsPerCandidate = Math.ceil((first.width * first.height) / WORKGROUP_SIZE);
    const workgroupLimit = Number(this.device.limits.maxComputeWorkgroupsPerDimension);
    if (workgroupLimit > 0 && (groupsPerCandidate > workgroupLimit || candidateCount > workgroupLimit)) {
      throw new Error('Alignment workload exceeds this WebGPU adapter workgroup dimensions.');
    }

    const partialBytes = groupsPerCandidate * candidateCount * Float32Array.BYTES_PER_ELEMENT;
    const maxBuffer = Number(this.device.limits.maxBufferSize);
    if (maxBuffer > 0 && partialBytes > maxBuffer) {
      throw new Error(`Reduction buffer ${partialBytes} exceeds WebGPU maxBufferSize ${maxBuffer}.`);
    }
    if (storageLimit > 0 && partialBytes > storageLimit) {
      throw new Error(`Reduction buffer ${partialBytes} exceeds WebGPU storage limit ${storageLimit}.`);
    }

    const params = new ArrayBuffer(32);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = first.width;
    paramsU32[1] = first.height;
    paramsU32[2] = second.width;
    paramsU32[3] = second.height;
    paramsF32[4] = options.sigma;
    paramsU32[5] = options.alphaThreshold;
    paramsU32[6] = groupsPerCandidate;
    paramsU32[7] = candidateCount;

    const firstBuffer = createBuffer(this.device, firstBytes, GPUBufferUsage.STORAGE);
    const secondBuffer = createBuffer(this.device, secondBytes, GPUBufferUsage.STORAGE);
    const candidateBuffer = createBuffer(this.device, int32Bytes(candidates), GPUBufferUsage.STORAGE);
    const paramsBuffer = createBuffer(this.device, new Uint8Array(params), GPUBufferUsage.UNIFORM);
    const partialBuffer = this.device.createBuffer({
      size: Math.max(4, partialBytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      size: Math.max(4, partialBytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    try {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: firstBuffer } },
          { binding: 1, resource: { buffer: secondBuffer } },
          { binding: 2, resource: { buffer: candidateBuffer } },
          { binding: 3, resource: { buffer: partialBuffer } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(groupsPerCandidate, candidateCount, 1);
      pass.end();
      encoder.copyBufferToBuffer(partialBuffer, 0, readback, 0, partialBytes);
      this.device.queue.submit([encoder.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const partials = new Float32Array(mapped);
      const scores = new Float64Array(candidateCount);
      for (let candidate = 0; candidate < candidateCount; candidate += 1) {
        let total = 0;
        const base = candidate * groupsPerCandidate;
        for (let group = 0; group < groupsPerCandidate; group += 1) {
          const value = partials[base + group];
          if (value === undefined) throw new Error('WebGPU reduction buffer ended unexpectedly.');
          total += value;
        }
        scores[candidate] = total / Math.max(options.normaliser, 1);
      }
      return scores;
    } finally {
      if (readback.mapState === 'mapped') readback.unmap();
      firstBuffer.destroy();
      secondBuffer.destroy();
      candidateBuffer.destroy();
      paramsBuffer.destroy();
      partialBuffer.destroy();
      readback.destroy();
    }
  }
}
