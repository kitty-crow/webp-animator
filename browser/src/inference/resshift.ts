import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import { nedtCpu } from './nedt.js';
import type { BrowserModelAsset } from './catalog.js';
import { floatTensor, int64Tensor, loadOrtModel, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import { cpuSoftSplat, halfWarp, zMetricCpu, type SoftSplatFunction } from './resshift-warp.js';
import { initialiseResShiftSample, makeResShiftSchedule, resShiftReverseStep, type ResShiftScheduleConfig } from './resshift-scheduler.js';
import { WebGpuSoftSplat } from './softsplat-webgpu.js';
import type { FlowTensor, NchwTensor } from './softsplat.js';
import { concatenateChannels, multiplyFlow, resizeFlowBilinear, resizeMetricBilinear } from './tensor-resize.js';

export interface ResShiftAssetBundle {
  readonly flow: BrowserModelAsset;
  readonly extractor: BrowserModelAsset;
  readonly synthesis: BrowserModelAsset;
  readonly schedule?: ResShiftScheduleConfig;
}

export interface ResShiftProgress {
  readonly stage: 'flow' | 'features' | 'warp' | 'reverse';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface ResShiftResult {
  readonly tensor: NchwTensor;
  readonly providers: readonly ModelExecutionProvider[];
  readonly splatBackend: 'webgpu' | 'cpu';
}

const DEFAULT_SCHEDULE: ResShiftScheduleConfig = {
  timesteps: 15,
  kappa: 2,
  p: 0.3,
  minNoiseLevel: 0.04,
  etasEnd: 0.99,
};

function validateEndpoints(first: NchwTensor, second: NchwTensor, tau: number): void {
  if (first.batch !== 1 || second.batch !== 1) throw new Error('Browser ResShift currently processes one endpoint pair per job.');
  if (first.channels !== 3 || second.channels !== 3) throw new Error('ResShift endpoints must be RGB NCHW tensors.');
  if (first.height !== second.height || first.width !== second.width) throw new Error('ResShift endpoint sizes differ.');
  if (first.data.length !== second.data.length) throw new Error('ResShift endpoint data lengths differ.');
  if (!Number.isFinite(tau) || tau < 0 || tau > 1) throw new Error('ResShift interpolation ratio must be in [0, 1].');
}

function outputTensor(outputs: ort.InferenceSession.OnnxValueMapType, name: string): ort.Tensor {
  const value = outputs[name];
  if (!value) throw new Error(`ONNX output ${name} is missing.`);
  return value;
}

function nchwFromOrt(tensor: ort.Tensor, expectedChannels: number | null = null): NchwTensor {
  const dims = tensorDimensions(tensor);
  if (dims.length !== 4) throw new Error(`Expected NCHW ONNX output, received ${dims.length} dimensions.`);
  const batch = dims[0];
  const channels = dims[1];
  const height = dims[2];
  const width = dims[3];
  if (batch === undefined || channels === undefined || height === undefined || width === undefined) throw new Error('ONNX output shape is incomplete.');
  if (expectedChannels !== null && channels !== expectedChannels) throw new Error(`Expected ${expectedChannels} channels, received ${channels}.`);
  const source = tensorFloatData(tensor);
  const data = new Float32Array(source.length);
  data.set(source);
  return { data, batch, channels, height, width };
}

function flowFromOrt(tensor: ort.Tensor): FlowTensor {
  const value = nchwFromOrt(tensor, 2);
  return { data: value.data, batch: value.batch, height: value.height, width: value.width };
}

function firstChannels(input: NchwTensor, channels: number): NchwTensor {
  if (channels <= 0 || channels > input.channels) throw new Error('Requested ResShift channel slice is invalid.');
  const pixels = input.height * input.width;
  const data = new Float32Array(input.batch * channels * pixels);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const source = (batch * input.channels + channel) * pixels;
      const destination = (batch * channels + channel) * pixels;
      data.set(input.data.subarray(source, source + pixels), destination);
    }
  }
  return { data, batch: input.batch, channels, height: input.height, width: input.width };
}

function normalNoise(length: number): Float32Array {
  const uniforms = new Uint32Array(Math.max(2, Math.ceil(length / 2) * 2));
  const maximumChunk = 16_384;
  for (let offset = 0; offset < uniforms.length; offset += maximumChunk) {
    crypto.getRandomValues(uniforms.subarray(offset, Math.min(uniforms.length, offset + maximumChunk)));
  }
  const output = new Float32Array(length);
  let destination = 0;
  for (let index = 0; index < uniforms.length && destination < length; index += 2) {
    const u0 = ((uniforms[index] ?? 0) + 1) / 4_294_967_297;
    const u1 = ((uniforms[index + 1] ?? 0) + 1) / 4_294_967_297;
    const radius = Math.sqrt(-2 * Math.log(u0));
    const angle = 2 * Math.PI * u1;
    output[destination] = radius * Math.cos(angle);
    destination += 1;
    if (destination < length) {
      output[destination] = radius * Math.sin(angle);
      destination += 1;
    }
  }
  return output;
}

function tensorFeed(input: NchwTensor): ort.Tensor {
  return floatTensor(input.data, [input.batch, input.channels, input.height, input.width]);
}

async function releaseSession(session: ort.InferenceSession): Promise<void> {
  await session.release();
}

async function loadFeatures(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
  first: NchwTensor,
  second: NchwTensor,
): Promise<{ readonly first: readonly NchwTensor[]; readonly second: readonly NchwTensor[]; readonly provider: ModelExecutionProvider }> {
  const loaded = await loadOrtModel(asset, profile);
  try {
    const firstOutputs = await loaded.session.run({ image_with_nedt: tensorFeed(first) });
    const secondOutputs = await loaded.session.run({ image_with_nedt: tensorFeed(second) });
    const firstFeatures: NchwTensor[] = [];
    const secondFeatures: NchwTensor[] = [];
    for (let index = 0; index < 4; index += 1) {
      firstFeatures.push(nchwFromOrt(outputTensor(firstOutputs, `feature_${index}`)));
      secondFeatures.push(nchwFromOrt(outputTensor(secondOutputs, `feature_${index}`)));
    }
    return { first: firstFeatures, second: secondFeatures, provider: loaded.provider };
  } finally {
    await releaseSession(loaded.session);
  }
}

async function loadFlows(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
  first: NchwTensor,
  second: NchwTensor,
): Promise<{ readonly forward: FlowTensor; readonly backward: FlowTensor; readonly provider: ModelExecutionProvider }> {
  const loaded = await loadOrtModel(asset, profile);
  try {
    const outputs = await loaded.session.run({ first: tensorFeed(first), second: tensorFeed(second) });
    return {
      forward: flowFromOrt(outputTensor(outputs, 'flow0to1_dydx')),
      backward: flowFromOrt(outputTensor(outputs, 'flow1to0_dydx')),
      provider: loaded.provider,
    };
  } finally {
    await releaseSession(loaded.session);
  }
}

function createSplat(profile: HardwareProfile): Promise<{
  readonly splat: SoftSplatFunction;
  readonly backend: 'webgpu' | 'cpu';
  readonly close: () => void;
}> {
  return (async () => {
    const adapter = profile.webgpu.adapter;
    if (adapter) {
      try {
        const gpu = await WebGpuSoftSplat.create(adapter);
        let healthy = true;
        const splat: SoftSplatFunction = async (input, flow, metric) => {
          if (healthy) {
            try {
              return await gpu.splat(input, flow, metric);
            } catch (error: unknown) {
              healthy = false;
              console.warn('WebGPU softsplat failed; using CPU fallback.', error);
            }
          }
          return cpuSoftSplat(input, flow, metric);
        };
        return { splat, backend: 'webgpu' as const, close: () => gpu.close() };
      } catch (error: unknown) {
        console.warn('WebGPU softsplat initialisation failed; using CPU fallback.', error);
      }
    }
    return { splat: cpuSoftSplat, backend: 'cpu' as const, close: () => undefined };
  })();
}

async function prepareWarps(
  firstRgb: NchwTensor,
  secondRgb: NchwTensor,
  flow0to1: FlowTensor,
  flow1to0: FlowTensor,
  firstFeatures: readonly NchwTensor[],
  secondFeatures: readonly NchwTensor[],
  tau: number,
  splat: SoftSplatFunction,
): Promise<{ readonly first: readonly NchwTensor[]; readonly second: readonly NchwTensor[] }> {
  const firstNedt = concatenateChannels(firstRgb, nedtCpu(firstRgb));
  const secondNedt = concatenateChannels(secondRgb, nedtCpu(secondRgb));
  const metrics = zMetricCpu(firstNedt, secondNedt, flow0to1, flow1to0);
  const flow0tot = multiplyFlow(flow0to1, tau);
  const flow1tot = multiplyFlow(flow1to0, 1 - tau);
  const base = await halfWarp(firstNedt, secondNedt, flow0tot, flow1tot, metrics.z0to1, metrics.z1to0, splat);
  const warpedFirst: NchwTensor[] = [base.first];
  const warpedSecond: NchwTensor[] = [base.second];
  if (firstFeatures.length !== secondFeatures.length) throw new Error('ResShift feature pyramid lengths differ.');

  for (let index = 0; index < firstFeatures.length; index += 1) {
    const feature0 = firstFeatures[index];
    const feature1 = secondFeatures[index];
    if (!feature0 || !feature1) throw new Error(`ResShift feature pyramid level ${index} is missing.`);
    const f0 = resizeFlowBilinear(flow0tot, feature0.height, feature0.width);
    const f1 = resizeFlowBilinear(flow1tot, feature0.height, feature0.width);
    const z0 = resizeMetricBilinear(metrics.z0to1, feature0.height, feature0.width);
    const z1 = resizeMetricBilinear(metrics.z1to0, feature0.height, feature0.width);
    const warped = await halfWarp(feature0, feature1, f0, f1, z0, z1, splat);
    warpedFirst.push(warped.first);
    warpedSecond.push(warped.second);
  }
  return { first: warpedFirst, second: warpedSecond };
}

function synthesisFeeds(
  sample: NchwTensor,
  warpedFirst: readonly NchwTensor[],
  warpedSecond: readonly NchwTensor[],
  timestep: number,
): Record<string, ort.Tensor> {
  if (warpedFirst.length !== 5 || warpedSecond.length !== 5) throw new Error('ResShift synthesis requires five warp pyramid levels per endpoint.');
  const feeds: Record<string, ort.Tensor> = {
    sample: tensorFeed(sample),
    timestep: int64Tensor(new BigInt64Array([BigInt(timestep)]), [1]),
  };
  for (let index = 0; index < 5; index += 1) {
    const first = warpedFirst[index];
    const second = warpedSecond[index];
    if (!first || !second) throw new Error(`ResShift synthesis warp level ${index} is missing.`);
    feeds[`warp0_${index}`] = tensorFeed(first);
    feeds[`warp1_${index}`] = tensorFeed(second);
  }
  return feeds;
}

export class ResShiftOnnxAdapter {
  constructor(
    private readonly assets: ResShiftAssetBundle,
    private readonly profile: HardwareProfile,
  ) {}

  async interpolate(
    first: NchwTensor,
    second: NchwTensor,
    tau: number,
    onProgress: (progress: ResShiftProgress) => void = () => undefined,
  ): Promise<ResShiftResult> {
    validateEndpoints(first, second, tau);
    const providers: ModelExecutionProvider[] = [];

    onProgress({ stage: 'flow', current: 0, total: 1, provider: null });
    const flows = await loadFlows(this.assets.flow, this.profile, first, second);
    providers.push(flows.provider);
    onProgress({ stage: 'flow', current: 1, total: 1, provider: flows.provider });

    const firstWithNedt = concatenateChannels(first, nedtCpu(first));
    const secondWithNedt = concatenateChannels(second, nedtCpu(second));
    onProgress({ stage: 'features', current: 0, total: 2, provider: null });
    const features = await loadFeatures(this.assets.extractor, this.profile, firstWithNedt, secondWithNedt);
    providers.push(features.provider);
    onProgress({ stage: 'features', current: 2, total: 2, provider: features.provider });

    const splatRuntime = await createSplat(this.profile);
    let warps: { readonly first: readonly NchwTensor[]; readonly second: readonly NchwTensor[] };
    try {
      onProgress({ stage: 'warp', current: 0, total: features.first.length + 1, provider: null });
      warps = await prepareWarps(first, second, flows.forward, flows.backward, features.first, features.second, tau, splatRuntime.splat);
      onProgress({ stage: 'warp', current: features.first.length + 1, total: features.first.length + 1, provider: null });
    } finally {
      splatRuntime.close();
    }

    const config = this.assets.schedule ?? DEFAULT_SCHEDULE;
    const schedule = makeResShiftSchedule(config);
    const initialEndpoints = {
      first: firstChannels(warps.first[0] ?? (() => { throw new Error('ResShift first base warp is missing.'); })(), 3).data,
      second: firstChannels(warps.second[0] ?? (() => { throw new Error('ResShift second base warp is missing.'); })(), 3).data,
      tau,
    };
    const originalEndpoints = { first: first.data, second: second.data, tau };
    let sampleData = initialiseResShiftSample(initialEndpoints, schedule, config.kappa, normalNoise(first.data.length));

    const synthesis = await loadOrtModel(this.assets.synthesis, this.profile);
    providers.push(synthesis.provider);
    try {
      for (let timestep = config.timesteps - 1; timestep >= 0; timestep -= 1) {
        const completed = config.timesteps - 1 - timestep;
        onProgress({ stage: 'reverse', current: completed, total: config.timesteps, provider: synthesis.provider });
        const sample: NchwTensor = {
          data: sampleData,
          batch: first.batch,
          channels: first.channels,
          height: first.height,
          width: first.width,
        };
        const outputs = await synthesis.session.run(synthesisFeeds(sample, warps.first, warps.second, timestep));
        const predicted = nchwFromOrt(outputTensor(outputs, 'predicted_x0'), 3);
        sampleData = resShiftReverseStep(
          sampleData,
          predicted.data,
          originalEndpoints,
          schedule,
          timestep,
          normalNoise(sampleData.length),
        );
      }
      onProgress({ stage: 'reverse', current: config.timesteps, total: config.timesteps, provider: synthesis.provider });
    } finally {
      await releaseSession(synthesis.session);
    }

    return {
      tensor: { data: sampleData, batch: first.batch, channels: 3, height: first.height, width: first.width },
      providers,
      splatBackend: splatRuntime.backend,
    };
  }
}
