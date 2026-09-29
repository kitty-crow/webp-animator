import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import {
  makeMogConcatLatent,
  makeMogEndpointVideo,
  mogNormalNoise,
  replaceMogFrames,
  sampleMogPosterior,
  selectMogFrames,
  warpMogLatentGuidance,
  type MogMotion,
  type NcthwTensor,
} from './mog-latent.js';
import { DEFAULT_MOG_SCHEDULE, makeMogDdimSchedule, mogDdimStep, type MogScheduleConfig } from './mog-scheduler.js';
import { floatTensor, int64Tensor, loadOrtModel, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import type { NchwTensor } from './softsplat.js';

export interface MogAssetBundle {
  readonly motion: BrowserModelAsset;
  readonly vaeEncoder: BrowserModelAsset;
  readonly imageCondition: BrowserModelAsset;
  readonly denoiser: BrowserModelAsset;
  readonly vaeDecoder: BrowserModelAsset;
  readonly frames?: number;
  readonly posteriorScaleFactor?: number;
  readonly referenceHiddenCount: number;
  readonly fps?: number;
  readonly schedule?: MogScheduleConfig;
}

export interface MogProgress {
  readonly stage: 'motion' | 'encode' | 'condition' | 'denoise' | 'decode';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface MogResult {
  readonly video: NcthwTensor;
  readonly providers: readonly ModelExecutionProvider[];
}

interface DenseFloatTensor {
  readonly data: Float32Array;
  readonly dims: readonly number[];
}

function outputTensor(outputs: ort.InferenceSession.OnnxValueMapType, name: string): ort.Tensor {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`MoG ONNX output ${name} is missing.`);
  return tensor;
}

function copiedFloatTensor(tensor: ort.Tensor): DenseFloatTensor {
  const source = tensorFloatData(tensor);
  const data = new Float32Array(source.length);
  data.set(source);
  return { data, dims: tensorDimensions(tensor) };
}

function ncthwFromDense(tensor: DenseFloatTensor, label: string): NcthwTensor {
  if (tensor.dims.length !== 5) throw new Error(`${label} must be a five-dimensional NCTHW tensor.`);
  const batch = tensor.dims[0];
  const channels = tensor.dims[1];
  const frames = tensor.dims[2];
  const height = tensor.dims[3];
  const width = tensor.dims[4];
  if (batch === undefined || channels === undefined || frames === undefined || height === undefined || width === undefined) throw new Error(`${label} dimensions are incomplete.`);
  const expected = batch * channels * frames * height * width;
  if (tensor.data.length !== expected) throw new Error(`${label} element count is inconsistent with its shape.`);
  return { data: tensor.data, batch, channels, frames, height, width };
}

function nchwFromDense(tensor: DenseFloatTensor, label: string): NchwTensor {
  if (tensor.dims.length !== 4) throw new Error(`${label} must be a four-dimensional NCHW tensor.`);
  const batch = tensor.dims[0];
  const channels = tensor.dims[1];
  const height = tensor.dims[2];
  const width = tensor.dims[3];
  if (batch === undefined || channels === undefined || height === undefined || width === undefined) throw new Error(`${label} dimensions are incomplete.`);
  const expected = batch * channels * height * width;
  if (tensor.data.length !== expected) throw new Error(`${label} element count is inconsistent with its shape.`);
  return { data: tensor.data, batch, channels, height, width };
}

function denseFeed(tensor: DenseFloatTensor): ort.Tensor {
  return floatTensor(tensor.data, tensor.dims);
}

function videoFeed(video: NcthwTensor): ort.Tensor {
  return floatTensor(video.data, [video.batch, video.channels, video.frames, video.height, video.width]);
}

function imageFeed(image: NchwTensor): ort.Tensor {
  return floatTensor(image.data, [image.batch, image.channels, image.height, image.width]);
}

async function release(session: ort.InferenceSession): Promise<void> {
  await session.release();
}

async function runMotion(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
  first: NchwTensor,
  second: NchwTensor,
): Promise<{ readonly motion: MogMotion; readonly provider: ModelExecutionProvider }> {
  const loaded = await loadOrtModel(asset, profile);
  try {
    const outputs = await loaded.session.run({ first: imageFeed(first), second: imageFeed(second) });
    return {
      motion: {
        flow: nchwFromDense(copiedFloatTensor(outputTensor(outputs, 'flow')), 'MoG flow'),
        mask: nchwFromDense(copiedFloatTensor(outputTensor(outputs, 'mask')), 'MoG motion mask'),
      },
      provider: loaded.provider,
    };
  } finally {
    await release(loaded.session);
  }
}

async function runEncoder(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
  video: NcthwTensor,
  hiddenCount: number,
): Promise<{
  readonly posterior: NcthwTensor;
  readonly hidden: readonly DenseFloatTensor[];
  readonly provider: ModelExecutionProvider;
}> {
  const loaded = await loadOrtModel(asset, profile);
  try {
    const outputs = await loaded.session.run({ video: videoFeed(video) });
    const posterior = ncthwFromDense(copiedFloatTensor(outputTensor(outputs, 'posterior_parameters')), 'MoG VAE posterior parameters');
    const hidden: DenseFloatTensor[] = [];
    for (let index = 0; index < hiddenCount; index += 1) {
      hidden.push(copiedFloatTensor(outputTensor(outputs, `reference_hidden_${index}`)));
    }
    return { posterior, hidden, provider: loaded.provider };
  } finally {
    await release(loaded.session);
  }
}

async function runCondition(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
  first: NchwTensor,
): Promise<{ readonly cross: DenseFloatTensor; readonly provider: ModelExecutionProvider }> {
  const loaded = await loadOrtModel(asset, profile);
  try {
    const outputs = await loaded.session.run({ first: imageFeed(first) });
    return { cross: copiedFloatTensor(outputTensor(outputs, 'cross_attention')), provider: loaded.provider };
  } finally {
    await release(loaded.session);
  }
}

function denoiserFeeds(
  sample: NcthwTensor,
  timestep: number,
  cross: DenseFloatTensor,
  concatLatent: NcthwTensor,
  motion: MogMotion,
  fps: number,
): Record<string, ort.Tensor> {
  return {
    sample: videoFeed(sample),
    timestep: int64Tensor(new BigInt64Array([BigInt(timestep)]), [1]),
    cross_attention: denseFeed(cross),
    concat_latent: videoFeed(concatLatent),
    flow: imageFeed(motion.flow),
    mask: imageFeed(motion.mask),
    fps: int64Tensor(new BigInt64Array([BigInt(fps)]), [1]),
  };
}

function decoderFeeds(latent: NcthwTensor, hidden: readonly DenseFloatTensor[]): Record<string, ort.Tensor> {
  const feeds: Record<string, ort.Tensor> = { latent: videoFeed(latent) };
  for (let index = 0; index < hidden.length; index += 1) {
    const tensor = hidden[index];
    if (!tensor) throw new Error(`MoG reference hidden tensor ${index} is missing.`);
    feeds[`reference_hidden_${index}`] = denseFeed(tensor);
  }
  return feeds;
}

function reducedDecodeIndices(frames: number): readonly number[] {
  if (frames < 4) throw new Error('MoG decoder correction requires at least four frames.');
  const output: number[] = [];
  for (let index = 0; index < frames; index += 1) {
    if (index === 1 || index === frames - 2) continue;
    output.push(index);
  }
  return output;
}

export class MogOnnxAdapter {
  constructor(
    private readonly assets: MogAssetBundle,
    private readonly profile: HardwareProfile,
  ) {}

  async generate(
    first: NchwTensor,
    second: NchwTensor,
    onProgress: (progress: MogProgress) => void = () => undefined,
  ): Promise<MogResult> {
    if (first.batch !== 1 || second.batch !== 1 || first.channels !== 3 || second.channels !== 3) throw new Error('MoG browser adapter expects one RGB endpoint pair.');
    if (first.height !== second.height || first.width !== second.width) throw new Error('MoG endpoint dimensions differ.');
    const frameCount = this.assets.frames ?? 16;
    const scaleFactor = this.assets.posteriorScaleFactor ?? 0.18215;
    const fps = this.assets.fps ?? 24;
    const scheduleConfig = this.assets.schedule ?? DEFAULT_MOG_SCHEDULE;
    if (scheduleConfig.inferenceSteps <= 0) throw new Error('MoG DDIM step count must be positive.');
    const providers: ModelExecutionProvider[] = [];

    onProgress({ stage: 'motion', current: 0, total: 1, provider: null });
    const motionResult = await runMotion(this.assets.motion, this.profile, first, second);
    providers.push(motionResult.provider);
    onProgress({ stage: 'motion', current: 1, total: 1, provider: motionResult.provider });

    const endpointVideo = makeMogEndpointVideo(first, second, frameCount);
    onProgress({ stage: 'encode', current: 0, total: 1, provider: null });
    const encoded = await runEncoder(this.assets.vaeEncoder, this.profile, endpointVideo, this.assets.referenceHiddenCount);
    providers.push(encoded.provider);
    const latent = sampleMogPosterior(encoded.posterior, scaleFactor);
    const warpedInterior = warpMogLatentGuidance(latent, motionResult.motion);
    const concatLatent = makeMogConcatLatent(latent, warpedInterior);
    onProgress({ stage: 'encode', current: 1, total: 1, provider: encoded.provider });

    onProgress({ stage: 'condition', current: 0, total: 1, provider: null });
    const condition = await runCondition(this.assets.imageCondition, this.profile, first);
    providers.push(condition.provider);
    onProgress({ stage: 'condition', current: 1, total: 1, provider: condition.provider });

    const schedule = makeMogDdimSchedule(scheduleConfig);
    let sample: NcthwTensor = {
      data: mogNormalNoise(latent.data.length),
      batch: latent.batch,
      channels: latent.channels,
      frames: latent.frames,
      height: latent.height,
      width: latent.width,
    };
    const denoiser = await loadOrtModel(this.assets.denoiser, this.profile);
    providers.push(denoiser.provider);
    try {
      for (let scheduleIndex = schedule.length - 1; scheduleIndex >= 0; scheduleIndex -= 1) {
        const step = schedule[scheduleIndex];
        if (!step) throw new Error(`MoG DDIM schedule entry ${scheduleIndex} is missing.`);
        const completed = schedule.length - 1 - scheduleIndex;
        onProgress({ stage: 'denoise', current: completed, total: schedule.length, provider: denoiser.provider });
        const outputs = await denoiser.session.run(denoiserFeeds(
          sample,
          step.trainingTimestep,
          condition.cross,
          concatLatent,
          motionResult.motion,
          fps,
        ));
        const velocity = ncthwFromDense(copiedFloatTensor(outputTensor(outputs, 'velocity')), 'MoG denoiser velocity');
        if (velocity.data.length !== sample.data.length) throw new Error('MoG denoiser output shape differs from the sample.');
        sample = {
          ...sample,
          data: mogDdimStep(sample.data, velocity.data, step, mogNormalNoise(sample.data.length)),
        };
      }
      onProgress({ stage: 'denoise', current: schedule.length, total: schedule.length, provider: denoiser.provider });
    } finally {
      await release(denoiser.session);
    }

    onProgress({ stage: 'decode', current: 0, total: 2, provider: null });
    const decoder = await loadOrtModel(this.assets.vaeDecoder, this.profile);
    providers.push(decoder.provider);
    let decoded: NcthwTensor;
    try {
      const fullOutputs = await decoder.session.run(decoderFeeds(sample, encoded.hidden));
      decoded = ncthwFromDense(copiedFloatTensor(outputTensor(fullOutputs, 'decoded')), 'MoG decoded video');
      onProgress({ stage: 'decode', current: 1, total: 2, provider: decoder.provider });

      const reducedLatent = selectMogFrames(sample, reducedDecodeIndices(sample.frames));
      const reducedOutputs = await decoder.session.run(decoderFeeds(reducedLatent, encoded.hidden));
      const reducedDecoded = ncthwFromDense(copiedFloatTensor(outputTensor(reducedOutputs, 'decoded')), 'MoG reduced decoded video');
      const targetStart = Math.floor(decoded.frames / 2) - 1;
      const sourceStart = Math.floor(decoded.frames / 2) - 2;
      decoded = replaceMogFrames(decoded, reducedDecoded, targetStart, sourceStart, 2);
      onProgress({ stage: 'decode', current: 2, total: 2, provider: decoder.provider });
    } finally {
      await release(decoder.session);
    }

    return { video: decoded, providers };
  }
}
