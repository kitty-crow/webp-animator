import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import { makeMogEndpointVideo, mogNormalNoise, sampleMogPosterior, type NcthwTensor } from './mog-latent.js';
import { floatTensor, int64Tensor, loadOrtModel, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import type { NchwTensor } from './softsplat.js';
import {
  makeToonCrafterDdimSchedule,
  toonCrafterDdimStep,
  toonCrafterScheduleForLatentWidth,
  type ToonCrafterScheduleConfig,
} from './tooncrafter-scheduler.js';

export interface ToonCrafterAssetBundle {
  readonly vaeEncoder: BrowserModelAsset;
  readonly imageCondition: BrowserModelAsset;
  readonly denoiser: BrowserModelAsset;
  readonly vaeDecoder: BrowserModelAsset;
  readonly frames?: number;
  readonly posteriorScaleFactor?: number;
  readonly referenceHiddenCount: number;
  readonly fps?: number;
  readonly schedule?: Omit<ToonCrafterScheduleConfig, 'spacing'>;
}

export interface ToonCrafterProgress {
  readonly stage: 'encode' | 'condition' | 'denoise' | 'decode';
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
}

export interface ToonCrafterResult {
  readonly video: NcthwTensor;
  readonly providers: readonly ModelExecutionProvider[];
}

interface DenseFloatTensor {
  readonly data: Float32Array;
  readonly dims: readonly number[];
}

function outputTensor(outputs: ort.InferenceSession.OnnxValueMapType, name: string): ort.Tensor {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`ToonCrafter ONNX output ${name} is missing.`);
  return tensor;
}

function copiedFloatTensor(tensor: ort.Tensor): DenseFloatTensor {
  const source = tensorFloatData(tensor);
  const data = new Float32Array(source.length);
  data.set(source);
  return { data, dims: tensorDimensions(tensor) };
}

function ncthwFromDense(tensor: DenseFloatTensor, label: string): NcthwTensor {
  if (tensor.dims.length !== 5) throw new Error(`${label} must be NCTHW.`);
  const batch = tensor.dims[0];
  const channels = tensor.dims[1];
  const frames = tensor.dims[2];
  const height = tensor.dims[3];
  const width = tensor.dims[4];
  if (batch === undefined || channels === undefined || frames === undefined || height === undefined || width === undefined) throw new Error(`${label} dimensions are incomplete.`);
  if (tensor.data.length !== batch * channels * frames * height * width) throw new Error(`${label} data length does not match its shape.`);
  return { data: tensor.data, batch, channels, frames, height, width };
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
    const posterior = ncthwFromDense(copiedFloatTensor(outputTensor(outputs, 'posterior_parameters')), 'ToonCrafter VAE posterior');
    const hidden: DenseFloatTensor[] = [];
    for (let index = 0; index < hiddenCount; index += 1) hidden.push(copiedFloatTensor(outputTensor(outputs, `reference_hidden_${index}`)));
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

function endpointConcat(latent: NcthwTensor): NcthwTensor {
  if (latent.frames < 2) throw new Error('ToonCrafter latent must contain at least two temporal positions.');
  const data = new Float32Array(latent.data.length);
  const frameSize = latent.height * latent.width;
  for (let channel = 0; channel < latent.channels; channel += 1) {
    for (const frame of [0, latent.frames - 1]) {
      const source = (channel * latent.frames + frame) * frameSize;
      data.set(latent.data.subarray(source, source + frameSize), source);
    }
  }
  return { ...latent, data };
}

function denoiserFeeds(
  sample: NcthwTensor,
  timestep: number,
  cross: DenseFloatTensor,
  concatLatent: NcthwTensor,
  fps: number,
): Record<string, ort.Tensor> {
  return {
    sample: videoFeed(sample),
    timestep: int64Tensor(new BigInt64Array([BigInt(timestep)]), [1]),
    cross_attention: denseFeed(cross),
    concat_latent: videoFeed(concatLatent),
    fps: int64Tensor(new BigInt64Array([BigInt(fps)]), [1]),
  };
}

function decoderFeeds(latent: NcthwTensor, hidden: readonly DenseFloatTensor[]): Record<string, ort.Tensor> {
  const feeds: Record<string, ort.Tensor> = { latent: videoFeed(latent) };
  for (let index = 0; index < hidden.length; index += 1) {
    const tensor = hidden[index];
    if (!tensor) throw new Error(`ToonCrafter reference hidden tensor ${index} is missing.`);
    feeds[`reference_hidden_${index}`] = denseFeed(tensor);
  }
  return feeds;
}

export class ToonCrafterOnnxAdapter {
  constructor(
    private readonly assets: ToonCrafterAssetBundle,
    private readonly profile: HardwareProfile,
  ) {}

  async generate(
    first: NchwTensor,
    second: NchwTensor,
    onProgress: (progress: ToonCrafterProgress) => void = () => undefined,
  ): Promise<ToonCrafterResult> {
    if (first.batch !== 1 || second.batch !== 1 || first.channels !== 3 || second.channels !== 3) throw new Error('ToonCrafter browser adapter expects one RGB endpoint pair.');
    if (first.height !== second.height || first.width !== second.width) throw new Error('ToonCrafter endpoint dimensions differ.');
    const frameCount = this.assets.frames ?? 16;
    const scaleFactor = this.assets.posteriorScaleFactor ?? 0.18215;
    const fps = this.assets.fps ?? 24;
    const providers: ModelExecutionProvider[] = [];

    const endpointVideo = makeMogEndpointVideo(first, second, frameCount);
    onProgress({ stage: 'encode', current: 0, total: 1, provider: null });
    const encoded = await runEncoder(this.assets.vaeEncoder, this.profile, endpointVideo, this.assets.referenceHiddenCount);
    providers.push(encoded.provider);
    const latent = sampleMogPosterior(encoded.posterior, scaleFactor);
    const concatLatent = endpointConcat(latent);
    onProgress({ stage: 'encode', current: 1, total: 1, provider: encoded.provider });

    onProgress({ stage: 'condition', current: 0, total: 1, provider: null });
    const condition = await runCondition(this.assets.imageCondition, this.profile, first);
    providers.push(condition.provider);
    onProgress({ stage: 'condition', current: 1, total: 1, provider: condition.provider });

    const selectedSchedule = toonCrafterScheduleForLatentWidth(latent.width);
    const scheduleConfig: ToonCrafterScheduleConfig = this.assets.schedule
      ? { ...this.assets.schedule, spacing: selectedSchedule.spacing }
      : selectedSchedule;
    const schedule = makeToonCrafterDdimSchedule(scheduleConfig);
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
        if (!step) throw new Error(`ToonCrafter DDIM step ${scheduleIndex} is missing.`);
        const completed = schedule.length - 1 - scheduleIndex;
        onProgress({ stage: 'denoise', current: completed, total: schedule.length, provider: denoiser.provider });
        const outputs = await denoiser.session.run(denoiserFeeds(
          sample,
          step.trainingTimestep,
          condition.cross,
          concatLatent,
          fps,
        ));
        const velocity = ncthwFromDense(copiedFloatTensor(outputTensor(outputs, 'velocity')), 'ToonCrafter denoiser velocity');
        if (velocity.data.length !== sample.data.length) throw new Error('ToonCrafter denoiser output shape differs from its sample.');
        sample = {
          ...sample,
          data: toonCrafterDdimStep(sample.data, velocity.data, step, mogNormalNoise(sample.data.length)),
        };
      }
      onProgress({ stage: 'denoise', current: schedule.length, total: schedule.length, provider: denoiser.provider });
    } finally {
      await release(denoiser.session);
    }

    onProgress({ stage: 'decode', current: 0, total: 1, provider: null });
    const decoder = await loadOrtModel(this.assets.vaeDecoder, this.profile);
    providers.push(decoder.provider);
    let decoded: NcthwTensor;
    try {
      const outputs = await decoder.session.run(decoderFeeds(sample, encoded.hidden));
      decoded = ncthwFromDense(copiedFloatTensor(outputTensor(outputs, 'decoded')), 'ToonCrafter decoded video');
    } finally {
      await release(decoder.session);
    }
    onProgress({ stage: 'decode', current: 1, total: 1, provider: decoder.provider });
    return { video: decoded, providers };
  }
}
