import type { RenderInterpolationEngine, RifeMultiplier } from '../inference-worker-protocol.js';
import type { HardwareProfile } from '../types.js';
import { loadModelManifest, type ManifestBackedFamily } from './model-manifest.js';
import { MogOnnxAdapter } from './mog.js';
import type { NcthwTensor } from './mog-latent.js';
import type { ModelExecutionProvider } from './ort-runtime.js';
import { ResShiftOnnxAdapter } from './resshift.js';
import type { NchwTensor } from './softsplat.js';
import { ToonCrafterOnnxAdapter } from './tooncrafter.js';

export type GenerativeInterpolationEngine = Exclude<RenderInterpolationEngine, 'none' | 'rife' | 'amt'>;

export interface GenerativeInterpolationProgress {
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly detail: string;
}

export interface GenerativeInterpolationResult {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

function modelFamily(engine: GenerativeInterpolationEngine): ManifestBackedFamily {
  return engine;
}

function nextMultiple(value: number, multiple: number): number {
  return Math.ceil(value / multiple) * multiple;
}

function imageToNchw(frame: ImageData, padMultiple = 1): NchwTensor {
  const width = nextMultiple(frame.width, padMultiple);
  const height = nextMultiple(frame.height, padMultiple);
  const pixels = width * height;
  const data = new Float32Array(3 * pixels);
  data.fill(-1);
  for (let y = 0; y < frame.height; y += 1) {
    for (let x = 0; x < frame.width; x += 1) {
      const rgba = (y * frame.width + x) * 4;
      const alpha = (frame.data[rgba + 3] ?? 0) / 255;
      const destination = y * width + x;
      for (let channel = 0; channel < 3; channel += 1) {
        const component = (frame.data[rgba + channel] ?? 0) * alpha;
        data[channel * pixels + destination] = component / 127.5 - 1;
      }
    }
  }
  return { data, batch: 1, channels: 3, height, width };
}

function blendedAlpha(first: ImageData, second: ImageData, x: number, y: number, ratio: number): number {
  const firstIndex = (y * first.width + x) * 4 + 3;
  const secondIndex = (y * second.width + x) * 4 + 3;
  const firstAlpha = first.data[firstIndex] ?? 0;
  const secondAlpha = second.data[secondIndex] ?? 0;
  return Math.max(0, Math.min(255, Math.round(firstAlpha * (1 - ratio) + secondAlpha * ratio)));
}

function byteFromModel(value: number): number {
  return Math.max(0, Math.min(255, Math.round((Math.max(-1, Math.min(1, value)) + 1) * 127.5)));
}

function imageFromNchw(
  tensor: NchwTensor,
  first: ImageData,
  second: ImageData,
  ratio: number,
): ImageData {
  if (tensor.batch !== 1 || tensor.channels < 3) throw new Error('Generative interpolation output must contain one RGB image.');
  if (first.width !== second.width || first.height !== second.height) throw new Error('Generative alpha endpoints differ in size.');
  if (tensor.width < first.width || tensor.height < first.height) throw new Error('Generative output is smaller than the source canvas.');
  const output = new ImageData(first.width, first.height);
  const modelPixels = tensor.width * tensor.height;
  for (let y = 0; y < first.height; y += 1) {
    for (let x = 0; x < first.width; x += 1) {
      const modelSpatial = y * tensor.width + x;
      const destination = (y * first.width + x) * 4;
      output.data[destination] = byteFromModel(tensor.data[modelSpatial] ?? -1);
      output.data[destination + 1] = byteFromModel(tensor.data[modelPixels + modelSpatial] ?? -1);
      output.data[destination + 2] = byteFromModel(tensor.data[2 * modelPixels + modelSpatial] ?? -1);
      output.data[destination + 3] = blendedAlpha(first, second, x, y, ratio);
    }
  }
  return output;
}

function nearestVideoFrame(video: NcthwTensor, ratio: number): number {
  if (video.frames < 3) throw new Error('Generative video output must contain at least three frames.');
  const target = ratio * (video.frames - 1);
  let selected = 1;
  let bestDistance = Math.abs(1 - target);
  for (let frame = 2; frame < video.frames - 1; frame += 1) {
    const distance = Math.abs(frame - target);
    if (distance < bestDistance) {
      selected = frame;
      bestDistance = distance;
    }
  }
  return selected;
}

function imageFromVideo(
  video: NcthwTensor,
  frame: number,
  first: ImageData,
  second: ImageData,
  ratio: number,
): ImageData {
  if (video.batch !== 1 || video.channels < 3) throw new Error('Generative video output must contain one RGB clip.');
  if (frame < 0 || frame >= video.frames) throw new Error('Generative video frame index is invalid.');
  if (video.width < first.width || video.height < first.height) throw new Error('Generative video is smaller than the source canvas.');
  const output = new ImageData(first.width, first.height);
  for (let y = 0; y < first.height; y += 1) {
    for (let x = 0; x < first.width; x += 1) {
      const destination = (y * first.width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        const source = ((((channel * video.frames) + frame) * video.height + y) * video.width) + x;
        output.data[destination + channel] = byteFromModel(video.data[source] ?? -1);
      }
      output.data[destination + 3] = blendedAlpha(first, second, x, y, ratio);
    }
  }
  return output;
}

function summariseProviders(providers: readonly ModelExecutionProvider[]): ModelExecutionProvider {
  if (providers.length === 0) throw new Error('Generative model did not report an execution provider.');
  return providers.includes('wasm') ? 'wasm' : 'webgpu';
}

function stageDetail(engine: GenerativeInterpolationEngine, stage: string): string {
  const label = engine === 'resshift' ? 'ResShift' : engine === 'mog' ? 'MoG' : 'ToonCrafter';
  return `${label} · ${stage}`;
}

export async function interpolateGenerativeFrames(
  engine: GenerativeInterpolationEngine,
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  fallbackDuration: number,
  multiplier: RifeMultiplier,
  profile: HardwareProfile,
  onProgress: (progress: GenerativeInterpolationProgress) => void,
  ensureActive: () => void,
): Promise<GenerativeInterpolationResult> {
  if (frames.length < 2) throw new Error(`${engine} interpolation requires at least two frames.`);
  if (durations.length !== frames.length) throw new Error('Generative frame and duration counts differ.');
  const manifest = await loadModelManifest(manifestUrl, modelFamily(engine));
  if (manifest.family === 'amt') throw new Error('AMT manifest cannot be used by the generative interpolation path.');
  const total = (frames.length - 1) * (multiplier - 1);
  let current = 0;
  const providers: ModelExecutionProvider[] = [];
  const outputFrames: ImageData[] = [];
  const outputDurations: (number | null)[] = [];

  for (let pairIndex = 0; pairIndex < frames.length - 1; pairIndex += 1) {
    ensureActive();
    const first = frames[pairIndex];
    const second = frames[pairIndex + 1];
    if (!first || !second) throw new Error(`Generative frame pair ${pairIndex} is incomplete.`);
    if (first.width !== second.width || first.height !== second.height) throw new Error('Generative frame pair dimensions differ.');
    const sourceDuration = durations[pairIndex] ?? fallbackDuration;
    const subDuration = Math.max(1, Math.round(sourceDuration / multiplier));
    outputFrames.push(first);
    outputDurations.push(subDuration);

    if (manifest.family === 'resshift') {
      const adapter = new ResShiftOnnxAdapter(manifest.bundle, profile);
      const firstTensor = imageToNchw(first);
      const secondTensor = imageToNchw(second);
      for (let ordinal = 1; ordinal < multiplier; ordinal += 1) {
        ensureActive();
        const ratio = ordinal / multiplier;
        const result = await adapter.interpolate(firstTensor, secondTensor, ratio, (update) => {
          ensureActive();
          onProgress({ current, total, provider: update.provider, detail: stageDetail(engine, update.stage) });
        });
        providers.push(...result.providers);
        outputFrames.push(imageFromNchw(result.tensor, first, second, ratio));
        outputDurations.push(subDuration);
        current += 1;
        onProgress({ current, total, provider: summariseProviders(result.providers), detail: stageDetail(engine, 'complete intermediate') });
      }
      continue;
    }

    const firstTensor = imageToNchw(first, 64);
    const secondTensor = imageToNchw(second, 64);
    if (manifest.family === 'mog') {
      const adapter = new MogOnnxAdapter(manifest.bundle, profile);
      const result = await adapter.generate(firstTensor, secondTensor, (update) => {
        ensureActive();
        onProgress({ current, total, provider: update.provider, detail: stageDetail(engine, update.stage) });
      });
      providers.push(...result.providers);
      const provider = summariseProviders(result.providers);
      for (let ordinal = 1; ordinal < multiplier; ordinal += 1) {
        const ratio = ordinal / multiplier;
        outputFrames.push(imageFromVideo(result.video, nearestVideoFrame(result.video, ratio), first, second, ratio));
        outputDurations.push(subDuration);
        current += 1;
        onProgress({ current, total, provider, detail: stageDetail(engine, 'selected generated frame') });
      }
      continue;
    }

    const adapter = new ToonCrafterOnnxAdapter(manifest.bundle, profile);
    const result = await adapter.generate(firstTensor, secondTensor, (update) => {
      ensureActive();
      onProgress({ current, total, provider: update.provider, detail: stageDetail(engine, update.stage) });
    });
    providers.push(...result.providers);
    const provider = summariseProviders(result.providers);
    for (let ordinal = 1; ordinal < multiplier; ordinal += 1) {
      const ratio = ordinal / multiplier;
      outputFrames.push(imageFromVideo(result.video, nearestVideoFrame(result.video, ratio), first, second, ratio));
      outputDurations.push(subDuration);
      current += 1;
      onProgress({ current, total, provider, detail: stageDetail(engine, 'selected generated frame') });
    }
  }

  const last = frames[frames.length - 1];
  if (!last) throw new Error('Generative interpolation final frame is missing.');
  outputFrames.push(last);
  outputDurations.push(durations[durations.length - 1] ?? fallbackDuration);
  return { frames: outputFrames, durations: outputDurations, provider: summariseProviders(providers) };
}
