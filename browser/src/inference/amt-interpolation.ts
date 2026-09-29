import type { RifeMultiplier } from '../inference-worker-protocol.js';
import type { HardwareProfile } from '../types.js';
import { AmtOnnxAdapter } from './amt.js';
import { loadModelManifest } from './model-manifest.js';
import type { ModelExecutionProvider } from './ort-runtime.js';
import type { NchwTensor } from './softsplat.js';

export interface AmtRenderProgress {
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly detail: string;
}

export interface AmtRenderResult {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

function imageRgbTensor(image: ImageData): NchwTensor {
  const pixels = image.width * image.height;
  const data = new Float32Array(3 * pixels);
  for (let index = 0; index < pixels; index += 1) {
    const rgba = index * 4;
    data[index] = (image.data[rgba] ?? 0) / 255;
    data[pixels + index] = (image.data[rgba + 1] ?? 0) / 255;
    data[2 * pixels + index] = (image.data[rgba + 2] ?? 0) / 255;
  }
  return { data, batch: 1, channels: 3, height: image.height, width: image.width };
}

function imageAlphaTensor(image: ImageData): NchwTensor {
  const pixels = image.width * image.height;
  const data = new Float32Array(3 * pixels);
  for (let index = 0; index < pixels; index += 1) {
    const alpha = (image.data[index * 4 + 3] ?? 0) / 255;
    data[index] = alpha;
    data[pixels + index] = alpha;
    data[2 * pixels + index] = alpha;
  }
  return { data, batch: 1, channels: 3, height: image.height, width: image.width };
}

function endpointsOpaque(first: ImageData, second: ImageData): boolean {
  const pixels = first.width * first.height;
  for (let index = 0; index < pixels; index += 1) {
    if ((first.data[index * 4 + 3] ?? 0) !== 255 || (second.data[index * 4 + 3] ?? 0) !== 255) return false;
  }
  return true;
}

function byteFromUnit(value: number): number {
  return Math.max(0, Math.min(255, Math.round(Math.max(0, Math.min(1, value)) * 255)));
}

function imageFromTensors(rgb: NchwTensor, alpha: NchwTensor | null, width: number, height: number): ImageData {
  if (rgb.batch !== 1 || rgb.channels !== 3 || rgb.width !== width || rgb.height !== height) throw new Error('AMT RGB output dimensions differ from the source canvas.');
  if (alpha !== null && (alpha.batch !== 1 || alpha.channels !== 3 || alpha.width !== width || alpha.height !== height)) throw new Error('AMT alpha output dimensions differ from the source canvas.');
  const output = new ImageData(width, height);
  const pixels = width * height;
  for (let index = 0; index < pixels; index += 1) {
    const destination = index * 4;
    output.data[destination] = byteFromUnit(rgb.data[index] ?? 0);
    output.data[destination + 1] = byteFromUnit(rgb.data[pixels + index] ?? 0);
    output.data[destination + 2] = byteFromUnit(rgb.data[2 * pixels + index] ?? 0);
    output.data[destination + 3] = alpha === null ? 255 : byteFromUnit(alpha.data[index] ?? 0);
  }
  return output;
}

export async function interpolateAmtFrames(
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  fallbackDuration: number,
  multiplier: RifeMultiplier,
  profile: HardwareProfile,
  onProgress: (progress: AmtRenderProgress) => void,
  ensureActive: () => void,
): Promise<AmtRenderResult> {
  if (frames.length < 2) throw new Error('AMT interpolation requires at least two frames.');
  if (durations.length !== frames.length) throw new Error('AMT frame and duration counts differ.');
  const manifest = await loadModelManifest(manifestUrl, 'amt');
  if (manifest.family !== 'amt') throw new Error('AMT manifest loader returned the wrong model family.');
  const adapter = new AmtOnnxAdapter(manifest.bundle, profile);
  const total = (frames.length - 1) * (multiplier - 1);
  let current = 0;
  let provider: ModelExecutionProvider | null = null;
  const outputFrames: ImageData[] = [];
  const outputDurations: (number | null)[] = [];

  for (let pairIndex = 0; pairIndex < frames.length - 1; pairIndex += 1) {
    ensureActive();
    const first = frames[pairIndex];
    const second = frames[pairIndex + 1];
    if (!first || !second) throw new Error(`AMT frame pair ${pairIndex} is incomplete.`);
    if (first.width !== second.width || first.height !== second.height) throw new Error('AMT frame pair dimensions differ.');
    const rgbFirst = imageRgbTensor(first);
    const rgbSecond = imageRgbTensor(second);
    const opaque = endpointsOpaque(first, second);
    const alphaFirst = opaque ? null : imageAlphaTensor(first);
    const alphaSecond = opaque ? null : imageAlphaTensor(second);
    const sourceDuration = durations[pairIndex] ?? fallbackDuration;
    const subDuration = Math.max(1, Math.round(sourceDuration / multiplier));
    outputFrames.push(first);
    outputDurations.push(subDuration);

    for (let ordinal = 1; ordinal < multiplier; ordinal += 1) {
      ensureActive();
      const ratio = ordinal / multiplier;
      onProgress({ current, total, provider, detail: `AMT · RGB t=${ratio.toFixed(3)}` });
      const rgb = await adapter.interpolate(rgbFirst, rgbSecond, ratio);
      provider = rgb.provider;
      let alpha: NchwTensor | null = null;
      let alphaScale = rgb.scaleFactor;
      if (alphaFirst !== null && alphaSecond !== null) {
        ensureActive();
        onProgress({ current, total, provider, detail: `AMT · alpha t=${ratio.toFixed(3)}` });
        const alphaResult = await adapter.interpolate(alphaFirst, alphaSecond, ratio);
        provider = provider === 'wasm' || alphaResult.provider === 'wasm' ? 'wasm' : 'webgpu';
        alpha = alphaResult.tensor;
        alphaScale = alphaResult.scaleFactor;
      }
      outputFrames.push(imageFromTensors(rgb.tensor, alpha, first.width, first.height));
      outputDurations.push(subDuration);
      current += 1;
      onProgress({ current, total, provider, detail: `AMT · complete · RGB ${rgb.scaleFactor}× · alpha ${alphaScale}×` });
    }
  }

  const last = frames[frames.length - 1];
  if (!last) throw new Error('AMT final frame is missing.');
  outputFrames.push(last);
  outputDurations.push(durations[durations.length - 1] ?? fallbackDuration);
  if (provider === null) throw new Error('AMT completed without an execution provider.');
  return { frames: outputFrames, durations: outputDurations, provider };
}
