import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import { loadModelManifest } from './model-manifest.js';
import { floatTensor, loadOrtModel, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import type { NchwTensor } from './softsplat.js';

export type FrameGeneratorFamily = 'eden' | 'speed';

export interface EdenAssetBundle {
  readonly generator: BrowserModelAsset;
  readonly internalWidth: number;
  readonly internalHeight: number;
  readonly latentDim: number;
  readonly cosSimMean: number;
  readonly cosSimStd: number;
  readonly seed: number;
}

export interface SpeedAssetBundle {
  readonly generator: BrowserModelAsset;
  readonly scales: readonly number[];
  readonly seed: number;
}

export interface FrameGeneratorProgress {
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly detail: string;
}

export interface FrameGeneratorResult {
  readonly frames: readonly ImageData[];
  readonly durations: readonly (number | null)[];
  readonly provider: ModelExecutionProvider;
}

interface BoundingBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

interface FittedTensor {
  readonly tensor: NchwTensor;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

class NormalRng {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    const normalised = Math.trunc(seed) >>> 0;
    this.state = normalised === 0 ? 0x6d2b79f5 : normalised;
  }

  private uniform(): number {
    let value = this.state;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.state = value >>> 0;
    return (this.state + 0.5) / 4_294_967_296;
  }

  next(): number {
    if (this.spare !== null) {
      const value = this.spare;
      this.spare = null;
      return value;
    }
    const first = Math.max(Number.MIN_VALUE, this.uniform());
    const second = this.uniform();
    const magnitude = Math.sqrt(-2 * Math.log(first));
    const angle = 2 * Math.PI * second;
    this.spare = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  array(length: number): Float32Array {
    if (!Number.isInteger(length) || length < 0) throw new Error('Noise length must be a non-negative integer.');
    const values = new Float32Array(length);
    for (let index = 0; index < values.length; index += 1) values[index] = this.next();
    return values;
  }
}

function imageBoundingBox(first: ImageData, second: ImageData): BoundingBox | null {
  if (first.width !== second.width || first.height !== second.height) throw new Error('Generator endpoint sizes differ.');
  let minX = first.width;
  let minY = first.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < first.height; y += 1) {
    for (let x = 0; x < first.width; x += 1) {
      const alphaIndex = (y * first.width + x) * 4 + 3;
      if ((first.data[alphaIndex] ?? 0) === 0 && (second.data[alphaIndex] ?? 0) === 0) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

function cropUnitRgb(image: ImageData, box: BoundingBox): NchwTensor {
  const pixels = box.width * box.height;
  const data = new Float32Array(3 * pixels);
  for (let y = 0; y < box.height; y += 1) {
    for (let x = 0; x < box.width; x += 1) {
      const source = ((box.y + y) * image.width + box.x + x) * 4;
      const spatial = y * box.width + x;
      data[spatial] = (image.data[source] ?? 0) / 255;
      data[pixels + spatial] = (image.data[source + 1] ?? 0) / 255;
      data[2 * pixels + spatial] = (image.data[source + 2] ?? 0) / 255;
    }
  }
  return { data, batch: 1, channels: 3, height: box.height, width: box.width };
}

function sampleBilinear(input: NchwTensor, channel: number, y: number, x: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(input.width - 1, x0 + 1);
  const y1 = Math.min(input.height - 1, y0 + 1);
  const clampedX0 = Math.max(0, Math.min(input.width - 1, x0));
  const clampedY0 = Math.max(0, Math.min(input.height - 1, y0));
  const wx = Math.max(0, Math.min(1, x - x0));
  const wy = Math.max(0, Math.min(1, y - y0));
  const pixels = input.width * input.height;
  const offset = channel * pixels;
  const at = (yy: number, xx: number): number => input.data[offset + yy * input.width + xx] ?? 0;
  const top = at(clampedY0, clampedX0) * (1 - wx) + at(clampedY0, x1) * wx;
  const bottom = at(y1, clampedX0) * (1 - wx) + at(y1, x1) * wx;
  return top * (1 - wy) + bottom * wy;
}

function resizeNchw(input: NchwTensor, width: number, height: number): NchwTensor {
  if (input.batch !== 1 || input.channels !== 3) throw new Error('Generator resize expects one RGB tensor.');
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) throw new Error('Generator resize dimensions are invalid.');
  const data = new Float32Array(3 * width * height);
  const sourceScaleX = input.width / width;
  const sourceScaleY = input.height / height;
  for (let channel = 0; channel < 3; channel += 1) {
    const offset = channel * width * height;
    for (let y = 0; y < height; y += 1) {
      const sourceY = (y + 0.5) * sourceScaleY - 0.5;
      for (let x = 0; x < width; x += 1) {
        const sourceX = (x + 0.5) * sourceScaleX - 0.5;
        data[offset + y * width + x] = sampleBilinear(input, channel, sourceY, sourceX);
      }
    }
  }
  return { data, batch: 1, channels: 3, height, width };
}

function fitWithReplicatedBorder(input: NchwTensor, width: number, height: number): FittedTensor {
  const scale = Math.min(width / input.width, height / input.height);
  const fittedWidth = Math.max(1, Math.min(width, Math.round(input.width * scale)));
  const fittedHeight = Math.max(1, Math.min(height, Math.round(input.height * scale)));
  const resized = resizeNchw(input, fittedWidth, fittedHeight);
  const xOffset = Math.floor((width - fittedWidth) / 2);
  const yOffset = Math.floor((height - fittedHeight) / 2);
  const data = new Float32Array(3 * width * height);
  const pixels = width * height;
  for (let channel = 0; channel < 3; channel += 1) {
    const outputOffset = channel * pixels;
    const resizedOffset = channel * fittedWidth * fittedHeight;
    for (let y = 0; y < height; y += 1) {
      const sourceY = Math.max(0, Math.min(fittedHeight - 1, y - yOffset));
      for (let x = 0; x < width; x += 1) {
        const sourceX = Math.max(0, Math.min(fittedWidth - 1, x - xOffset));
        data[outputOffset + y * width + x] = resized.data[resizedOffset + sourceY * fittedWidth + sourceX] ?? 0;
      }
    }
  }
  return {
    tensor: { data, batch: 1, channels: 3, height, width },
    x: xOffset,
    y: yOffset,
    width: fittedWidth,
    height: fittedHeight,
  };
}

function cropFittedAndResize(input: NchwTensor, fitted: FittedTensor, width: number, height: number): NchwTensor {
  if (input.width !== fitted.tensor.width || input.height !== fitted.tensor.height) throw new Error('Generator fitted output dimensions changed.');
  const data = new Float32Array(3 * fitted.width * fitted.height);
  const inputPixels = input.width * input.height;
  const cropPixels = fitted.width * fitted.height;
  for (let channel = 0; channel < 3; channel += 1) {
    for (let y = 0; y < fitted.height; y += 1) {
      const source = channel * inputPixels + (fitted.y + y) * input.width + fitted.x;
      const destination = channel * cropPixels + y * fitted.width;
      data.set(input.data.subarray(source, source + fitted.width), destination);
    }
  }
  return resizeNchw({ data, batch: 1, channels: 3, height: fitted.height, width: fitted.width }, width, height);
}

function tensorFromOutput(value: ort.Tensor): NchwTensor {
  const dims = tensorDimensions(value);
  if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== 3 || dims[2] === undefined || dims[3] === undefined) {
    throw new Error('Generator ONNX output must be [1,3,H,W].');
  }
  const source = tensorFloatData(value);
  const data = new Float32Array(source.length);
  data.set(source);
  return { data, batch: 1, channels: 3, height: dims[2], width: dims[3] };
}

function cosineDifference(first: NchwTensor, second: NchwTensor, mean: number, std: number): number {
  if (first.width !== second.width || first.height !== second.height) throw new Error('EDEN difference endpoints differ in size.');
  if (!Number.isFinite(std) || std === 0) throw new Error('EDEN cosine standard deviation is invalid.');
  const pixels = first.width * first.height;
  let sum = 0;
  for (let spatial = 0; spatial < pixels; spatial += 1) {
    let dot = 0;
    let firstNorm = 0;
    let secondNorm = 0;
    for (let channel = 0; channel < 3; channel += 1) {
      const index = channel * pixels + spatial;
      const a = first.data[index] ?? 0;
      const b = second.data[index] ?? 0;
      dot += a * b;
      firstNorm += a * a;
      secondNorm += b * b;
    }
    const denominator = Math.sqrt(firstNorm) * Math.sqrt(secondNorm);
    sum += denominator > 1e-12 ? dot / denominator : 0;
  }
  return (sum / Math.max(1, pixels) - mean) / std;
}

function compositeMidpoint(
  rgb: NchwTensor,
  first: ImageData,
  second: ImageData,
  box: BoundingBox,
): ImageData {
  if (rgb.width !== box.width || rgb.height !== box.height) throw new Error('Generator midpoint crop size changed unexpectedly.');
  const output = new ImageData(first.width, first.height);
  const pixels = rgb.width * rgb.height;
  for (let y = 0; y < box.height; y += 1) {
    for (let x = 0; x < box.width; x += 1) {
      const spatial = y * box.width + x;
      const destination = ((box.y + y) * first.width + box.x + x) * 4;
      output.data[destination] = Math.max(0, Math.min(255, Math.round((rgb.data[spatial] ?? 0) * 255)));
      output.data[destination + 1] = Math.max(0, Math.min(255, Math.round((rgb.data[pixels + spatial] ?? 0) * 255)));
      output.data[destination + 2] = Math.max(0, Math.min(255, Math.round((rgb.data[2 * pixels + spatial] ?? 0) * 255)));
      const firstAlpha = first.data[destination + 3] ?? 0;
      const secondAlpha = second.data[destination + 3] ?? 0;
      output.data[destination + 3] = Math.round((firstAlpha + secondAlpha) / 2);
    }
  }
  return output;
}

function nextScaledSize(width: number, height: number, scale: number, divisor: number): { readonly width: number; readonly height: number } {
  const scaledWidth = Math.max(divisor * 2, Math.round(width * scale / divisor) * divisor);
  const scaledHeight = Math.max(divisor * 2, Math.round(height * scale / divisor) * divisor);
  return { width: scaledWidth, height: scaledHeight };
}

function memoryFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();
  return message.includes('memory') || message.includes('allocation') || message.includes('buffer') || message.includes('device lost');
}

export class EdenOnnxAdapter {
  constructor(
    private readonly assets: EdenAssetBundle,
    private readonly profile: HardwareProfile,
    private readonly rng: NormalRng,
  ) {}

  async midpoint(first: NchwTensor, second: NchwTensor): Promise<{ readonly tensor: NchwTensor; readonly provider: ModelExecutionProvider }> {
    const fittedFirst = fitWithReplicatedBorder(first, this.assets.internalWidth, this.assets.internalHeight);
    const fittedSecond = fitWithReplicatedBorder(second, this.assets.internalWidth, this.assets.internalHeight);
    const loaded = await loadOrtModel(this.assets.generator, this.profile);
    try {
      const tokenCount = (this.assets.internalWidth / 32) * (this.assets.internalHeight / 32);
      const noise = this.rng.array(tokenCount * this.assets.latentDim);
      const difference = cosineDifference(first, second, this.assets.cosSimMean, this.assets.cosSimStd);
      const outputs = await loaded.session.run({
        first: floatTensor(fittedFirst.tensor.data, [1, 3, this.assets.internalHeight, this.assets.internalWidth]),
        second: floatTensor(fittedSecond.tensor.data, [1, 3, this.assets.internalHeight, this.assets.internalWidth]),
        noise: floatTensor(noise, [1, tokenCount, this.assets.latentDim]),
        difference: floatTensor(new Float32Array([difference]), [1, 1]),
      });
      const midpoint = outputs['midpoint'];
      if (!midpoint) throw new Error('EDEN ONNX output midpoint is missing.');
      const tensor = cropFittedAndResize(tensorFromOutput(midpoint), fittedFirst, first.width, first.height);
      return { tensor, provider: loaded.provider };
    } finally {
      await loaded.session.release();
    }
  }
}

export class SpeedOnnxAdapter {
  constructor(
    private readonly assets: SpeedAssetBundle,
    private readonly profile: HardwareProfile,
    private readonly rng: NormalRng,
  ) {}

  async midpoint(first: NchwTensor, second: NchwTensor): Promise<{ readonly tensor: NchwTensor; readonly provider: ModelExecutionProvider; readonly scale: number }> {
    let lastError: unknown = null;
    for (let index = 0; index < this.assets.scales.length; index += 1) {
      const scale = this.assets.scales[index];
      if (scale === undefined || !Number.isFinite(scale) || scale <= 0 || scale > 1) throw new Error('SPEED manifest contains an invalid scale.');
      const size = nextScaledSize(first.width, first.height, scale, 16);
      const scaledFirst = resizeNchw(first, size.width, size.height);
      const scaledSecond = resizeNchw(second, size.width, size.height);
      const loaded = await loadOrtModel(this.assets.generator, this.profile);
      try {
        const noise = this.rng.array(3 * size.width * size.height);
        const outputs = await loaded.session.run({
          first: floatTensor(scaledFirst.data, [1, 3, size.height, size.width]),
          second: floatTensor(scaledSecond.data, [1, 3, size.height, size.width]),
          noise: floatTensor(noise, [1, 3, size.height, size.width]),
        });
        const midpoint = outputs['midpoint'];
        if (!midpoint) throw new Error('SPEED ONNX output midpoint is missing.');
        return {
          tensor: resizeNchw(tensorFromOutput(midpoint), first.width, first.height),
          provider: loaded.provider,
          scale,
        };
      } catch (error: unknown) {
        lastError = error;
        if (!memoryFailure(error) || index + 1 >= this.assets.scales.length) throw error;
        console.warn(`SPEED scale ${scale} failed under memory pressure; trying the next fallback.`, error);
      } finally {
        await loaded.session.release();
      }
    }
    throw lastError instanceof Error ? lastError : new Error('SPEED exhausted all low-memory fallbacks.');
  }
}

export async function generateMidpointAnchors(
  family: FrameGeneratorFamily,
  manifestUrl: string,
  frames: readonly ImageData[],
  durations: readonly (number | null)[],
  fallbackDuration: number,
  profile: HardwareProfile,
  onProgress: (progress: FrameGeneratorProgress) => void,
  ensureActive: () => void,
): Promise<FrameGeneratorResult> {
  if (frames.length < 2) throw new Error(`${family} generation requires at least two frames.`);
  if (durations.length !== frames.length) throw new Error('Generator frame and duration counts differ.');
  const manifest = await loadModelManifest(manifestUrl, family);
  if (manifest.family !== 'eden' && manifest.family !== 'speed') throw new Error('Generator manifest loader returned an interpolation model.');
  const rng = new NormalRng(manifest.bundle.seed);
  const eden = manifest.family === 'eden' ? new EdenOnnxAdapter(manifest.bundle, profile, rng) : null;
  const speed = manifest.family === 'speed' ? new SpeedOnnxAdapter(manifest.bundle, profile, rng) : null;
  const outputFrames: ImageData[] = [];
  const outputDurations: (number | null)[] = [];
  let provider: ModelExecutionProvider | null = null;
  const total = frames.length - 1;

  for (let pairIndex = 0; pairIndex < frames.length - 1; pairIndex += 1) {
    ensureActive();
    const first = frames[pairIndex];
    const second = frames[pairIndex + 1];
    if (!first || !second) throw new Error(`Generator frame pair ${pairIndex} is incomplete.`);
    const sourceDuration = durations[pairIndex] ?? fallbackDuration;
    const halfDuration = Math.max(1, Math.round(sourceDuration / 2));
    outputFrames.push(first);
    outputDurations.push(halfDuration);
    const box = imageBoundingBox(first, second);
    if (box === null) {
      outputFrames.push(new ImageData(first.width, first.height));
      outputDurations.push(halfDuration);
      onProgress({ current: pairIndex + 1, total, provider, detail: `${family.toUpperCase()} · transparent midpoint` });
      continue;
    }
    const firstCrop = cropUnitRgb(first, box);
    const secondCrop = cropUnitRgb(second, box);
    if (eden !== null) {
      onProgress({ current: pairIndex, total, provider, detail: 'EDEN · two-step Euler midpoint' });
      const result = await eden.midpoint(firstCrop, secondCrop);
      provider = provider === 'wasm' || result.provider === 'wasm' ? 'wasm' : 'webgpu';
      outputFrames.push(compositeMidpoint(result.tensor, first, second, box));
    } else if (speed !== null) {
      onProgress({ current: pairIndex, total, provider, detail: 'SPEED · noisy midpoint' });
      const result = await speed.midpoint(firstCrop, secondCrop);
      provider = provider === 'wasm' || result.provider === 'wasm' ? 'wasm' : 'webgpu';
      outputFrames.push(compositeMidpoint(result.tensor, first, second, box));
    } else {
      throw new Error('Generator adapter was not initialised.');
    }
    outputDurations.push(halfDuration);
    onProgress({ current: pairIndex + 1, total, provider, detail: `${family.toUpperCase()} · midpoint complete` });
  }

  const last = frames[frames.length - 1];
  if (!last) throw new Error('Generator final frame is missing.');
  outputFrames.push(last);
  outputDurations.push(durations[durations.length - 1] ?? fallbackDuration);
  if (provider === null) provider = 'wasm';
  return { frames: outputFrames, durations: outputDurations, provider };
}
