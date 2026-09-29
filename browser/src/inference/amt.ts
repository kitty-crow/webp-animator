import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import { floatTensor, loadOrtModel, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import type { NchwTensor } from './softsplat.js';

export interface AmtScaleAsset {
  readonly scaleFactor: number;
  readonly divisor: number;
  readonly asset: BrowserModelAsset;
}

export interface AmtAssetBundle {
  readonly scales: readonly AmtScaleAsset[];
}

export interface AmtInterpolationResult {
  readonly tensor: NchwTensor;
  readonly provider: ModelExecutionProvider;
  readonly scaleFactor: number;
}

function validateImage(image: NchwTensor, label: string): void {
  if (image.batch !== 1 || image.channels !== 3) throw new Error(`${label} must be one RGB NCHW tensor.`);
  if (!Number.isInteger(image.height) || image.height <= 0 || !Number.isInteger(image.width) || image.width <= 0) {
    throw new Error(`${label} dimensions are invalid.`);
  }
  if (image.data.length !== 3 * image.height * image.width) throw new Error(`${label} data length is invalid.`);
}

function tensorIndex(channel: number, y: number, x: number, height: number, width: number): number {
  return (channel * height + y) * width + x;
}

interface PaddedTensor {
  readonly tensor: NchwTensor;
  readonly left: number;
  readonly top: number;
}

function padReplicate(input: NchwTensor, divisor: number): PaddedTensor {
  if (!Number.isInteger(divisor) || divisor <= 0) throw new Error('AMT padding divisor must be positive.');
  const padHeight = (((Math.floor(input.height / divisor) + 1) * divisor) - input.height) % divisor;
  const padWidth = (((Math.floor(input.width / divisor) + 1) * divisor) - input.width) % divisor;
  const left = Math.floor(padWidth / 2);
  const right = padWidth - left;
  const top = Math.floor(padHeight / 2);
  const bottom = padHeight - top;
  const width = input.width + left + right;
  const height = input.height + top + bottom;
  const data = new Float32Array(3 * height * width);

  for (let channel = 0; channel < 3; channel += 1) {
    for (let y = 0; y < height; y += 1) {
      const sourceY = Math.max(0, Math.min(input.height - 1, y - top));
      for (let x = 0; x < width; x += 1) {
        const sourceX = Math.max(0, Math.min(input.width - 1, x - left));
        data[tensorIndex(channel, y, x, height, width)] = input.data[tensorIndex(channel, sourceY, sourceX, input.height, input.width)] ?? 0;
      }
    }
  }
  return { tensor: { data, batch: 1, channels: 3, height, width }, left, top };
}

function unpad(input: NchwTensor, width: number, height: number, left: number, top: number): NchwTensor {
  if (input.batch !== 1 || input.channels !== 3) throw new Error('AMT output shape is invalid.');
  if (left < 0 || top < 0 || left + width > input.width || top + height > input.height) throw new Error('AMT unpad rectangle is invalid.');
  const data = new Float32Array(3 * width * height);
  for (let channel = 0; channel < 3; channel += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        data[tensorIndex(channel, y, x, height, width)] = input.data[tensorIndex(channel, y + top, x + left, input.height, input.width)] ?? 0;
      }
    }
  }
  return { data, batch: 1, channels: 3, height, width };
}

function nchwFromOutput(value: ort.Tensor): NchwTensor {
  const dims = tensorDimensions(value);
  if (dims.length !== 4) throw new Error('AMT output is not NCHW.');
  const batch = dims[0];
  const channels = dims[1];
  const height = dims[2];
  const width = dims[3];
  if (batch !== 1 || channels !== 3 || height === undefined || width === undefined) throw new Error('AMT output dimensions are invalid.');
  const source = tensorFloatData(value);
  const data = new Float32Array(source.length);
  data.set(source);
  return { data, batch, channels, height, width };
}

function likelyMemoryFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase();
  return message.includes('memory')
    || message.includes('allocation')
    || message.includes('buffer')
    || message.includes('resource')
    || message.includes('device lost');
}

export class AmtOnnxAdapter {
  constructor(
    private readonly assets: AmtAssetBundle,
    private readonly profile: HardwareProfile,
  ) {
    if (assets.scales.length === 0) throw new Error('AMT requires at least one exported scale graph.');
  }

  async interpolate(first: NchwTensor, second: NchwTensor, ratio: number): Promise<AmtInterpolationResult> {
    validateImage(first, 'AMT first endpoint');
    validateImage(second, 'AMT second endpoint');
    if (first.width !== second.width || first.height !== second.height) throw new Error('AMT endpoint sizes differ.');
    if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) throw new Error('AMT interpolation ratio must be in [0, 1].');
    const variants = [...this.assets.scales].sort((left, right) => right.scaleFactor - left.scaleFactor);
    let lastError: unknown = null;

    for (let index = 0; index < variants.length; index += 1) {
      const variant = variants[index];
      if (!variant) throw new Error(`AMT scale variant ${index} is missing.`);
      const firstPadded = padReplicate(first, variant.divisor);
      const secondPadded = padReplicate(second, variant.divisor);
      const loaded = await loadOrtModel(variant.asset, this.profile);
      try {
        const outputs = await loaded.session.run({
          first: floatTensor(firstPadded.tensor.data, [1, 3, firstPadded.tensor.height, firstPadded.tensor.width]),
          second: floatTensor(secondPadded.tensor.data, [1, 3, secondPadded.tensor.height, secondPadded.tensor.width]),
          timestep: floatTensor(new Float32Array([ratio]), [1, 1, 1, 1]),
        });
        const output = outputs['interpolated'];
        if (!output) throw new Error('AMT ONNX output interpolated is missing.');
        const tensor = unpad(nchwFromOutput(output), first.width, first.height, firstPadded.left, firstPadded.top);
        return { tensor, provider: loaded.provider, scaleFactor: variant.scaleFactor };
      } catch (error: unknown) {
        lastError = error;
        if (!likelyMemoryFailure(error) || index + 1 >= variants.length) throw error;
        console.warn(`AMT scale ${variant.scaleFactor} failed under memory pressure; trying the next exported fallback.`, error);
      } finally {
        await loaded.session.release();
      }
    }
    throw lastError instanceof Error ? lastError : new Error('AMT exhausted all exported scale fallbacks.');
  }
}
