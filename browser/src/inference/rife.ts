import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import { browserModelDefinition } from './catalog.js';
import type { ModelExecutionProvider } from './ort-runtime.js';
import { floatTensor, loadOrtModel, tensorFloatData } from './ort-runtime.js';

export interface RifeInterpolationResult {
  readonly image: ImageData;
  readonly provider: ModelExecutionProvider;
}

interface RgbPlaneSource {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

interface RifeTensorResult {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
}

const RIFE_PADDING_MULTIPLE = 128;

export function rifePaddedDimension(value: number): number {
  if (!Number.isInteger(value) || value <= 0) throw new Error('RIFE dimensions must be positive integers.');
  return Math.ceil(value / RIFE_PADDING_MULTIPLE) * RIFE_PADDING_MULTIPLE;
}

function requireAsset() {
  const definition = browserModelDefinition('rife');
  const asset = definition.assets[0];
  if (!asset) throw new Error('RIFE browser model asset is not configured.');
  return asset;
}

function assertSameSize(first: ImageData, second: ImageData): void {
  if (first.width !== second.width || first.height !== second.height) {
    throw new Error('RIFE endpoint frames must have identical dimensions.');
  }
}

function bothOpaque(first: ImageData, second: ImageData): boolean {
  for (let offset = 3; offset < first.data.length; offset += 4) {
    if (first.data[offset] !== 255 || second.data[offset] !== 255) return false;
  }
  return true;
}

function fillRgbPlanes(
  target: Float32Array,
  targetChannelOffset: number,
  planeWidth: number,
  planeHeight: number,
  source: RgbPlaneSource,
): void {
  if (source.width > planeWidth || source.height > planeHeight) throw new Error('RIFE padded RGB plane is smaller than its source.');
  const planePixels = planeWidth * planeHeight;
  for (let y = 0; y < source.height; y += 1) {
    for (let x = 0; x < source.width; x += 1) {
      const sourceIndex = y * source.width + x;
      const sourceOffset = sourceIndex * 4;
      const targetIndex = y * planeWidth + x;
      target[targetChannelOffset + targetIndex] = (source.data[sourceOffset] ?? 0) / 255;
      target[targetChannelOffset + planePixels + targetIndex] = (source.data[sourceOffset + 1] ?? 0) / 255;
      target[targetChannelOffset + planePixels * 2 + targetIndex] = (source.data[sourceOffset + 2] ?? 0) / 255;
    }
  }
}

function fillAlphaAsRgbPlanes(
  target: Float32Array,
  targetChannelOffset: number,
  planeWidth: number,
  planeHeight: number,
  source: ImageData,
): void {
  if (source.width > planeWidth || source.height > planeHeight) throw new Error('RIFE padded alpha plane is smaller than its source.');
  const planePixels = planeWidth * planeHeight;
  for (let y = 0; y < source.height; y += 1) {
    for (let x = 0; x < source.width; x += 1) {
      const sourceIndex = y * source.width + x;
      const targetIndex = y * planeWidth + x;
      const alpha = (source.data[sourceIndex * 4 + 3] ?? 0) / 255;
      target[targetChannelOffset + targetIndex] = alpha;
      target[targetChannelOffset + planePixels + targetIndex] = alpha;
      target[targetChannelOffset + planePixels * 2 + targetIndex] = alpha;
    }
  }
}

function buildInput(
  first: ImageData,
  second: ImageData,
  ratio: number,
  alpha: boolean,
): { readonly data: Float32Array; readonly width: number; readonly height: number } {
  const width = rifePaddedDimension(first.width);
  const height = rifePaddedDimension(first.height);
  const pixels = width * height;
  const input = new Float32Array(pixels * 7);
  if (alpha) {
    fillAlphaAsRgbPlanes(input, 0, width, height, first);
    fillAlphaAsRgbPlanes(input, pixels * 3, width, height, second);
  } else {
    fillRgbPlanes(input, 0, width, height, first);
    fillRgbPlanes(input, pixels * 3, width, height, second);
  }
  input.fill(ratio, pixels * 6, pixels * 7);
  return { data: input, width, height };
}

function outputTensor(
  outputs: Readonly<Record<string, ort.Tensor>>,
  outputName: string,
): ort.Tensor {
  const tensor = outputs[outputName];
  if (!tensor) throw new Error(`RIFE ONNX output ${outputName} is missing.`);
  return tensor;
}

function outputToImage(
  rgb: RifeTensorResult,
  alpha: RifeTensorResult | null,
  width: number,
  height: number,
): ImageData {
  const paddedPixels = rgb.width * rgb.height;
  if (rgb.data.length < paddedPixels * 3) throw new Error('RIFE RGB output is smaller than expected.');
  if (alpha !== null) {
    if (alpha.width !== rgb.width || alpha.height !== rgb.height) throw new Error('RIFE alpha output dimensions differ from RGB.');
    if (alpha.data.length < paddedPixels) throw new Error('RIFE alpha output is smaller than expected.');
  }

  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = y * rgb.width + x;
      const targetIndex = y * width + x;
      const offset = targetIndex * 4;
      rgba[offset] = Math.round(Math.max(0, Math.min(1, rgb.data[sourceIndex] ?? 0)) * 255);
      rgba[offset + 1] = Math.round(Math.max(0, Math.min(1, rgb.data[paddedPixels + sourceIndex] ?? 0)) * 255);
      rgba[offset + 2] = Math.round(Math.max(0, Math.min(1, rgb.data[paddedPixels * 2 + sourceIndex] ?? 0)) * 255);
      rgba[offset + 3] = alpha === null
        ? 255
        : Math.round(Math.max(0, Math.min(1, alpha.data[sourceIndex] ?? 0)) * 255);
    }
  }
  return new ImageData(rgba, width, height);
}

export class RifeOnnxAdapter {
  static async create(profile: HardwareProfile): Promise<RifeOnnxAdapter> {
    const loaded = await loadOrtModel(requireAsset(), profile);
    if (loaded.session.inputNames.length !== 1) {
      loaded.session.release();
      throw new Error(`RIFE 4.25 v2 adapter expects one 7-channel input, found ${loaded.session.inputNames.length}.`);
    }
    if (loaded.session.outputNames.length < 1) {
      loaded.session.release();
      throw new Error('RIFE 4.25 v2 model exposes no outputs.');
    }
    return new RifeOnnxAdapter(loaded.session, loaded.provider);
  }

  private constructor(
    private readonly session: ort.InferenceSession,
    readonly provider: ModelExecutionProvider,
  ) {}

  private async infer(first: ImageData, second: ImageData, ratio: number, alpha: boolean): Promise<RifeTensorResult> {
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    if (!inputName || !outputName) throw new Error('RIFE ONNX model has invalid input/output metadata.');

    const input = buildInput(first, second, ratio, alpha);
    const feeds: Record<string, ort.Tensor> = {
      [inputName]: floatTensor(input.data, [1, 7, input.height, input.width]),
    };
    const outputs = await this.session.run(feeds);
    return {
      data: tensorFloatData(outputTensor(outputs, outputName)).slice(),
      width: input.width,
      height: input.height,
    };
  }

  async interpolate(first: ImageData, second: ImageData, ratio: number): Promise<RifeInterpolationResult> {
    assertSameSize(first, second);
    if (!Number.isFinite(ratio) || ratio <= 0 || ratio >= 1) {
      throw new Error('RIFE interpolation ratio must be strictly between 0 and 1.');
    }

    const rgb = await this.infer(first, second, ratio, false);
    const alpha = bothOpaque(first, second)
      ? null
      : await this.infer(first, second, ratio, true);
    return {
      image: outputToImage(rgb, alpha, first.width, first.height),
      provider: this.provider,
    };
  }

  async interpolateMany(
    first: ImageData,
    second: ImageData,
    ratios: readonly number[],
  ): Promise<readonly RifeInterpolationResult[]> {
    const results: RifeInterpolationResult[] = [];
    for (const ratio of ratios) results.push(await this.interpolate(first, second, ratio));
    return results;
  }

  close(): void {
    this.session.release();
  }
}
