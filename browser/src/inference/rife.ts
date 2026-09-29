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
  source: RgbPlaneSource,
): void {
  const pixels = source.width * source.height;
  for (let index = 0; index < pixels; index += 1) {
    const sourceOffset = index * 4;
    target[targetChannelOffset + index] = (source.data[sourceOffset] ?? 0) / 255;
    target[targetChannelOffset + pixels + index] = (source.data[sourceOffset + 1] ?? 0) / 255;
    target[targetChannelOffset + pixels * 2 + index] = (source.data[sourceOffset + 2] ?? 0) / 255;
  }
}

function fillAlphaAsRgbPlanes(
  target: Float32Array,
  targetChannelOffset: number,
  source: ImageData,
): void {
  const pixels = source.width * source.height;
  for (let index = 0; index < pixels; index += 1) {
    const alpha = (source.data[index * 4 + 3] ?? 0) / 255;
    target[targetChannelOffset + index] = alpha;
    target[targetChannelOffset + pixels + index] = alpha;
    target[targetChannelOffset + pixels * 2 + index] = alpha;
  }
}

function buildInput(first: ImageData, second: ImageData, ratio: number, alpha: boolean): Float32Array {
  const pixels = first.width * first.height;
  const input = new Float32Array(pixels * 7);
  if (alpha) {
    fillAlphaAsRgbPlanes(input, 0, first);
    fillAlphaAsRgbPlanes(input, pixels * 3, second);
  } else {
    fillRgbPlanes(input, 0, first);
    fillRgbPlanes(input, pixels * 3, second);
  }
  input.fill(ratio, pixels * 6, pixels * 7);
  return input;
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
  rgb: Float32Array,
  alpha: Float32Array | null,
  width: number,
  height: number,
): ImageData {
  const pixels = width * height;
  if (rgb.length < pixels * 3) throw new Error('RIFE RGB output is smaller than expected.');
  if (alpha !== null && alpha.length < pixels) throw new Error('RIFE alpha output is smaller than expected.');

  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let index = 0; index < pixels; index += 1) {
    const offset = index * 4;
    rgba[offset] = Math.round(Math.max(0, Math.min(1, rgb[index] ?? 0)) * 255);
    rgba[offset + 1] = Math.round(Math.max(0, Math.min(1, rgb[pixels + index] ?? 0)) * 255);
    rgba[offset + 2] = Math.round(Math.max(0, Math.min(1, rgb[pixels * 2 + index] ?? 0)) * 255);
    rgba[offset + 3] = alpha === null
      ? 255
      : Math.round(Math.max(0, Math.min(1, alpha[index] ?? 0)) * 255);
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

  private async infer(first: ImageData, second: ImageData, ratio: number, alpha: boolean): Promise<Float32Array> {
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    if (!inputName || !outputName) throw new Error('RIFE ONNX model has invalid input/output metadata.');

    const feeds: Record<string, ort.Tensor> = {
      [inputName]: floatTensor(
        buildInput(first, second, ratio, alpha),
        [1, 7, first.height, first.width],
      ),
    };
    const outputs = await this.session.run(feeds);
    return tensorFloatData(outputTensor(outputs, outputName)).slice();
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
