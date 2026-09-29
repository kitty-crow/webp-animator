import type * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import { loadModelManifest } from './model-manifest.js';
import { loadOrtModel, floatTensor, tensorDimensions, tensorFloatData, type ModelExecutionProvider } from './ort-runtime.js';
import { auditProPainterFrame, type ProPainterAuditResult } from './propainter-mask.js';

export interface ProPainterAssetBundle {
  readonly repairWindow: BrowserModelAsset;
  readonly windowSize: number;
  readonly internalWidth: number;
  readonly internalHeight: number;
  readonly overlap: number;
}

export interface ProPainterProgress {
  readonly current: number;
  readonly total: number;
  readonly provider: ModelExecutionProvider | null;
  readonly detail: string;
}

export interface ProPainterRepairResult {
  readonly frames: readonly ImageData[];
  readonly provider: ModelExecutionProvider | null;
  readonly repairedPixels: number;
  readonly windows: number;
}

interface NchwTensor {
  readonly data: Float32Array;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
}

interface RepairPlanes {
  readonly red: Float32Array;
  readonly green: Float32Array;
  readonly blue: Float32Array;
  readonly count: Uint16Array;
}

function requireCommonCanvas(frames: readonly ImageData[]): { readonly width: number; readonly height: number } {
  const first = frames[0];
  if (!first) throw new Error('ProPainter repair requires at least one frame.');
  for (const frame of frames) {
    if (frame.width !== first.width || frame.height !== first.height) {
      throw new Error('ProPainter repair requires a common frame canvas.');
    }
  }
  return { width: first.width, height: first.height };
}

function originalAudit(frame: ImageData): ProPainterAuditResult {
  const pixels = frame.width * frame.height;
  const expectedAlpha = new Uint8Array(pixels);
  for (let index = 0; index < pixels; index += 1) expectedAlpha[index] = frame.data[index * 4 + 3] ?? 0;
  return {
    mask: new Uint8Array(pixels),
    expectedAlpha,
    width: frame.width,
    height: frame.height,
    maskPixels: 0,
  };
}

function auditsFor(frames: readonly ImageData[]): readonly ProPainterAuditResult[] {
  return frames.map((frame, index): ProPainterAuditResult => {
    if (index === 0 || index + 1 === frames.length) return originalAudit(frame);
    const previous = frames[index - 1];
    const following = frames[index + 1];
    if (!previous || !following) throw new Error(`ProPainter temporal neighbours are missing at frame ${index}.`);
    return auditProPainterFrame(previous, frame, following);
  });
}

function frameToNchw(frame: ImageData): NchwTensor {
  const pixels = frame.width * frame.height;
  const data = new Float32Array(3 * pixels);
  for (let index = 0; index < pixels; index += 1) {
    const rgba = index * 4;
    const alpha = (frame.data[rgba + 3] ?? 0) / 255;
    for (let channel = 0; channel < 3; channel += 1) {
      const value = (frame.data[rgba + channel] ?? 0) * alpha;
      data[channel * pixels + index] = value / 127.5 - 1;
    }
  }
  return { data, channels: 3, height: frame.height, width: frame.width };
}

function bilinearSample(
  data: Float32Array,
  offset: number,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const wx = Math.max(0, Math.min(1, x - x0));
  const wy = Math.max(0, Math.min(1, y - y0));
  const v00 = data[offset + y0 * width + x0] ?? 0;
  const v01 = data[offset + y0 * width + x1] ?? 0;
  const v10 = data[offset + y1 * width + x0] ?? 0;
  const v11 = data[offset + y1 * width + x1] ?? 0;
  const top = v00 + (v01 - v00) * wx;
  const bottom = v10 + (v11 - v10) * wx;
  return top + (bottom - top) * wy;
}

function resizeNchw(input: NchwTensor, width: number, height: number): NchwTensor {
  if (input.width === width && input.height === height) return input;
  const sourcePixels = input.width * input.height;
  const targetPixels = width * height;
  const data = new Float32Array(input.channels * targetPixels);
  const scaleX = input.width / width;
  const scaleY = input.height / height;
  for (let channel = 0; channel < input.channels; channel += 1) {
    const sourceOffset = channel * sourcePixels;
    const targetOffset = channel * targetPixels;
    for (let y = 0; y < height; y += 1) {
      const sourceY = (y + 0.5) * scaleY - 0.5;
      for (let x = 0; x < width; x += 1) {
        const sourceX = (x + 0.5) * scaleX - 0.5;
        data[targetOffset + y * width + x] = bilinearSample(
          input.data,
          sourceOffset,
          input.width,
          input.height,
          sourceX,
          sourceY,
        );
      }
    }
  }
  return { data, channels: input.channels, height, width };
}

function resizeMask(mask: Uint8Array, sourceWidth: number, sourceHeight: number, width: number, height: number): Float32Array {
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.max(0, Math.min(sourceHeight - 1, Math.floor((y + 0.5) * sourceHeight / height)));
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.max(0, Math.min(sourceWidth - 1, Math.floor((x + 0.5) * sourceWidth / width)));
      output[y * width + x] = (mask[sourceY * sourceWidth + sourceX] ?? 0) > 0 ? 1 : 0;
    }
  }
  return output;
}

function parseRepairWindow(value: ort.Tensor, windowSize: number, height: number, width: number): Float32Array {
  const dims = tensorDimensions(value);
  if (
    dims.length !== 5 ||
    dims[0] !== 1 ||
    dims[1] !== windowSize ||
    dims[2] !== 3 ||
    dims[3] !== height ||
    dims[4] !== width
  ) {
    throw new Error(`ProPainter output must be [1,${windowSize},3,${height},${width}].`);
  }
  return tensorFloatData(value);
}

function windowStarts(frameCount: number, windowSize: number, overlap: number): readonly number[] {
  if (frameCount <= windowSize) return [0];
  const stride = windowSize - overlap;
  if (stride <= 0) throw new Error('ProPainter overlap must be smaller than its window size.');
  const starts: number[] = [];
  for (let start = 0; start < frameCount; start += stride) {
    const clamped = Math.min(start, frameCount - windowSize);
    if (starts[starts.length - 1] !== clamped) starts.push(clamped);
    if (clamped + windowSize >= frameCount) break;
  }
  return starts;
}

function buildWindowInputs(
  frames: readonly ImageData[],
  audits: readonly ProPainterAuditResult[],
  start: number,
  bundle: ProPainterAssetBundle,
): { readonly frames: Float32Array; readonly masks: Float32Array; readonly actualIndices: readonly number[] } {
  const pixels = bundle.internalWidth * bundle.internalHeight;
  const rgb = new Float32Array(bundle.windowSize * 3 * pixels);
  const masks = new Float32Array(bundle.windowSize * pixels);
  const actualIndices: number[] = [];
  const finalIndex = frames.length - 1;
  for (let local = 0; local < bundle.windowSize; local += 1) {
    const actual = Math.min(finalIndex, start + local);
    const frame = frames[actual];
    const audit = audits[actual];
    if (!frame || !audit) throw new Error(`ProPainter window frame ${actual} is missing.`);
    const resized = resizeNchw(frameToNchw(frame), bundle.internalWidth, bundle.internalHeight);
    const resizedMask = resizeMask(audit.mask, frame.width, frame.height, bundle.internalWidth, bundle.internalHeight);
    rgb.set(resized.data, local * 3 * pixels);
    masks.set(resizedMask, local * pixels);
    actualIndices.push(actual);
  }
  return { frames: rgb, masks, actualIndices };
}

function accumulateWindow(
  output: Float32Array,
  actualIndices: readonly number[],
  frames: readonly ImageData[],
  audits: readonly ProPainterAuditResult[],
  planes: readonly RepairPlanes[],
  bundle: ProPainterAssetBundle,
): void {
  const modelPixels = bundle.internalWidth * bundle.internalHeight;
  for (let local = 0; local < bundle.windowSize; local += 1) {
    const actual = actualIndices[local];
    if (actual === undefined || local > 0 && actual === actualIndices[local - 1]) continue;
    const frame = frames[actual];
    const audit = audits[actual];
    const target = planes[actual];
    if (!frame || !audit || !target || audit.maskPixels === 0) continue;
    const modelData = new Float32Array(3 * modelPixels);
    const windowOffset = local * 3 * modelPixels;
    modelData.set(output.subarray(windowOffset, windowOffset + 3 * modelPixels));
    const restored = resizeNchw(
      { data: modelData, channels: 3, height: bundle.internalHeight, width: bundle.internalWidth },
      frame.width,
      frame.height,
    );
    const pixels = frame.width * frame.height;
    for (let index = 0; index < pixels; index += 1) {
      if ((audit.mask[index] ?? 0) === 0) continue;
      target.red[index] = (target.red[index] ?? 0) + (restored.data[index] ?? -1);
      target.green[index] = (target.green[index] ?? 0) + (restored.data[pixels + index] ?? -1);
      target.blue[index] = (target.blue[index] ?? 0) + (restored.data[2 * pixels + index] ?? -1);
      const count = target.count[index] ?? 0;
      target.count[index] = Math.min(65535, count + 1);
    }
  }
}

function byteFromNormalised(value: number): number {
  return Math.max(0, Math.min(255, Math.round((Math.max(-1, Math.min(1, value)) + 1) * 127.5)));
}

function composeRepairs(
  frames: readonly ImageData[],
  audits: readonly ProPainterAuditResult[],
  planes: readonly RepairPlanes[],
): { readonly frames: readonly ImageData[]; readonly repairedPixels: number } {
  let repairedPixels = 0;
  const output = frames.map((frame, frameIndex): ImageData => {
    const audit = audits[frameIndex];
    const plane = planes[frameIndex];
    if (!audit || !plane || audit.maskPixels === 0) return new ImageData(new Uint8ClampedArray(frame.data), frame.width, frame.height);
    const copy = new ImageData(new Uint8ClampedArray(frame.data), frame.width, frame.height);
    const pixels = frame.width * frame.height;
    for (let index = 0; index < pixels; index += 1) {
      if ((audit.mask[index] ?? 0) === 0) continue;
      const count = plane.count[index] ?? 0;
      if (count === 0) continue;
      const rgba = index * 4;
      copy.data[rgba] = byteFromNormalised((plane.red[index] ?? 0) / count);
      copy.data[rgba + 1] = byteFromNormalised((plane.green[index] ?? 0) / count);
      copy.data[rgba + 2] = byteFromNormalised((plane.blue[index] ?? 0) / count);
      copy.data[rgba + 3] = audit.expectedAlpha[index] ?? copy.data[rgba + 3] ?? 0;
      repairedPixels += 1;
    }
    return copy;
  });
  return { frames: output, repairedPixels };
}

export async function repairProPainterFrames(
  manifestUrl: string,
  frames: readonly ImageData[],
  profile: HardwareProfile,
  onProgress: (progress: ProPainterProgress) => void,
  ensureActive: () => void,
): Promise<ProPainterRepairResult> {
  if (frames.length < 3) return { frames, provider: null, repairedPixels: 0, windows: 0 };
  requireCommonCanvas(frames);
  const audits = auditsFor(frames);
  const maskPixels = audits.reduce((total, audit) => total + audit.maskPixels, 0);
  if (maskPixels === 0) return { frames, provider: null, repairedPixels: 0, windows: 0 };

  const manifest = await loadModelManifest(manifestUrl, 'propainter');
  if (manifest.family !== 'propainter') throw new Error('ProPainter manifest loader returned a different model family.');
  const bundle = manifest.bundle;
  const starts = windowStarts(frames.length, bundle.windowSize, bundle.overlap);
  const planes: RepairPlanes[] = frames.map((frame): RepairPlanes => {
    const pixels = frame.width * frame.height;
    return {
      red: new Float32Array(pixels),
      green: new Float32Array(pixels),
      blue: new Float32Array(pixels),
      count: new Uint16Array(pixels),
    };
  });

  const loaded = await loadOrtModel(bundle.repairWindow, profile);
  try {
    for (let windowIndex = 0; windowIndex < starts.length; windowIndex += 1) {
      ensureActive();
      const start = starts[windowIndex];
      if (start === undefined) throw new Error(`ProPainter window ${windowIndex} has no start index.`);
      onProgress({
        current: windowIndex,
        total: starts.length,
        provider: loaded.provider,
        detail: `ProPainter · repair window ${windowIndex + 1}/${starts.length}`,
      });
      const input = buildWindowInputs(frames, audits, start, bundle);
      const outputs = await loaded.session.run({
        frames: floatTensor(input.frames, [1, bundle.windowSize, 3, bundle.internalHeight, bundle.internalWidth]),
        masks: floatTensor(input.masks, [1, bundle.windowSize, 1, bundle.internalHeight, bundle.internalWidth]),
      });
      const repaired = outputs['repaired'];
      if (!repaired) throw new Error('ProPainter ONNX output repaired is missing.');
      accumulateWindow(
        parseRepairWindow(repaired, bundle.windowSize, bundle.internalHeight, bundle.internalWidth),
        input.actualIndices,
        frames,
        audits,
        planes,
        bundle,
      );
      onProgress({
        current: windowIndex + 1,
        total: starts.length,
        provider: loaded.provider,
        detail: `ProPainter · repair window ${windowIndex + 1}/${starts.length} complete`,
      });
    }
  } finally {
    await loaded.session.release();
  }

  const composed = composeRepairs(frames, audits, planes);
  return {
    frames: composed.frames,
    provider: loaded.provider,
    repairedPixels: composed.repairedPixels,
    windows: starts.length,
  };
}
