import type { BrowserImageDecoderConstructor, DecodedInput, ProgressUpdate } from './types.js';

type ProgressCallback = (update: ProgressUpdate) => void;

function imageDecoderConstructor(): BrowserImageDecoderConstructor | null {
  const globals = globalThis as typeof globalThis & { readonly ImageDecoder?: BrowserImageDecoderConstructor };
  return globals.ImageDecoder ?? null;
}

function makeCanvas(width: number, height: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  throw new Error('This browser cannot create a canvas inside a worker. OffscreenCanvas is required for background rendering.');
}

function context2d(canvas: OffscreenCanvas | HTMLCanvasElement): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D is unavailable.');
  return context as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
}

async function decodeAnimated(file: File): Promise<{ readonly frames: readonly ImageData[]; readonly durations: readonly (number | null)[] } | null> {
  const Decoder = imageDecoderConstructor();
  if (!Decoder || !file.type) return null;
  let supported = false;
  try {
    supported = await Decoder.isTypeSupported(file.type);
  } catch (error: unknown) {
    console.debug('ImageDecoder support probe failed.', error);
    return null;
  }
  if (!supported) return null;

  const decoder = new Decoder({ data: await file.arrayBuffer(), type: file.type });
  try {
    await decoder.tracks.ready;
    const count = Math.max(1, decoder.tracks.selectedTrack?.frameCount ?? 1);
    const frames: ImageData[] = [];
    const durations: (number | null)[] = [];
    for (let index = 0; index < count; index += 1) {
      const decoded = await decoder.decode({ frameIndex: index, completeFramesOnly: true });
      const frame = decoded.image;
      try {
        const width = frame.displayWidth || frame.codedWidth;
        const height = frame.displayHeight || frame.codedHeight;
        const canvas = makeCanvas(width, height);
        const context = context2d(canvas);
        context.drawImage(frame, 0, 0, width, height);
        frames.push(context.getImageData(0, 0, width, height));
        durations.push(frame.duration !== null && frame.duration > 0 ? Math.max(1, Math.round(frame.duration / 1000)) : null);
      } finally {
        frame.close();
      }
    }
    return { frames, durations };
  } finally {
    decoder.close();
  }
}

async function decodeStill(file: File): Promise<ImageData> {
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = makeCanvas(bitmap.width, bitmap.height);
    const context = context2d(canvas);
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height);
  } finally {
    bitmap.close();
  }
}

export async function decodeInputFiles(files: readonly File[], onProgress: ProgressCallback): Promise<DecodedInput> {
  const frames: ImageData[] = [];
  const sourceDurations: (number | null)[] = [];
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    if (!file) throw new Error(`Input file ${index} is missing.`);
    onProgress({ stage: 'decode', current: index, total: files.length, fileName: file.name });
    const animated = files.length === 1 ? await decodeAnimated(file) : null;
    if (animated && animated.frames.length > 0) {
      frames.push(...animated.frames);
      sourceDurations.push(...animated.durations);
    } else {
      frames.push(await decodeStill(file));
      sourceDurations.push(null);
    }
  }
  onProgress({ stage: 'decode', current: files.length, total: files.length, fileName: null });
  return { frames, sourceDurations };
}

export async function imageDataToWebpBlob(image: ImageData, quality: number): Promise<Blob> {
  const canvas = makeCanvas(image.width, image.height);
  context2d(canvas).putImageData(image, 0, 0);
  const blob = canvas instanceof OffscreenCanvas
    ? await canvas.convertToBlob({ type: 'image/webp', quality })
    : await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((candidate: Blob | null) => {
          if (candidate) resolve(candidate);
          else reject(new Error('Browser failed to encode WebP.'));
        }, 'image/webp', quality);
      });
  if (!blob.type.includes('webp')) throw new Error('Browser Canvas does not expose a WebP encoder.');
  return blob;
}
