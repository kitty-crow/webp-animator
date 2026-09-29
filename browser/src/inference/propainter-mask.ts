export interface ProPainterAuditResult {
  readonly mask: Uint8Array;
  readonly expectedAlpha: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly maskPixels: number;
}

function requireSameCanvas(frames: readonly ImageData[]): void {
  const first = frames[0];
  if (!first) throw new Error('ProPainter audit requires frames.');
  for (const frame of frames) {
    if (frame.width !== first.width || frame.height !== first.height) {
      throw new Error('ProPainter audit requires a common frame canvas.');
    }
  }
}

function alphaPlane(image: ImageData): Float32Array {
  const pixels = image.width * image.height;
  const result = new Float32Array(pixels);
  for (let index = 0; index < pixels; index += 1) {
    result[index] = (image.data[index * 4 + 3] ?? 0) / 255;
  }
  return result;
}

function maxFilterLine(
  source: Float32Array,
  target: Float32Array,
  length: number,
  sourceStart: number,
  sourceStride: number,
  targetStart: number,
  targetStride: number,
  radius: number,
): void {
  const deque = new Int32Array(length);
  let head = 0;
  let tail = 0;
  let addedThrough = -1;

  for (let outputIndex = 0; outputIndex < length; outputIndex += 1) {
    const right = Math.min(length - 1, outputIndex + radius);
    while (addedThrough < right) {
      addedThrough += 1;
      const candidate = source[sourceStart + addedThrough * sourceStride] ?? 0;
      while (tail > head) {
        const priorIndex = deque[tail - 1];
        if (priorIndex === undefined) break;
        const prior = source[sourceStart + priorIndex * sourceStride] ?? 0;
        if (prior > candidate) break;
        tail -= 1;
      }
      deque[tail] = addedThrough;
      tail += 1;
    }

    const left = Math.max(0, outputIndex - radius);
    while (tail > head) {
      const firstIndex = deque[head];
      if (firstIndex === undefined || firstIndex >= left) break;
      head += 1;
    }
    const maximumIndex = deque[head];
    target[targetStart + outputIndex * targetStride] = maximumIndex === undefined
      ? 0
      : source[sourceStart + maximumIndex * sourceStride] ?? 0;
  }
}

export function dilatePlane(
  source: Float32Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  if (source.length !== width * height) throw new Error('Dilation plane dimensions are inconsistent.');
  const safeRadius = Math.max(0, Math.floor(radius));
  if (safeRadius === 0) return source.slice();

  const horizontal = new Float32Array(source.length);
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    maxFilterLine(source, horizontal, width, y * width, 1, y * width, 1, safeRadius);
  }
  for (let x = 0; x < width; x += 1) {
    maxFilterLine(horizontal, output, height, x, width, x, width, safeRadius);
  }
  return output;
}

export function auditProPainterFrame(
  previous: ImageData,
  current: ImageData,
  following: ImageData,
): ProPainterAuditResult {
  requireSameCanvas([previous, current, following]);
  const width = current.width;
  const height = current.height;
  const radius = Math.max(8, Math.min(48, Math.round(Math.max(width, height) * 0.02)));
  const localRadius = Math.max(3, Math.floor(radius / 3));

  const previousAlpha = alphaPlane(previous);
  const currentAlpha = alphaPlane(current);
  const followingAlpha = alphaPlane(following);
  const previousSupport = dilatePlane(previousAlpha, width, height, radius);
  const followingSupport = dilatePlane(followingAlpha, width, height, radius);
  const localVisible = dilatePlane(currentAlpha, width, height, localRadius);
  const pixels = width * height;
  const rawMask = new Float32Array(pixels);
  const expected = new Float32Array(pixels);

  for (let index = 0; index < pixels; index += 1) {
    const prev = previousSupport[index] ?? 0;
    const currentValue = currentAlpha[index] ?? 0;
    const next = followingSupport[index] ?? 0;
    const local = localVisible[index] ?? 0;
    const both = Math.min(prev, next);
    const either = Math.max(prev, next);
    const strongHole = currentValue < 0.12 && local > 0.42 && both > 0.52;
    const movingEdgeHole = currentValue < 0.18 && local > 0.58 && either > 0.70;
    const weakFill = currentValue >= 0.01 && currentValue < 0.34 && local > 0.60 && both > 0.68;
    rawMask[index] = strongHole || movingEdgeHole || weakFill ? 1 : 0;
    expected[index] = Math.max(
      currentValue,
      Math.min(either, Math.max(both, local * 0.88)),
    );
  }

  const expandedMask = dilatePlane(rawMask, width, height, 2);
  const mask = new Uint8Array(pixels);
  const expectedAlpha = new Uint8Array(pixels);
  let maskPixels = 0;
  for (let index = 0; index < pixels; index += 1) {
    const selected = (expandedMask[index] ?? 0) > 0.5;
    mask[index] = selected ? 255 : 0;
    if (selected) maskPixels += 1;
    expectedAlpha[index] = Math.round(Math.max(0, Math.min(1, expected[index] ?? 0)) * 255);
  }
  return { mask, expectedAlpha, width, height, maskPixels };
}
