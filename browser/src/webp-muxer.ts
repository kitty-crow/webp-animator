import { imageDataToWebpBlob } from './frame-codec.js';

const encoder = new TextEncoder();

function fourCc(text: string): Uint8Array {
  return encoder.encode(text);
}

function writeU16(view: Uint8Array, offset: number, value: number): void {
  view[offset] = value & 0xff;
  view[offset + 1] = (value >>> 8) & 0xff;
}

function writeU24(view: Uint8Array, offset: number, value: number): void {
  view[offset] = value & 0xff;
  view[offset + 1] = (value >>> 8) & 0xff;
  view[offset + 2] = (value >>> 16) & 0xff;
}

function writeU32(view: Uint8Array, offset: number, value: number): void {
  view[offset] = value & 0xff;
  view[offset + 1] = (value >>> 8) & 0xff;
  view[offset + 2] = (value >>> 16) & 0xff;
  view[offset + 3] = (value >>> 24) & 0xff;
}

function concat(parts: readonly Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function chunk(type: string, payload: Uint8Array): Uint8Array {
  const paddedLength = payload.byteLength + (payload.byteLength & 1);
  const output = new Uint8Array(8 + paddedLength);
  output.set(fourCc(type), 0);
  writeU32(output, 4, payload.byteLength);
  output.set(payload, 8);
  return output;
}

function readAscii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + length));
}

function readU32(bytes: Uint8Array, offset: number): number {
  const b0 = bytes[offset];
  const b1 = bytes[offset + 1];
  const b2 = bytes[offset + 2];
  const b3 = bytes[offset + 3];
  if (b0 === undefined || b1 === undefined || b2 === undefined || b3 === undefined) throw new Error('Unexpected end of WebP container.');
  return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) >>> 0;
}

function parseStillChunks(bytes: Uint8Array): readonly Uint8Array[] {
  if (bytes.byteLength < 20 || readAscii(bytes, 0, 4) !== 'RIFF' || readAscii(bytes, 8, 4) !== 'WEBP') {
    throw new Error('Canvas returned an invalid WebP file.');
  }
  const output: Uint8Array[] = [];
  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const type = readAscii(bytes, offset, 4);
    const size = readU32(bytes, offset + 4);
    const end = offset + 8 + size;
    if (end > bytes.byteLength) break;
    if (type === 'ALPH' || type === 'VP8 ' || type === 'VP8L') output.push(bytes.slice(offset, end + (size & 1)));
    offset = end + (size & 1);
  }
  if (!output.some((item) => {
    const type = readAscii(item, 0, 4);
    return type === 'VP8 ' || type === 'VP8L';
  })) throw new Error('Encoded frame contains no VP8 payload.');
  return output;
}

function hasTransparency(frame: ImageData): boolean {
  for (let index = 3; index < frame.data.length; index += 4) {
    if (frame.data[index] !== 255) return true;
  }
  return false;
}

function vp8x(width: number, height: number, alpha: boolean): Uint8Array {
  const payload = new Uint8Array(10);
  payload[0] = 0x02 | (alpha ? 0x10 : 0x00);
  writeU24(payload, 4, width - 1);
  writeU24(payload, 7, height - 1);
  return chunk('VP8X', payload);
}

function anim(loop: number): Uint8Array {
  const payload = new Uint8Array(6);
  writeU32(payload, 0, 0);
  writeU16(payload, 4, Math.max(0, Math.min(0xffff, Math.round(loop))));
  return chunk('ANIM', payload);
}

function anmf(width: number, height: number, duration: number, encoded: readonly Uint8Array[]): Uint8Array {
  const header = new Uint8Array(16);
  writeU24(header, 0, 0);
  writeU24(header, 3, 0);
  writeU24(header, 6, width - 1);
  writeU24(header, 9, height - 1);
  writeU24(header, 12, Math.max(1, Math.min(0xffffff, Math.round(duration))));
  header[15] = 0x02;
  return chunk('ANMF', concat([header, ...encoded]));
}

export async function encodeAnimatedWebp(
  frames: readonly ImageData[],
  options: { readonly duration: number; readonly durations: readonly (number | null)[]; readonly loop: number; readonly quality: number },
  onProgress: (current: number, total: number) => void,
): Promise<Blob> {
  const first = frames[0];
  if (!first) throw new Error('No frames to encode.');
  const width = first.width;
  const height = first.height;
  if (!frames.every((frame) => frame.width === width && frame.height === height)) throw new Error('All animation frames must share one canvas size.');

  const alpha = frames.some(hasTransparency);
  const frameChunks: Uint8Array[] = [];
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    if (!frame) throw new Error(`Frame ${index} is missing.`);
    onProgress(index, frames.length);
    const still = await imageDataToWebpBlob(frame, options.quality);
    const chunks = parseStillChunks(new Uint8Array(await still.arrayBuffer()));
    frameChunks.push(anmf(width, height, options.durations[index] ?? options.duration, chunks));
  }

  const body = concat([fourCc('WEBP'), vp8x(width, height, alpha), anim(options.loop), ...frameChunks]);
  const riff = new Uint8Array(8 + body.byteLength);
  riff.set(fourCc('RIFF'), 0);
  writeU32(riff, 4, body.byteLength);
  riff.set(body, 8);
  onProgress(frames.length, frames.length);
  return new Blob([riff], { type: 'image/webp' });
}
