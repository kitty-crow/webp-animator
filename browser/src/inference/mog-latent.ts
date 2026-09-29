import type { NchwTensor } from './softsplat.js';
import { resizeNchwBilinear } from './tensor-resize.js';

export interface NcthwTensor {
  readonly data: Float32Array;
  readonly batch: number;
  readonly channels: number;
  readonly frames: number;
  readonly height: number;
  readonly width: number;
}

export interface MogMotion {
  readonly flow: NchwTensor;
  readonly mask: NchwTensor;
}

function ncthwIndex(
  batch: number,
  channel: number,
  frame: number,
  y: number,
  x: number,
  channels: number,
  frames: number,
  height: number,
  width: number,
): number {
  return ((((batch * channels + channel) * frames + frame) * height + y) * width) + x;
}

function nchwIndex(
  batch: number,
  channel: number,
  y: number,
  x: number,
  channels: number,
  height: number,
  width: number,
): number {
  return (((batch * channels + channel) * height + y) * width) + x;
}

function validateVideo(video: NcthwTensor): void {
  if (video.batch !== 1) throw new Error('Browser MoG currently supports one video per inference job.');
  if (video.channels <= 0 || video.frames < 2 || video.height <= 0 || video.width <= 0) throw new Error('MoG video shape is invalid.');
  const expected = video.batch * video.channels * video.frames * video.height * video.width;
  if (video.data.length !== expected) throw new Error(`MoG video length mismatch: expected ${expected}, received ${video.data.length}.`);
}

function validateImagePair(first: NchwTensor, second: NchwTensor): void {
  if (first.batch !== 1 || second.batch !== 1) throw new Error('Browser MoG endpoint batch must be one.');
  if (first.channels !== second.channels || first.height !== second.height || first.width !== second.width) throw new Error('MoG endpoint shapes differ.');
}

export function makeMogEndpointVideo(first: NchwTensor, second: NchwTensor, frames = 16): NcthwTensor {
  validateImagePair(first, second);
  if (!Number.isInteger(frames) || frames < 2) throw new Error('MoG frame count must be an integer of at least two.');
  const output = new Float32Array(first.batch * first.channels * frames * first.height * first.width);
  const split = Math.floor(frames / 2);
  for (let channel = 0; channel < first.channels; channel += 1) {
    for (let frame = 0; frame < frames; frame += 1) {
      const source = frame < split ? first : second;
      for (let y = 0; y < first.height; y += 1) {
        for (let x = 0; x < first.width; x += 1) {
          output[ncthwIndex(0, channel, frame, y, x, first.channels, frames, first.height, first.width)] =
            source.data[nchwIndex(0, channel, y, x, source.channels, source.height, source.width)] ?? 0;
        }
      }
    }
  }
  return {
    data: output,
    batch: 1,
    channels: first.channels,
    frames,
    height: first.height,
    width: first.width,
  };
}

function normalNoise(length: number): Float32Array {
  const random = new Uint32Array(Math.max(2, Math.ceil(length / 2) * 2));
  for (let offset = 0; offset < random.length; offset += 16_384) {
    crypto.getRandomValues(random.subarray(offset, Math.min(random.length, offset + 16_384)));
  }
  const output = new Float32Array(length);
  let destination = 0;
  for (let source = 0; source < random.length && destination < length; source += 2) {
    const u0 = ((random[source] ?? 0) + 1) / 4_294_967_297;
    const u1 = ((random[source + 1] ?? 0) + 1) / 4_294_967_297;
    const radius = Math.sqrt(-2 * Math.log(u0));
    const angle = 2 * Math.PI * u1;
    output[destination] = radius * Math.cos(angle);
    destination += 1;
    if (destination < length) {
      output[destination] = radius * Math.sin(angle);
      destination += 1;
    }
  }
  return output;
}

export function sampleMogPosterior(parameters: NcthwTensor, scaleFactor: number): NcthwTensor {
  validateVideo(parameters);
  if (parameters.channels % 2 !== 0) throw new Error('MoG VAE posterior parameters must contain mean and log-variance channel halves.');
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) throw new Error('MoG VAE scale factor must be positive.');
  const channels = parameters.channels / 2;
  const output = new Float32Array(parameters.batch * channels * parameters.frames * parameters.height * parameters.width);
  const noise = normalNoise(output.length);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let frame = 0; frame < parameters.frames; frame += 1) {
      for (let y = 0; y < parameters.height; y += 1) {
        for (let x = 0; x < parameters.width; x += 1) {
          const meanIndex = ncthwIndex(0, channel, frame, y, x, parameters.channels, parameters.frames, parameters.height, parameters.width);
          const logvarIndex = ncthwIndex(0, channel + channels, frame, y, x, parameters.channels, parameters.frames, parameters.height, parameters.width);
          const destination = ncthwIndex(0, channel, frame, y, x, channels, parameters.frames, parameters.height, parameters.width);
          const mean = parameters.data[meanIndex] ?? 0;
          const logvar = Math.max(-30, Math.min(20, parameters.data[logvarIndex] ?? 0));
          const standardDeviation = Math.exp(0.5 * logvar);
          output[destination] = (mean + standardDeviation * (noise[destination] ?? 0)) * scaleFactor;
        }
      }
    }
  }
  return {
    data: output,
    batch: parameters.batch,
    channels,
    frames: parameters.frames,
    height: parameters.height,
    width: parameters.width,
  };
}

function frameAsNchw(video: NcthwTensor, frame: number): NchwTensor {
  validateVideo(video);
  if (!Number.isInteger(frame) || frame < 0 || frame >= video.frames) throw new Error('MoG video frame index is invalid.');
  const data = new Float32Array(video.batch * video.channels * video.height * video.width);
  for (let channel = 0; channel < video.channels; channel += 1) {
    for (let y = 0; y < video.height; y += 1) {
      for (let x = 0; x < video.width; x += 1) {
        data[nchwIndex(0, channel, y, x, video.channels, video.height, video.width)] =
          video.data[ncthwIndex(0, channel, frame, y, x, video.channels, video.frames, video.height, video.width)] ?? 0;
      }
    }
  }
  return { data, batch: 1, channels: video.channels, height: video.height, width: video.width };
}

function repeatImage(image: NchwTensor, repetitions: number): NchwTensor {
  if (image.batch !== 1) throw new Error('MoG latent repeat expects a single image batch.');
  const plane = image.channels * image.height * image.width;
  const data = new Float32Array(repetitions * plane);
  for (let index = 0; index < repetitions; index += 1) data.set(image.data, index * plane);
  return { data, batch: repetitions, channels: image.channels, height: image.height, width: image.width };
}

function sliceChannels(input: NchwTensor, start: number, count: number): NchwTensor {
  if (start < 0 || count <= 0 || start + count > input.channels) throw new Error('MoG motion channel slice is invalid.');
  const pixels = input.height * input.width;
  const data = new Float32Array(input.batch * count * pixels);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let channel = 0; channel < count; channel += 1) {
      const source = (batch * input.channels + start + channel) * pixels;
      const destination = (batch * count + channel) * pixels;
      data.set(input.data.subarray(source, source + pixels), destination);
    }
  }
  return { data, batch: input.batch, channels: count, height: input.height, width: input.width };
}

function multiply(input: NchwTensor, scalar: number): NchwTensor {
  const data = new Float32Array(input.data.length);
  for (let index = 0; index < data.length; index += 1) data[index] = (input.data[index] ?? 0) * scalar;
  return { ...input, data };
}

function borderSample(input: NchwTensor, batch: number, channel: number, y: number, x: number): number {
  const clampedX = Math.max(0, Math.min(input.width - 1, x));
  const clampedY = Math.max(0, Math.min(input.height - 1, y));
  return input.data[nchwIndex(batch, channel, clampedY, clampedX, input.channels, input.height, input.width)] ?? 0;
}

/** Matches emavfi/model/warplayer.py: [dx,dy] flow, border padding, align_corners=true. */
export function mogBackwarp(input: NchwTensor, flow: NchwTensor): NchwTensor {
  if (flow.channels !== 2 || flow.batch !== input.batch || flow.height !== input.height || flow.width !== input.width) {
    throw new Error('MoG backwarp flow shape is invalid.');
  }
  const data = new Float32Array(input.data.length);
  for (let batch = 0; batch < input.batch; batch += 1) {
    for (let y = 0; y < input.height; y += 1) {
      for (let x = 0; x < input.width; x += 1) {
        const dx = flow.data[nchwIndex(batch, 0, y, x, 2, input.height, input.width)] ?? 0;
        const dy = flow.data[nchwIndex(batch, 1, y, x, 2, input.height, input.width)] ?? 0;
        const sourceX = x + dx;
        const sourceY = y + dy;
        const x0 = Math.floor(sourceX);
        const y0 = Math.floor(sourceY);
        const x1 = x0 + 1;
        const y1 = y0 + 1;
        const wx = sourceX - x0;
        const wy = sourceY - y0;
        for (let channel = 0; channel < input.channels; channel += 1) {
          const top = borderSample(input, batch, channel, y0, x0) * (1 - wx)
            + borderSample(input, batch, channel, y0, x1) * wx;
          const bottom = borderSample(input, batch, channel, y1, x0) * (1 - wx)
            + borderSample(input, batch, channel, y1, x1) * wx;
          data[nchwIndex(batch, channel, y, x, input.channels, input.height, input.width)] = top * (1 - wy) + bottom * wy;
        }
      }
    }
  }
  return { ...input, data };
}

export function warpMogLatentGuidance(latents: NcthwTensor, motion: MogMotion): NcthwTensor {
  validateVideo(latents);
  const interior = latents.frames - 2;
  if (motion.flow.batch !== interior || motion.flow.channels !== 4 || motion.mask.batch !== interior || motion.mask.channels !== 1) {
    throw new Error('MoG motion guidance does not match the latent temporal length.');
  }
  const scale = latents.height / motion.flow.height;
  const resizedFlow = multiply(resizeNchwBilinear(motion.flow, latents.height, latents.width), scale);
  const resizedMask = resizeNchwBilinear(motion.mask, latents.height, latents.width);
  const first = repeatImage(frameAsNchw(latents, 0), interior);
  const second = repeatImage(frameAsNchw(latents, latents.frames - 1), interior);
  const warpedFirst = mogBackwarp(first, sliceChannels(resizedFlow, 0, 2));
  const warpedSecond = mogBackwarp(second, sliceChannels(resizedFlow, 2, 2));
  const output = new Float32Array(latents.batch * latents.channels * interior * latents.height * latents.width);

  for (let frame = 0; frame < interior; frame += 1) {
    for (let channel = 0; channel < latents.channels; channel += 1) {
      for (let y = 0; y < latents.height; y += 1) {
        for (let x = 0; x < latents.width; x += 1) {
          const source = nchwIndex(frame, channel, y, x, latents.channels, latents.height, latents.width);
          const maskIndex = nchwIndex(frame, 0, y, x, 1, latents.height, latents.width);
          const weight = resizedMask.data[maskIndex] ?? 0;
          output[ncthwIndex(0, channel, frame, y, x, latents.channels, interior, latents.height, latents.width)] =
            (warpedFirst.data[source] ?? 0) * weight + (warpedSecond.data[source] ?? 0) * (1 - weight);
        }
      }
    }
  }

  return { data: output, batch: 1, channels: latents.channels, frames: interior, height: latents.height, width: latents.width };
}

export function makeMogConcatLatent(latents: NcthwTensor, interior: NcthwTensor): NcthwTensor {
  validateVideo(latents);
  if (
    interior.batch !== latents.batch || interior.channels !== latents.channels ||
    interior.frames !== latents.frames - 2 || interior.height !== latents.height || interior.width !== latents.width
  ) throw new Error('MoG warped interior latent shape is invalid.');
  const data = new Float32Array(latents.data);
  for (let channel = 0; channel < latents.channels; channel += 1) {
    for (let frame = 0; frame < interior.frames; frame += 1) {
      for (let y = 0; y < latents.height; y += 1) {
        for (let x = 0; x < latents.width; x += 1) {
          data[ncthwIndex(0, channel, frame + 1, y, x, latents.channels, latents.frames, latents.height, latents.width)] =
            interior.data[ncthwIndex(0, channel, frame, y, x, interior.channels, interior.frames, interior.height, interior.width)] ?? 0;
        }
      }
    }
  }
  return { ...latents, data };
}

export function selectMogFrames(input: NcthwTensor, frameIndices: readonly number[]): NcthwTensor {
  validateVideo(input);
  if (frameIndices.length === 0) throw new Error('MoG temporal selection cannot be empty.');
  const data = new Float32Array(input.batch * input.channels * frameIndices.length * input.height * input.width);
  for (let destinationFrame = 0; destinationFrame < frameIndices.length; destinationFrame += 1) {
    const sourceFrame = frameIndices[destinationFrame];
    if (sourceFrame === undefined || !Number.isInteger(sourceFrame) || sourceFrame < 0 || sourceFrame >= input.frames) throw new Error('MoG temporal selection contains an invalid index.');
    for (let channel = 0; channel < input.channels; channel += 1) {
      for (let y = 0; y < input.height; y += 1) {
        for (let x = 0; x < input.width; x += 1) {
          data[ncthwIndex(0, channel, destinationFrame, y, x, input.channels, frameIndices.length, input.height, input.width)] =
            input.data[ncthwIndex(0, channel, sourceFrame, y, x, input.channels, input.frames, input.height, input.width)] ?? 0;
        }
      }
    }
  }
  return { ...input, data, frames: frameIndices.length };
}

export function replaceMogFrames(target: NcthwTensor, source: NcthwTensor, targetStart: number, sourceStart: number, count: number): NcthwTensor {
  validateVideo(target);
  validateVideo(source);
  if (source.channels !== target.channels || source.height !== target.height || source.width !== target.width) throw new Error('MoG decoder correction tensors differ spatially.');
  if (targetStart < 0 || sourceStart < 0 || count < 0 || targetStart + count > target.frames || sourceStart + count > source.frames) throw new Error('MoG decoder correction range is invalid.');
  const data = new Float32Array(target.data);
  for (let offset = 0; offset < count; offset += 1) {
    for (let channel = 0; channel < target.channels; channel += 1) {
      for (let y = 0; y < target.height; y += 1) {
        for (let x = 0; x < target.width; x += 1) {
          data[ncthwIndex(0, channel, targetStart + offset, y, x, target.channels, target.frames, target.height, target.width)] =
            source.data[ncthwIndex(0, channel, sourceStart + offset, y, x, source.channels, source.frames, source.height, source.width)] ?? 0;
        }
      }
    }
  }
  return { ...target, data };
}

export { normalNoise as mogNormalNoise };
