import * as ort from 'onnxruntime-web/webgpu';

import type { HardwareProfile } from '../types.js';
import type { BrowserModelAsset } from './catalog.js';
import { loadModelBytes } from './model-store.js';

export type ModelExecutionProvider = 'webgpu' | 'wasm';

export interface LoadedOrtModel {
  readonly session: ort.InferenceSession;
  readonly provider: ModelExecutionProvider;
  readonly asset: BrowserModelAsset;
}

let environmentConfigured = false;

function configureEnvironment(profile: HardwareProfile): void {
  if (environmentConfigured) return;
  environmentConfigured = true;

  ort.env.wasm.numThreads = profile.sharedMemory ? Math.max(1, profile.workerCount) : 1;
  ort.env.wasm.simd = profile.wasmSimd;
  ort.env.wasm.wasmPaths = new URL('../../vendor/', import.meta.url).href;
}

async function createSession(
  bytes: Uint8Array,
  provider: ModelExecutionProvider,
): Promise<ort.InferenceSession> {
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: provider === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
    executionMode: 'sequential',
  };
  return ort.InferenceSession.create(bytes, options);
}

export async function loadOrtModel(
  asset: BrowserModelAsset,
  profile: HardwareProfile,
): Promise<LoadedOrtModel> {
  configureEnvironment(profile);
  const bytes = await loadModelBytes(asset);

  if (profile.webgpu.available) {
    try {
      return {
        session: await createSession(bytes, 'webgpu'),
        provider: 'webgpu',
        asset,
      };
    } catch (error: unknown) {
      console.warn(`WebGPU could not initialise ${asset.id}; retrying with WASM.`, error);
    }
  }

  return {
    session: await createSession(bytes, 'wasm'),
    provider: 'wasm',
    asset,
  };
}

export function floatTensor(data: Float32Array, dimensions: readonly number[]): ort.Tensor {
  return new ort.Tensor('float32', data, [...dimensions]);
}

export function tensorFloatData(tensor: ort.Tensor): Float32Array {
  if (!(tensor.data instanceof Float32Array)) {
    throw new Error(`Expected float32 ONNX output, received ${tensor.type}.`);
  }
  return tensor.data;
}
