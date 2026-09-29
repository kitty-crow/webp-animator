import type { HardwareProfile, WebGlCapability, WebGpuCapability } from './types.js';

const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
  0x03, 0x02, 0x01, 0x00,
  0x0a, 0x08, 0x01, 0x06, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0x0b,
]);

interface NavigatorMemory extends Navigator {
  readonly deviceMemory?: number;
  readonly userAgentData?: { readonly mobile?: boolean };
}

interface PerformanceMemory extends Performance {
  readonly memory?: { readonly jsHeapSizeLimit?: number };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function probeWebGl2(): WebGlCapability {
  try {
    const canvas: OffscreenCanvas | HTMLCanvasElement | null = typeof OffscreenCanvas === 'function'
      ? new OffscreenCanvas(2, 2)
      : typeof document !== 'undefined'
        ? document.createElement('canvas')
        : null;
    if (!canvas) return { available: false, renderer: null, maxTextureSize: 0, error: null };
    const context = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    });
    if (!context || !('getParameter' in context)) {
      return { available: false, renderer: null, maxTextureSize: 0, error: null };
    }
    const gl = context as WebGL2RenderingContext;
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    const rawRenderer: unknown = debug
      ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER);
    return {
      available: true,
      renderer: typeof rawRenderer === 'string' ? rawRenderer : 'WebGL2 adapter',
      maxTextureSize: Number(gl.getParameter(gl.MAX_TEXTURE_SIZE) || 0),
      error: null,
    };
  } catch (error: unknown) {
    return { available: false, renderer: null, maxTextureSize: 0, error: errorMessage(error) };
  }
}

async function probeWebGpu(): Promise<WebGpuCapability> {
  if (!navigator.gpu?.requestAdapter) {
    return { available: false, adapter: null, info: null, limits: null, error: null };
  }

  try {
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { available: false, adapter: null, info: null, limits: null, error: null };
    }

    const info = adapter.info;
    return {
      available: true,
      adapter,
      info: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      limits: {
        maxBufferSize: Number(adapter.limits.maxBufferSize),
        maxStorageBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
        maxComputeWorkgroupsPerDimension: Number(adapter.limits.maxComputeWorkgroupsPerDimension),
      },
      error: null,
    };
  } catch (error: unknown) {
    return { available: false, adapter: null, info: null, limits: null, error: errorMessage(error) };
  }
}

export async function detectHardware(): Promise<HardwareProfile> {
  const nav = navigator as NavigatorMemory;
  const perf = performance as PerformanceMemory;
  const globals = globalThis as typeof globalThis & { readonly ImageDecoder?: unknown };
  const logicalCores = Math.max(1, Number(nav.hardwareConcurrency || 1));
  const deviceMemoryGiB = Number(nav.deviceMemory || 0) || null;
  const heapLimitBytes = Number(perf.memory?.jsHeapSizeLimit || 0);
  const heapLimitMiB = heapLimitBytes > 0 ? Math.floor(heapLimitBytes / 1048576) : null;
  const mobile = Boolean(nav.userAgentData?.mobile) || /Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent);
  const crossOriginIsolated = globalThis.crossOriginIsolated === true;
  const sharedMemory = crossOriginIsolated && typeof SharedArrayBuffer === 'function';
  const workerSupport = typeof Worker === 'function';
  const wasm = typeof WebAssembly === 'object';
  const wasmSimd = wasm && WebAssembly.validate(SIMD_PROBE);
  const webgl2 = probeWebGl2();
  const webgpu = await probeWebGpu();

  let memoryTier: HardwareProfile['memoryTier'] = 'low';
  if (!mobile && ((deviceMemoryGiB !== null && deviceMemoryGiB >= 8) || (heapLimitMiB !== null && heapLimitMiB >= 3072))) {
    memoryTier = 'high';
  } else if (
    (deviceMemoryGiB !== null && deviceMemoryGiB >= 4) ||
    (heapLimitMiB !== null && heapLimitMiB >= 1536) ||
    (!mobile && logicalCores >= 8)
  ) {
    memoryTier = 'medium';
  }

  let memoryBudgetMiB: number;
  if (deviceMemoryGiB !== null) {
    memoryBudgetMiB = Math.floor(Math.min(2048, Math.max(256, deviceMemoryGiB * 1024 * 0.20)));
  } else if (heapLimitMiB !== null) {
    memoryBudgetMiB = Math.floor(Math.min(2048, Math.max(256, heapLimitMiB * 0.35)));
  } else {
    memoryBudgetMiB = mobile ? 320 : memoryTier === 'high' ? 1280 : memoryTier === 'medium' ? 768 : 384;
  }

  const maxWorkersByMemory = memoryTier === 'low' ? 2 : memoryTier === 'medium' ? 6 : 12;
  const workerCount = workerSupport
    ? clamp(logicalCores > 2 ? logicalCores - 1 : 1, 1, Math.min(16, maxWorkersByMemory))
    : 0;

  return {
    logicalCores,
    workerCount,
    deviceMemoryGiB,
    heapLimitMiB,
    memoryTier,
    memoryBudgetMiB,
    mobile,
    sharedMemory,
    crossOriginIsolated,
    workerSupport,
    wasm,
    wasmSimd,
    webgpu,
    webgl2,
    offscreenCanvas: typeof OffscreenCanvas === 'function',
    imageDecoder: typeof globals.ImageDecoder === 'function',
  };
}

export function gpuLabel(profile: HardwareProfile): string {
  if (profile.webgpu.available) {
    const info = profile.webgpu.info;
    const detail = info?.description || info?.device || info?.architecture || info?.vendor;
    return detail ? `WebGPU · ${detail}` : 'WebGPU adapter';
  }
  if (profile.webgl2.available) return `WebGL2 · ${profile.webgl2.renderer ?? 'adapter'}`;
  return 'No browser GPU API';
}
