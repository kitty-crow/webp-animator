export type BrowserModelStatus =
  | 'adapter-ready'
  | 'component-export-required'
  | 'export-required'
  | 'licence-gated';

export type BrowserModelFamily =
  | 'rife'
  | 'resshift'
  | 'mog'
  | 'tooncrafter'
  | 'propainter'
  | 'amt'
  | 'eden'
  | 'speed';

export interface BrowserModelAsset {
  readonly id: string;
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
  readonly licence: string;
}

export interface BrowserModelComponent {
  readonly id: string;
  readonly purpose: string;
  readonly exportFormat: 'onnx';
}

export interface BrowserModelDefinition {
  readonly family: BrowserModelFamily;
  readonly label: string;
  readonly status: BrowserModelStatus;
  readonly preferredProvider: 'webgpu';
  readonly fallbackProvider: 'wasm';
  readonly assets: readonly BrowserModelAsset[];
  readonly components: readonly BrowserModelComponent[];
  readonly notes: string;
}

const RIFE_425_ONNX: BrowserModelAsset = {
  id: 'rife-v4.25-v2',
  url: 'https://huggingface.co/notaneimu/onnx-image-models/resolve/main/rife_v4.25_v2.onnx?download=true',
  sha256: '65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584',
  bytes: 22_748_049,
  licence: 'MIT',
};

export const BROWSER_MODEL_CATALOG: readonly BrowserModelDefinition[] = [
  {
    family: 'rife',
    label: 'Practical-RIFE 4.25',
    status: 'adapter-ready',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [RIFE_425_ONNX],
    components: [
      { id: 'ifnet', purpose: 'Arbitrary-timestep frame interpolation', exportFormat: 'onnx' },
    ],
    notes: 'Exact 4.25 generation used by the native worker. The browser adapter accepts the v2 7-channel ONNX graph and runs alpha through the same network when required.',
  },
  {
    family: 'resshift',
    label: 'Multi-Input ResShift',
    status: 'component-export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'flow', purpose: 'Endpoint optical-flow estimation', exportFormat: 'onnx' },
      { id: 'reverse-denoiser', purpose: 'Residual-shifting reverse diffusion step', exportFormat: 'onnx' },
    ],
    notes: 'The released path contains CUDA/CuPy warping. Browser parity requires replacing those warps with ONNX GridSample-compatible graph operations and keeping the reverse-process loop in TypeScript.',
  },
  {
    family: 'mog',
    label: 'Motion-Aware Generative VFI',
    status: 'component-export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'ema-vfi', purpose: 'Motion/flow guidance', exportFormat: 'onnx' },
      { id: 'vae-encoder', purpose: 'Frame-to-latent encoding', exportFormat: 'onnx' },
      { id: 'video-denoiser', purpose: 'Latent video denoising step', exportFormat: 'onnx' },
      { id: 'vae-decoder', purpose: 'Latent-to-frame decoding', exportFormat: 'onnx' },
    ],
    notes: 'The DDIM loop belongs in TypeScript. Splitting the graph allows component offload and avoids requiring the whole PyTorch model in memory at once.',
  },
  {
    family: 'tooncrafter',
    label: 'ToonCrafter',
    status: 'component-export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'vae-encoder', purpose: 'Endpoint latent and reference-context encoding', exportFormat: 'onnx' },
      { id: 'image-embedder', purpose: 'Endpoint image conditioning', exportFormat: 'onnx' },
      { id: 'image-projector', purpose: 'Image-conditioning projection', exportFormat: 'onnx' },
      { id: 'video-denoiser', purpose: 'Latent video denoising step', exportFormat: 'onnx' },
      { id: 'vae-decoder', purpose: 'Context-aware latent decoding', exportFormat: 'onnx' },
    ],
    notes: 'Empty text conditioning can be precomputed. The large checkpoint should be split into ONNX external-data shards and cached locally rather than bundled into the Pages repository.',
  },
  {
    family: 'propainter',
    label: 'ProPainter temporal repair',
    status: 'licence-gated',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'raft', purpose: 'Bidirectional optical flow', exportFormat: 'onnx' },
      { id: 'flow-completion', purpose: 'Recurrent flow completion', exportFormat: 'onnx' },
      { id: 'image-propagation', purpose: 'Flow-guided feature propagation', exportFormat: 'onnx' },
      { id: 'transformer-inpaint', purpose: 'Temporal transformer inpainting', exportFormat: 'onnx' },
    ],
    notes: 'The mask audit/dilation is suitable for TypeScript/WebGPU. Upstream ProPainter weights are non-commercial, so the browser build must not silently redistribute them without respecting that licence.',
  },
  {
    family: 'amt',
    label: 'AMT',
    status: 'export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [{ id: 'interpolator', purpose: 'Arbitrary-timestep interpolation', exportFormat: 'onnx' }],
    notes: 'A single-graph ONNX adapter should be sufficient once the current checkpoint is exported and verified.',
  },
  {
    family: 'eden',
    label: 'EDEN',
    status: 'export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [{ id: 'interpolator', purpose: 'Frame generation/interpolation', exportFormat: 'onnx' }],
    notes: 'Requires a model-specific export audit before the browser adapter can be enabled.',
  },
  {
    family: 'speed',
    label: 'SPEED',
    status: 'export-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [{ id: 'interpolator', purpose: 'Frame interpolation', exportFormat: 'onnx' }],
    notes: 'Requires a model-specific export audit before the browser adapter can be enabled.',
  },
];

export function browserModelDefinition(family: BrowserModelFamily): BrowserModelDefinition {
  const definition = BROWSER_MODEL_CATALOG.find((candidate) => candidate.family === family);
  if (!definition) throw new Error(`Unknown browser model family: ${family}`);
  return definition;
}
