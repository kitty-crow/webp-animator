export type BrowserModelStatus =
  | 'adapter-ready'
  | 'adapter-ready-assets-required'
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

export interface BrowserExternalDataAsset {
  readonly id: string;
  readonly path: string;
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
}

export interface BrowserModelAsset {
  readonly id: string;
  readonly url: string;
  readonly sha256: string;
  readonly bytes: number;
  readonly licence: string;
  readonly externalData?: readonly BrowserExternalDataAsset[];
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
    family: 'amt',
    label: 'AMT-S',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'scale100', purpose: 'Full-scale arbitrary-timestep interpolation', exportFormat: 'onnx' },
      { id: 'scale075', purpose: '0.75× low-memory interpolation fallback', exportFormat: 'onnx' },
      { id: 'scale050', purpose: '0.5× low-memory interpolation fallback', exportFormat: 'onnx' },
      { id: 'scale025', purpose: '0.25× low-memory interpolation fallback', exportFormat: 'onnx' },
    ],
    notes: 'The browser exporter emits the same four internal scale factors used by the native low-VRAM worker. The UI consumes a manifest containing hashes, sizes and external-data shard locations. Upstream AMT is CC-BY-NC-4.0.',
  },
  {
    family: 'resshift',
    label: 'Multi-Input ResShift',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'flow', purpose: 'Bidirectional endpoint optical-flow estimation', exportFormat: 'onnx' },
      { id: 'extractor', purpose: 'Multi-scale endpoint feature extraction', exportFormat: 'onnx' },
      { id: 'synthesis', purpose: 'Residual-shifting synthesis denoiser', exportFormat: 'onnx' },
    ],
    notes: 'CUDA/CuPy-only softmax splatting and NEDT are replaced by browser implementations. Flow, feature extraction and synthesis are exported offline, while the exact reverse diffusion process remains in strict TypeScript.',
  },
  {
    family: 'mog',
    label: 'Motion-Aware Generative VFI',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'motion', purpose: 'EMA-VFI motion and flow guidance', exportFormat: 'onnx' },
      { id: 'vae-encoder', purpose: 'Endpoint latent and reference-context encoding', exportFormat: 'onnx' },
      { id: 'image-condition', purpose: 'Image conditioning and projection', exportFormat: 'onnx' },
      { id: 'denoiser', purpose: 'Latent video v-prediction step', exportFormat: 'onnx' },
      { id: 'vae-decoder', purpose: 'Reference-aware latent decoding', exportFormat: 'onnx' },
    ],
    notes: 'The browser owns latent warping, 50-step eta-1 DDIM, v-prediction, dynamic rescaling and component offload. The exporter precomputes runtime-invariant empty text conditioning and emits hash-verified external-data shards.',
  },
  {
    family: 'tooncrafter',
    label: 'ToonCrafter',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [
      { id: 'vae-encoder', purpose: 'Endpoint latent and reference-context encoding', exportFormat: 'onnx' },
      { id: 'image-condition', purpose: 'Endpoint image conditioning and projection', exportFormat: 'onnx' },
      { id: 'denoiser', purpose: 'Latent video v-prediction step', exportFormat: 'onnx' },
      { id: 'vae-decoder', purpose: 'Context-aware latent decoding', exportFormat: 'onnx' },
    ],
    notes: 'The browser keeps the native width-dependent DDIM spacing and dynamic rescale policy. Empty text conditioning is exported as runtime-invariant data and large tensors are externalised for lazy verified caching.',
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
      { id: 'image-propagation', purpose: 'Flow-guided image propagation', exportFormat: 'onnx' },
      { id: 'transformer-inpaint', purpose: 'Feature propagation and temporal transformer inpainting', exportFormat: 'onnx' },
    ],
    notes: 'Mask audit and alpha restoration are browser-native. The neural stages require user-supplied exported assets because upstream ProPainter weights are non-commercial and must not be silently redistributed.',
  },
  {
    family: 'eden',
    label: 'EDEN',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [{ id: 'generator', purpose: 'Two-step Euler structural midpoint generation and decode', exportFormat: 'onnx' }],
    notes: 'The browser exporter wraps the native t=0→0.75→1 Euler solve and decoder into one ONNX graph. Midpoint crop, alpha composition, provider fallback and persistent-worker orchestration are strict TypeScript.',
  },
  {
    family: 'speed',
    label: 'SPEED',
    status: 'adapter-ready-assets-required',
    preferredProvider: 'webgpu',
    fallbackProvider: 'wasm',
    assets: [],
    components: [{ id: 'generator', purpose: 'Noise-conditioned structural midpoint generation', exportFormat: 'onnx' }],
    notes: 'The browser exporter emits the SPEED midpoint graph. The adapter reproduces the native 1.0/0.75/0.5/0.375 low-memory retry scales automatically and runs inside the persistent worker.',
  },
];

export function browserModelDefinition(family: BrowserModelFamily): BrowserModelDefinition {
  const definition = BROWSER_MODEL_CATALOG.find((candidate) => candidate.family === family);
  if (!definition) throw new Error(`Unknown browser model family: ${family}`);
  return definition;
}
