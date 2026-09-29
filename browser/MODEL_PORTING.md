# Browser model migration

The browser build does not attempt to ship CPython plus PyTorch inside WebAssembly. The target architecture is:

1. Keep orchestration, schedules, image preparation, mask logic and memory policy in strict TypeScript.
2. Export neural components to ONNX.
3. Run ONNX through ONNX Runtime Web, preferring WebGPU and falling back to WASM automatically.
4. Download model weights only when the user chooses the corresponding model, verify their SHA-256, then cache them locally.
5. Split large generative pipelines into independently loadable components so low-memory devices can evict one graph before loading the next.

## Why not compile the current Python stack directly?

Pyodide is the closest equivalent to "Emscripten for Python": it is CPython compiled to WebAssembly/Emscripten, and `pyodide-build` can build compatible Python extension wheels. It is useful when a browser feature genuinely depends on Python semantics or a pure-Python package.

It is not a replacement for exporting these PyTorch models. A Pyodide build would still need a browser-compatible PyTorch implementation and GPU kernels. CUDA and CuPy code would not become WebGPU merely because the Python interpreter was compiled to WASM. It would also add a large Python runtime before any model weights were loaded.

For this project, ONNX Runtime Web is the lower and more appropriate boundary. The same TypeScript pipeline can ask for `webgpu` first and `wasm` second without the user choosing a backend.

## Current status

### RIFE 4.25

Browser adapter is implemented.

- Uses the Practical-RIFE 4.25 v2 ONNX graph.
- Model asset is pinned by SHA-256 and byte size.
- Lazy model download and Cache Storage reuse.
- WebGPU first, WASM fallback.
- Arbitrary interpolation ratio.
- Alpha interpolation mirrors the native worker by running alpha through RIFE when either endpoint contains transparency.
- 2x, 4x and 8x interpolation are wired into the static page.

### Multi-Input ResShift

The browser reverse-process scheduler is implemented in TypeScript from the upstream equations. The remaining neural/warping components are:

- RAFT endpoint flow.
- Multi-scale feature warper.
- Synthesis network producing `predicted_x0` for each reverse step.

The current upstream warper contains a CUDA/CuPy dependency. That must be replaced by portable ONNX/WebGPU operations, preferably GridSample-compatible graph operations, rather than trying to load CuPy in the browser.

The stochastic reverse loop stays in TypeScript, which means the synthesis graph is invoked once per diffusion step and can use either WebGPU or WASM.

### MoG

The generic deterministic/stochastic DDIM scheduler is implemented in TypeScript. Planned ONNX components:

1. EMA-VFI flow guidance.
2. VAE encoder.
3. Motion-guided video denoiser.
4. VAE decoder.

The DDIM loop remains TypeScript. This allows component offload between stages and avoids loading the full PyTorch pipeline into memory.

### ToonCrafter

Uses the same TypeScript DDIM foundation. Planned ONNX components:

1. VAE encoder.
2. Image embedder.
3. Image projector.
4. Video denoiser.
5. Context-aware VAE decoder.

Empty text conditioning can be generated once and cached. Large checkpoints should use ONNX external-data shards rather than being committed to the Pages repository.

### ProPainter

The deterministic alpha-hole audit and dilation have been ported to TypeScript, including an O(n) separable max-filter implementation rather than a quadratic-radius implementation.

Remaining model components:

1. Bidirectional RAFT flow.
2. Recurrent flow completion.
3. Flow-guided image propagation.
4. Transformer inpainting.

ProPainter's upstream weights are licence-restricted for non-commercial use. The browser catalogue therefore marks it as licence-gated. The site must not silently redistribute those weights without satisfying their licence terms.

## Model asset policy

Model binaries are not committed to GitHub Pages. A browser model asset must provide:

- a stable HTTPS URL with CORS support;
- an expected byte size;
- an expected SHA-256 digest;
- a declared licence/provenance;
- a component ID and model-family association.

The browser verifies size and SHA-256 before creating an inference session. A corrupt cached entry is discarded and fetched again.

## Backend policy

There is no user-facing backend selector.

- Model sessions try WebGPU when the hardware pre-flight says WebGPU is available.
- If session creation fails, the same model is retried through WASM.
- WASM threading follows the same SharedArrayBuffer/cross-origin-isolation detection used elsewhere in the browser build.
- WebGL2 remains a useful raster fallback, but ONNX model compute targets WebGPU/WASM because WebGPU is the modern general-purpose GPU path.

## CI invariants

The browser model work remains subject to the existing strict TypeScript policy. CI rejects explicit `any`, TypeScript suppression directives, relaxed strict flags, compile errors, and model-migration regression failures. The deployment boundary contains transpiled JavaScript/WASM only, never TypeScript source.
