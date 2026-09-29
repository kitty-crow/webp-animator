# WebP Animator

A LAN-accessible Python web tool for building animated WebP files from raster frames or an existing animated WebP.

## Features

- Upload multiple raster frames and preview them as thumbnails.
- Upload an existing animated WebP and automatically extract all of its frames.
- Drag frames into the exact animation order before processing.
- Remove or move individual frames before generation.
- Pixel-similarity registration with horizontal, vertical, or free X/Y translation.
- Optional aspect-ratio-preserving auto-shrink for frames larger than the preceding frame.
- Fix-to-frame-1 mode that independently optimises uniform scale and X/Y pan for every frame against the first frame.
- No source-pixel cropping during registration; the canvas expands as needed.
- Conventional RIFE and AMT interpolation plus optional generative VFI with Multi-Input ResShift Diffusion and MoG.
- Durable global Job IDs with server-side source/settings/result persistence.
- Browser IndexedDB snapshots of the current workspace and final WebP.
- Animated final-WebP preview and repeat download without regenerating.
- Live upload, WebP extraction, alignment, interpolation and encoding progress.
- Lossless WebP by default, including animation-size optimisation, with optional lossy output.

## Setup

Create and activate a project virtual environment, then install the lightweight application dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recommended server is the persistent-job entry point:

```bash
python app_all.py
```

It listens on:

```text
0.0.0.0:18743
```

On the same computer use `http://127.0.0.1:18743`. From another device on the LAN use `http://<server-LAN-IP>:18743`. The operating-system firewall must allow inbound TCP `18743`.

`app.py` remains available as the minimal in-memory server, but it does not provide the durable global workspace layer.

## Recoverable jobs and final WebP cache

`app_all.py` gives each workspace a 32-character Global Job ID.

The browser keeps the current workspace in IndexedDB and the server keeps submitted source frames, settings and the latest finished WebP under:

```text
.webp-jobs/<job-id>/
```

Jobs have no automatic age-based expiry. Press **New job** to start a fresh workspace; previous Job IDs remain restorable.

After a WebP finishes, the page shows an animated preview and caches the finished WebP in IndexedDB. The same result can therefore be downloaded again without rerunning alignment, interpolation or WebP encoding. If the browser copy is unavailable, the server-side copy can still be recovered by Job ID.

If the server restarts during an active render, the job is marked interrupted and `app_all.py` resumes it from the persisted sources/settings on startup.

## Auto-shrink oversized frames

Choose **Shrink oversized frames to fit previous, then align** when source frames have inconsistent raster dimensions.

For every frame after the first, the tool checks the preceding processed frame. If the current frame is wider or taller, it applies the largest uniform scale that makes the entire current frame fit inside the previous frame:

```text
scale = min(previous width / current width,
            previous height / current height,
            1.0)
```

The frame is never enlarged, its aspect ratio is always preserved, and Lanczos resampling is used for the reduction. Registration then runs on the fitted frames.

## Fix existing WebP to frame 1

Choose **Fix to frame 1: optimise scale + pan against first frame** to stabilise an existing animation whose frames have inconsistent zoom or position.

The first frame is never changed. Every later frame is independently compared with frame 1, and the tool searches for the uniform scale plus allowed X/Y translation that produces the highest pixel-similarity score. Because the same anchor is used every time, corrections do not accumulate from one frame to the next.

Scaling is always uniform, so aspect ratio cannot be distorted. The optimiser can shrink or enlarge a frame when that improves the match. Translation still respects the **Allowed movement** setting, so it can be constrained to horizontal only, vertical only, both axes, or disabled.

When an animated WebP is uploaded as the first source, the browser automatically selects this mode. You can switch back to sequential alignment if desired.

## Existing animated WebP input

You can upload a single animated WebP just like any other image.

The server decodes the WebP into its constituent frames and returns them to the browser as PNG frames. Those extracted frames then behave like normal uploaded frames: you can reorder them, remove unwanted frames, align them, interpolate them and generate a new WebP.

If the source WebP contains frame-duration metadata, the most common source duration is used as the suggested source frame duration in the UI. The final output still uses the single duration selected in the form.

## RIFE interpolation

RIFE is optional. The normal align-and-encode workflow works without it.

To install Practical-RIFE and the recommended 4.25 model into an isolated environment:

```bash
source .venv/bin/activate
python setup_rife.py
```

The installer creates:

```text
.rife-venv/
third_party/Practical-RIFE/
third_party/Practical-RIFE/train_log/
```

The model is deliberately not committed to this repository. `setup_rife.py` downloads the Practical-RIFE 4.25 weights.

Practical-RIFE currently documents Python `<= 3.11`. The installer prefers `python3.11` or `python3.10` when available, then falls back to the current interpreter with a warning. To force a specific interpreter:

```bash
RIFE_BOOTSTRAP_PYTHON=python3.11 python setup_rife.py
```

After setup, restart the application. The page reports whether RIFE is ready.

## Generative frame interpolation

The **Interpolator** selector also supports two endpoint-constrained generative VFI families. These are substantially slower than RIFE or AMT and are intended for gaps where conventional optical-flow-style interpolation struggles with occlusion, large motion, articulation or newly revealed image content.

### Multi-Input ResShift Diffusion VFI

Install it into its own environment with:

```bash
python setup_resshift.py
```

The worker uses both endpoint frames and explicit interpolation time. It keeps one model resident per worker, crops transparent/unused canvas area before inference, reuses endpoint optical flow for multiple requested times, and progressively reduces its internal inference resolution after CUDA OOM. Diffusion-step progress is forwarded to the normal WebP Animator progress bar.

The installer creates:

```text
.resshift-venv/
third_party/Multi-Input-Resshift-Diffusion-VFI/
third_party/Multi-Input-Resshift-Diffusion-VFI/_webp_model/
```

### Motion-Aware Generative Frame Interpolation (MoG)

MoG has separate animation and real-world checkpoints. Install either one or both:

```bash
python setup_mog.py --variant ani
python setup_mog.py --variant real
python setup_mog.py --variant both
```

MoG is much heavier than ResShift. Its worker keeps the main diffusion model in FP16, avoids a temporary FP32 CUDA copy, stages the separate motion estimator onto the GPU only while calculating motion guidance, enables per-frame autoencoder decoding, crops empty canvas area and retries at smaller internal resolutions after CUDA OOM.

The released MoG checkpoints are very large. A 4 GB GPU can still be below the model's practical minimum even at the smallest fallback resolution. This is reported as a clear engine error rather than silently switching to a different interpolation model.

### Multi-GPU execution

Generative interpolation gaps are independent jobs. When more than one CUDA GPU is visible, WebP Animator shards gaps across GPUs and runs one model worker per GPU in parallel. This avoids pretending separate VRAM pools are one device and scales naturally on equal-card systems.

To restrict or select devices, use the normal CUDA mask:

```bash
CUDA_VISIBLE_DEVICES=0,1 python app_all.py
```

To cap generative workers while still leaving more CUDA devices visible to other applications:

```bash
GENERATIVE_VFI_GPU_WORKERS=1 python app_all.py
```

Existing RIFE, AMT, EDEN and SPEED execution paths are not replaced by the generative engines.

### Processing order

```text
uploaded raster frames / extracted WebP frames
  -> user-defined drag order
  -> selected geometry correction
  -> common expanded canvas
  -> smart/manual gap selection
  -> selected interpolator and/or structural generator
  -> optimised lossless/optional lossy animated WebP encoding
```

The source timing is redistributed according to the exact temporal positions returned by the selected interpolation engine, so adding generated frames does not lengthen the animation.

## Environment overrides

RIFE path overrides:

```text
RIFE_PYTHON
RIFE_DIR
RIFE_MODEL_DIR
```

ResShift path overrides:

```text
RESSHIFT_PYTHON
RESSHIFT_DIR
RESSHIFT_MODEL_DIR
RESSHIFT_BOOTSTRAP_PYTHON
RESSHIFT_TORCH_INDEX
```

MoG path overrides:

```text
MOG_PYTHON
MOG_DIR
MOG_ANI_CHECKPOINT
MOG_REAL_CHECKPOINT
MOG_FLOW_CHECKPOINT
MOG_ANI_CONFIG
MOG_REAL_CONFIG
MOG_BOOTSTRAP_PYTHON
MOG_TORCH_INDEX
MOG_DDIM_STEPS
```
