# WebP Animator

A small LAN-accessible Python web tool for building animated WebP files from raster frames or an existing animated WebP.

Features:

- Upload multiple raster frames and preview them as thumbnails.
- Upload an existing animated WebP and automatically extract all of its frames.
- Drag frames into the exact animation order before processing.
- Remove or move individual frames before generation.
- Pixel-similarity registration with horizontal, vertical, or free X/Y translation.
- Optional aspect-ratio-preserving auto-shrink for frames larger than the preceding frame.
- Fix-to-frame-1 mode that independently optimises uniform scale and X/Y pan for every frame against the first frame.
- No source-pixel cropping during registration; the canvas expands as needed.
- Optional Practical-RIFE 4.25 interpolation at 2x, 4x, or 8x.
- Optional paid **OpenAI Interrogator** semantic interpolation with planning, generation, visual auditing, retries, whole-sequence context and loop awareness.
- Recoverable OpenAI jobs with server-owned Job IDs, localStorage discovery, persistent temporary source/attempt data, spend limits and resume-after-disconnect behaviour.
- Live upload, WebP extraction, alignment, interpolation and encoding progress.
- Lossless WebP by default, including animation-size optimisation, with optional lossy output.

## Main application setup

Create and activate a project virtual environment, then install the lightweight web-app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the original local/RIFE-only server:

```bash
python app.py
```

For the full application including the optional OpenAI Interrogator:

```bash
python app_openai.py
```

Both entry points use:

```text
0.0.0.0:18743
```

Run only one at a time. On the same computer use `http://127.0.0.1:18743`; from another device on the LAN use `http://<server-LAN-IP>:18743`. The operating-system firewall must allow inbound TCP `18743`.

## OpenAI Interrogator

OpenAI is optional. The normal alignment/RIFE/WebP pipeline does not depend on an API key.

The Interrogator uses a pinned `kitty-crow/openai-schema` git submodule for structured planner and auditor calls, and a tiny local Node/Bun bridge for the OpenAI API. Python remains the authoritative job owner and launches the bridge on demand on `127.0.0.1:18744` only.

Initialise and build the vendored schema library:

```bash
git submodule update --init --recursive
python setup_openai.py
```

Create `.env` in the repository root containing:

```text
OPENAI_API_KEY=...
```

`.env` is ignored by Git and the key is never sent to the browser.

The Interrogator supports:

- **Single**: one missing frame between two selected temporal anchors.
- **Fixed**: N missing frames generated progressively between two anchors.
- **Auto**: progressively add frames until the planner judges another frame is not materially worthwhile, subject to hard frame/spend limits.
- **Open or loop topology**: loop closure can explicitly use the last frame -> first frame gap.
- **Whole-animation context**: a labelled contact sheet gives the planner/auditor context from the known animation while immediate anchors are sent separately at full detail.
- **Optional user guidance**: the planner treats user wording as intent and refines it after visually inspecting the frames. The base prompt itself is deliberately content-agnostic.
- **Audited generations**: every candidate is visually checked against the anchors, plan, user intent and sequence context. Scores are rubric scores, not probabilities.
- **Learning retries**: a rejected image, its plan, audit diagnosis and later user feedback are reused as negative/corrective evidence for the next attempt.
- **Cost visibility**: the UI estimates planning + image generation/edit + audit before starting, shows the next-attempt estimate, tracks observed usage where available, and can stop before a configured maximum spend is exceeded.
- **Background/recovery**: after the upload returns a Job ID, the browser is no longer part of execution. Jobs can be restored after tab/app closure or network loss; a server restart marks active jobs `interrupted` so they can be resumed rather than silently discarded.
- **Hybrid OpenAI + RIFE**: accepted semantic frames can be inserted into the ordinary animation list and then smoothed cheaply with local RIFE.

OpenAI job data is temporary but deliberately durable enough for recovery and retries. It lives under `.openai-jobs/`, which is ignored by Git. Active jobs are never removed by cleanup; completed jobs default to a 24-hour retention period and review/budget/interrupted jobs to seven days.

See [`OPENAI_INTERROGATOR.md`](OPENAI_INTERROGATOR.md) for the detailed architecture and state machine.

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

The RIFE model is deliberately not committed to this repository. `setup_rife.py` downloads the official Practical-RIFE 4.25 weights.

Practical-RIFE currently documents Python `<= 3.11`. The installer prefers `python3.11` or `python3.10` when available, then falls back to the current interpreter with a warning. To force a specific interpreter:

```bash
RIFE_BOOTSTRAP_PYTHON=python3.11 python setup_rife.py
```

After setup, restart the application. The page will report whether RIFE is ready.

### Processing order

The ordinary pipeline is:

```text
uploaded raster frames / extracted WebP frames
  -> user-defined drag order
  -> selected geometry correction (sequential / fit-previous / fix-to-frame-1)
  -> common expanded canvas
  -> optional RIFE interpolation
  -> lossless/optional lossy animated WebP encoding
```

With OpenAI assistance, semantically generated and audited frames are inserted into the ordered browser frame list before the ordinary pipeline. RIFE can then be used as a free final smoothing pass around those stronger semantic anchors.

The selected source frame duration is divided by the RIFE interpolation multiplier so the animation keeps approximately the same playback speed. For example, `200 ms` source frames with `4x` RIFE produce `50 ms` output frames, roughly `20 fps`.

## Environment overrides

RIFE path overrides:

```text
RIFE_PYTHON
RIFE_DIR
RIFE_MODEL_DIR
```

OpenAI bridge overrides:

```text
OPENAI_API_KEY
OPENAI_API_BASE
OPENAI_BRIDGE_HOST
OPENAI_BRIDGE_PORT
OPENAI_BRIDGE_RUNTIME
```

The OpenAI bridge host should normally remain `127.0.0.1`.
