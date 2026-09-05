# WebP Animator

A small LAN-accessible Python web tool for building animated WebP files from raster frames.

Features:

- Upload multiple raster frames and preview them as thumbnails.
- Drag frames into the exact animation order before processing.
- Pixel-similarity registration with horizontal, vertical, or free X/Y translation.
- No resizing or source-pixel cropping during registration; the canvas expands as needed.
- Optional Practical-RIFE 4.25 interpolation at 2x, 4x, or 8x.
- Live upload, alignment, interpolation, and encoding progress.
- Lossless WebP by default, with optional lossy output.

## Main application setup

Create and activate a project virtual environment, then install the lightweight web-app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the server:

```bash
python app.py
```

It binds to:

```text
0.0.0.0:18743
```

On the same computer:

```text
http://127.0.0.1:18743
```

From another device on the same LAN:

```text
http://<server-LAN-IP>:18743
```

Your operating-system firewall must allow inbound TCP connections on port `18743`.

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

After setup, restart `app.py`. The page will report whether RIFE is ready.

### Processing order

When RIFE is enabled, the pipeline is:

```text
uploaded frames
  -> user-defined drag order
  -> pixel registration
  -> common expanded canvas
  -> RIFE interpolation
  -> animated WebP encoding
```

The selected source frame duration is divided by the interpolation multiplier so the animation keeps approximately the same playback speed. For example, `200 ms` source frames with `4x` RIFE produce `50 ms` output frames, roughly `20 fps`.

## Environment overrides

The server normally discovers the RIFE installation created by `setup_rife.py`. These variables can override the paths:

```text
RIFE_PYTHON
RIFE_DIR
RIFE_MODEL_DIR
```
