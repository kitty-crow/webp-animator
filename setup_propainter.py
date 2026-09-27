#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run, validate_cuda_runtime

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "ProPainter"
VENV = ROOT / ".propainter-venv"
RELEASE = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
WEIGHTS = {
    "ProPainter.pth": f"{RELEASE}/ProPainter.pth",
    "recurrent_flow_completion.pth": f"{RELEASE}/recurrent_flow_completion.pth",
    "raft-things.pth": f"{RELEASE}/raft-things.pth",
}


def validate_runtime(python: Path) -> None:
    code = f"""
import os
import sys
from pathlib import Path
import torch
source = Path({str(SOURCE)!r})
os.chdir(source)
sys.path.insert(0, str(source))
for name in {repr(list(WEIGHTS))}:
    path = source / 'weights' / name
    value = torch.load(str(path), map_location='cpu')
    assert value is not None, f'Invalid ProPainter checkpoint: {{path}}'
    del value
import cv2
import einops
import timm
print('ProPainter runtime imports: ok')
print('ProPainter checkpoint CPU loads: ok')
"""
    run([python, "-c", code])


def main():
    print("Installing ProPainter temporal repair for WebP Animator")
    ensure_repo("https://github.com/sczhou/ProPainter.git", SOURCE)
    python = ensure_venv(VENV, choose_python("PROPAINTER_BOOTSTRAP_PYTHON"))

    run([
        python,
        "-m",
        "pip",
        "install",
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ])
    run([
        python,
        "-m",
        "pip",
        "install",
        "numpy<2",
        "av",
        "addict",
        "einops",
        "future",
        "scipy",
        "opencv-python-headless",
        "matplotlib",
        "scikit-image",
        "imageio-ffmpeg",
        "pyyaml",
        "requests",
        "timm",
        "yapf",
        "tqdm",
        "Pillow>=9",
    ])
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, "11.8")

    for name, url in WEIGHTS.items():
        download(url, SOURCE / "weights" / name)

    validate_runtime(python)
    print("\nProPainter setup complete.")
    print(f"Python: {python}")
    print(f"Source: {SOURCE}")
    print("Mode:   CUDA + FP16 low-VRAM temporal repair")
    print("Runtime preflight: CUDA, imports and checkpoint CPU loads all passed.")


if __name__ == "__main__":
    main()
