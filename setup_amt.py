#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "AMT"
VENV = ROOT / ".amt-venv"
CHECKPOINT = SOURCE / "pretrained" / "amt-s.pth"
CHECKPOINT_URL = "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"


def main():
    print("Installing AMT for WebP Animator")
    ensure_repo("https://github.com/MCG-NKU/AMT.git", SOURCE)
    python = ensure_venv(VENV, choose_python("AMT_BOOTSTRAP_PYTHON"))
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
        "Pillow>=9",
        "opencv-python-headless>=4.5",
        "imageio>=2.19",
        "omegaconf>=2.3",
        "tqdm>=4.64",
    ])
    download(CHECKPOINT_URL, CHECKPOINT)
    print("\nAMT setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
