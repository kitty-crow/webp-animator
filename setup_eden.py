#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "EDEN"
VENV = ROOT / ".eden-venv"
CHECKPOINT = SOURCE / "checkpoints" / "eden.pt"
CHECKPOINT_URL = "https://huggingface.co/zhZ524/EDEN/resolve/main/eden.pt"


def main():
    print("Installing EDEN for WebP Animator")
    ensure_repo("https://github.com/bbldCVer/EDEN.git", SOURCE)
    python = ensure_venv(VENV, choose_python("EDEN_BOOTSTRAP_PYTHON"))
    run([
        python,
        "-m",
        "pip",
        "install",
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ])
    run([
        python,
        "-m",
        "pip",
        "install",
        "numpy==1.24.3",
        "Pillow>=10",
        "PyYAML>=6",
        "einops>=0.7",
        "scipy>=1.10",
        "torchdiffeq>=0.2.4",
        "lpips==0.1.4",
        "opencv-python-headless>=4.5",
    ])
    # xFormers is an acceleration, not a correctness dependency. Some Windows /
    # Pascal combinations do not have a compatible wheel, and the worker falls
    # back to ordinary PyTorch attention cleanly in that case.
    completed = subprocess.run([
        str(python),
        "-m",
        "pip",
        "install",
        "xformers==0.0.22",
    ])
    if completed.returncode != 0:
        print("WARNING: xformers could not be installed; EDEN will use PyTorch attention.")
    download(CHECKPOINT_URL, CHECKPOINT)
    print("\nEDEN setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
