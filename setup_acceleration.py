#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def run(command):
    command = [str(item) for item in command]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main():
    print("Installing optional CUDA acceleration into the WebP Animator environment")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch==2.5.1+cu118",
        "torchvision==0.20.1+cu118",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ])
    run([sys.executable, "-m", "pip", "check"])
    run([
        sys.executable,
        "-c",
        "import torch; print('torch', torch.__version__); print('CUDA', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); assert str(torch.version.cuda or '').startswith('11.8'); assert torch.cuda.is_available(), 'main WebP Animator environment cannot initialise CUDA'",
    ])
    print("\nCUDA acceleration setup complete.")
    print("Runtime preflight: CUDA initialisation passed.")
    print("Restart app_all.py so GPU frame matching and smart analysis can be detected.")


if __name__ == "__main__":
    main()
