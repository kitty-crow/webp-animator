#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "SPEED"
VENV = ROOT / ".speed-venv"
CHECKPOINT = SOURCE / "ckpts" / "speed.pt"
CHECKPOINT_URL = "https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt"


def main():
    print("Installing SPEED for WebP Animator")
    ensure_repo("https://github.com/bbldCVer/SPEED.git", SOURCE)
    python = ensure_venv(VENV, choose_python("SPEED_BOOTSTRAP_PYTHON"))
    requirements = SOURCE / "requirements-cu118.txt"
    if not requirements.is_file():
        raise RuntimeError("SPEED requirements-cu118.txt was not found after cloning the repository.")
    run([python, "-m", "pip", "install", "-r", requirements], cwd=SOURCE)
    download(CHECKPOINT_URL, CHECKPOINT)
    print("\nSPEED setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Pascal GPUs automatically use FP16 rather than unsupported BF16.")


if __name__ == "__main__":
    main()
