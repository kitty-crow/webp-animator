#!/usr/bin/env python3

from setup_engine_common import ROOT, download, ensure_repo, ensure_venv, run

SPEED_DIR = ROOT / "third_party" / "SPEED"
VENV_DIR = ROOT / ".speed-venv"
CHECKPOINT = SPEED_DIR / "ckpts" / "speed.pt"


def main():
    print("Installing SPEED for WebP Animator")
    ensure_repo("https://github.com/bbldCVer/SPEED.git", SPEED_DIR)
    python = ensure_venv(VENV_DIR)
    # Use the upstream CUDA 11.8 dependency set. JASPER's NVIDIA/PyTorch stack
    # already targets CUDA 11.8, and SPEED requires xFormers + CuPy.
    run([python, "-m", "pip", "install", "-r", str(SPEED_DIR / "requirements-cu118.txt")], cwd=SPEED_DIR)
    download("https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt", CHECKPOINT)
    print("\nSPEED setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SPEED_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("WebP Animator uses fp16 by default on Pascal GPUs; do not select bf16 on the GTX 1050 Ti/1080.")


if __name__ == "__main__":
    main()
