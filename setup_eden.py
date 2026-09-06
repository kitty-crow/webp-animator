#!/usr/bin/env python3

from setup_engine_common import ROOT, download, ensure_repo, ensure_venv, pip_install

EDEN_DIR = ROOT / "third_party" / "EDEN"
VENV_DIR = ROOT / ".eden-venv"
CHECKPOINT = EDEN_DIR / "checkpoints" / "eden.pt"


def main():
    print("Installing EDEN for WebP Animator")
    ensure_repo("https://github.com/bbldCVer/EDEN.git", EDEN_DIR)
    python = ensure_venv(VENV_DIR)
    pip_install(
        python,
        "torch==2.0.1+cu118",
        "torchvision==0.15.2+cu118",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    )
    pip_install(
        python,
        "xformers==0.0.22",
        "numpy==1.24.3",
        "Pillow",
        "PyYAML==6.0.2",
        "einops==0.7.0",
        "torchdiffeq==0.2.4",
        "scipy",
    )
    download("https://huggingface.co/zhZ524/EDEN/resolve/main/eden.pt", CHECKPOINT)
    print("\nEDEN setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {EDEN_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
