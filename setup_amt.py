#!/usr/bin/env python3

from setup_engine_common import ROOT, download, ensure_repo, ensure_venv, pip_install

AMT_DIR = ROOT / "third_party" / "AMT"
VENV_DIR = ROOT / ".amt-venv"
CHECKPOINT = AMT_DIR / "pretrained" / "amt-s.pth"


def main():
    print("Installing AMT for WebP Animator")
    ensure_repo("https://github.com/MCG-NKU/AMT.git", AMT_DIR)
    python = ensure_venv(VENV_DIR)
    pip_install(
        python,
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    )
    pip_install(
        python,
        "numpy<2",
        "opencv-python-headless<5",
        "imageio>=2.19,<3",
        "omegaconf==2.3.0",
        "Pillow",
        "tqdm",
    )
    download("https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth", CHECKPOINT)
    print("\nAMT setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {AMT_DIR}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
