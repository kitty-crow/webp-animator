#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run, validate_cuda_runtime

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "AMT"
VENV = ROOT / ".amt-venv"
CONFIG = SOURCE / "cfgs" / "AMT-S.yaml"
CHECKPOINT = SOURCE / "pretrained" / "amt-s.pth"
CHECKPOINT_URL = "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth"


def validate_runtime(python: Path) -> None:
    code = f"""
import gc
import sys
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
import amt_worker
import torch
torch_module, device, model = amt_worker.load_amt(Path({str(SOURCE)!r}), Path({str(CONFIG)!r}), Path({str(CHECKPOINT)!r}))
assert device.type == 'cuda', 'AMT runtime did not select CUDA'
print('AMT model load: ok')
del model
gc.collect()
torch.cuda.empty_cache()
"""
    run([python, "-c", code])


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
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, "11.8")
    download(CHECKPOINT_URL, CHECKPOINT)
    validate_runtime(python)
    print("\nAMT setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Runtime preflight: CUDA and actual AMT model load passed.")


if __name__ == "__main__":
    main()
