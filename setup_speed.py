#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run, validate_cuda_runtime

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "SPEED"
VENV = ROOT / ".speed-venv"
CONFIG = SOURCE / "configs" / "eval_config.yaml"
CHECKPOINT = SOURCE / "ckpts" / "speed.pt"
CHECKPOINT_URL = "https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt"


def validate_runtime(python: Path) -> None:
    code = f"""
import gc
import os
import sys
from pathlib import Path
import torch
source = Path({str(SOURCE)!r})
os.chdir(source)
sys.path.insert(0, str(source))
import inference as speed_inference
device = speed_inference.get_device(None)
assert device.type == 'cuda', f'SPEED runtime selected {{device}} instead of CUDA'
model = speed_inference.build_model(
    {str(CONFIG)!r},
    {str(CHECKPOINT)!r},
    device,
    strict_load=False,
)
model.eval()
torch.cuda.synchronize()
print('SPEED model load: ok')
del model
gc.collect()
torch.cuda.empty_cache()
"""
    run([python, "-c", code])


def main():
    print("Installing SPEED for WebP Animator")
    ensure_repo("https://github.com/bbldCVer/SPEED.git", SOURCE)
    python = ensure_venv(VENV, choose_python("SPEED_BOOTSTRAP_PYTHON"))
    requirements = SOURCE / "requirements-cu118.txt"
    if not requirements.is_file():
        raise RuntimeError("SPEED requirements-cu118.txt was not found after cloning the repository.")
    run([python, "-m", "pip", "install", "-r", requirements], cwd=SOURCE)
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, "11.8")
    download(CHECKPOINT_URL, CHECKPOINT)
    validate_runtime(python)
    print("\nSPEED setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Pascal GPUs automatically use FP16 rather than unsupported BF16.")
    print("Runtime preflight: CUDA and actual SPEED model load passed.")


if __name__ == "__main__":
    main()
