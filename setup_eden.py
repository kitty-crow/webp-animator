#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from setup_engine_common import choose_python, download, ensure_repo, ensure_venv, run, validate_cuda_runtime

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "EDEN"
VENV = ROOT / ".eden-venv"
CONFIG = SOURCE / "configs" / "eval_eden.yaml"
CHECKPOINT = SOURCE / "checkpoints" / "eden.pt"
CHECKPOINT_URL = "https://huggingface.co/zhZ524/EDEN/resolve/main/eden.pt"


def validate_runtime(python: Path) -> None:
    code = f"""
import gc
import os
import sys
from pathlib import Path
import torch
import yaml
source = Path({str(SOURCE)!r})
config_path = Path({str(CONFIG)!r})
checkpoint_path = Path({str(CHECKPOINT)!r})
os.chdir(source)
sys.path.insert(0, str(source))
from src.models import load_model
config = yaml.unsafe_load(config_path.read_text(encoding='utf-8'))
model_args = dict(config['model_args'])
if bool(model_args.get('use_xformers', False)):
    try:
        import xformers
    except Exception:
        model_args['use_xformers'] = False
model = load_model(config['model_name'], **model_args)
checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
model.load_state_dict(checkpoint['eden'])
del checkpoint
model = model.to('cuda').eval()
torch.cuda.synchronize()
print('EDEN model load: ok')
del model
gc.collect()
torch.cuda.empty_cache()
"""
    run([python, "-c", code])


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
        "cupy-cuda11x>=13,<14",
    ])
    # EDEN's src.utils imports FloLPIPS at module import time, and its correlation
    # kernel imports CuPy even though the interpolation worker does not calculate
    # that metric. Keeping CuPy here avoids an upstream import-time failure.
    # xFormers is an acceleration rather than a correctness dependency. Some
    # Windows/Pascal combinations do not have a compatible wheel, and the worker
    # falls back to ordinary PyTorch attention cleanly in that case.
    completed = subprocess.run([
        str(python),
        "-m",
        "pip",
        "install",
        "xformers==0.0.22",
    ])
    if completed.returncode != 0:
        print("WARNING: xformers could not be installed; EDEN will use PyTorch attention.")
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, "11.8", require_cupy=True)
    download(CHECKPOINT_URL, CHECKPOINT)
    validate_runtime(python)
    print("\nEDEN setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Runtime preflight: CUDA, CuPy and actual EDEN model load passed.")


if __name__ == "__main__":
    main()
