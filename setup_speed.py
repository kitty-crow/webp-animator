#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setup_engine_common import (
    choose_python,
    download,
    ensure_repo,
    ensure_venv,
    installed_torch_cuda,
    run,
    validate_cuda_runtime,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "SPEED"
VENV = ROOT / ".speed-venv"
CONFIG = SOURCE / "configs" / "eval_config.yaml"
CHECKPOINT = SOURCE / "ckpts" / "speed.pt"
CHECKPOINT_URL = "https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
EXPECTED_CUDA = "11.8"


def _filtered_requirements() -> Path:
    """Install SPEED's shared dependencies without letting pip replace CUDA torch.

    Upstream requirements-cu118.txt uses --extra-index-url together with unpinned
    torch/torchvision. pip is therefore free to satisfy those names from ordinary
    PyPI, which can leave an apparently installed SPEED environment with CPU-only
    PyTorch. WebP Animator owns the CUDA stack explicitly instead.
    """
    source = (SOURCE / "requirements.txt").read_text(encoding="utf-8")
    destination = VENV / "speed-upstream-requirements.txt"
    kept: list[str] = []
    for raw in source.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line or line.startswith("#"):
            if line.startswith("#"):
                kept.append(line)
            continue
        if lower.startswith("torch") or lower.startswith("torchvision"):
            continue
        if lower.startswith("xformers") or lower.startswith("cupy-cuda"):
            continue
        kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return destination


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

    # Install from the CUDA-only index, not an extra index. This prevents pip from
    # silently selecting a CPU wheel from PyPI on Windows.
    command = [python, "-m", "pip", "install"]
    if installed_torch_cuda(python) != EXPECTED_CUDA:
        command.append("--force-reinstall")
    command += [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "--index-url",
        TORCH_INDEX,
    ]
    run(command)
    run([python, "-m", "pip", "uninstall", "-y", "cupy-cuda11x", "cupy-cuda12x", "xformers"], check=False)
    run([python, "-m", "pip", "install", "cupy-cuda11x>=12,<14"])
    run([python, "-m", "pip", "install", "-r", _filtered_requirements()], cwd=SOURCE)
    run([python, "-m", "pip", "check"])

    validate_cuda_runtime(python, EXPECTED_CUDA, require_cupy=True)
    download(CHECKPOINT_URL, CHECKPOINT)
    validate_runtime(python)
    print("\nSPEED setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"CUDA:       cu118 / PyTorch runtime {EXPECTED_CUDA}")
    print("Pascal GPUs use PyTorch SDPA when xFormers is unavailable/unsupported.")
    print("Runtime preflight: CUDA, CuPy and actual SPEED model load passed.")


if __name__ == "__main__":
    main()
