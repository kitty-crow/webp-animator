#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from setup_engine_common import (
    choose_torch_cuda_variant,
    ensure_repo,
    ensure_venv,
    installed_torch_cuda,
    run,
    validate_cuda_runtime,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "ToonCrafter"
VENV = ROOT / ".tooncrafter-venv"
REPOSITORY = "https://github.com/Doubiiu/ToonCrafter.git"
MODEL_REPOSITORY = "Doubiiu/ToonCrafter"

CUDA_VARIANTS = {
    "cu118": {
        "index": "https://download.pytorch.org/whl/cu118",
        "runtime": "11.8",
    },
    "cu121": {
        "index": "https://download.pytorch.org/whl/cu121",
        "runtime": "12.1",
    },
}


def choose_python() -> str:
    explicit = os.environ.get("TOONCRAFTER_BOOTSTRAP_PYTHON")
    candidates = [explicit] if explicit else []
    candidates += ["python3.10", "python3.9", "python3.8", sys.executable]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).is_file():
            return str(resolved)
    raise RuntimeError("Could not find Python 3.8-3.10 for the ToonCrafter environment.")


def _filtered_requirements() -> Path:
    source_requirements = (SOURCE / "requirements.txt").read_text(encoding="utf-8")
    destination = VENV / "tooncrafter-upstream-requirements.txt"
    kept: list[str] = []
    for raw in source_requirements.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("torch==") or lower == "torchvision" or lower.startswith("xformers"):
            continue
        if lower.startswith("gradio") or lower.startswith("moviepy") or lower == "av":
            continue
        kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return destination


def install_requirements(python: Path) -> tuple[str, str]:
    variant, driver = choose_torch_cuda_variant("TOONCRAFTER_CUDA_VARIANT", cuda12_variant="cu121")
    metadata = CUDA_VARIANTS[variant]
    index = os.environ.get("TOONCRAFTER_TORCH_INDEX", str(metadata["index"]))
    expected_cuda = str(metadata["runtime"])
    driver_text = f"{driver[0]}.{driver[1]}" if driver else "unknown"
    print(f"ToonCrafter CUDA stack: {variant} (driver reports CUDA {driver_text})")
    print(f"PyTorch index: {index}")

    command = [python, "-m", "pip", "install"]
    if installed_torch_cuda(python) != expected_cuda:
        command.append("--force-reinstall")
    command += [
        "torch==2.1.0",
        "torchvision==0.16.0",
        "--index-url",
        index,
    ]
    run(command)
    run([python, "-m", "pip", "install", "xformers==0.0.22.post7"])
    run([python, "-m", "pip", "install", "-r", _filtered_requirements()])
    run([python, "-m", "pip", "install", "huggingface_hub>=0.25,<1"])
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, expected_cuda)
    return variant, expected_cuda


def download_model(python: Path) -> Path:
    checkpoint_dir = SOURCE / "checkpoints" / "tooncrafter_512_interp_v1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / "model.ckpt"
    if destination.is_file() and destination.stat().st_size > 1024 * 1024:
        print(f"Already present: {destination}")
        return destination
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={MODEL_REPOSITORY!r}, filename='model.ckpt', "
        f"local_dir={str(checkpoint_dir)!r}))"
    )
    run([python, "-c", code])
    if not destination.is_file():
        raise RuntimeError(f"ToonCrafter checkpoint download did not create {destination}")
    return destination


def validate_runtime(python: Path, checkpoint: Path) -> None:
    code = f"""
import os
import sys
from pathlib import Path
import torch
import decord
import einops
import omegaconf
import pytorch_lightning
import transformers
import xformers

source = Path({str(SOURCE)!r})
checkpoint = Path({str(checkpoint)!r})
os.chdir(source)
sys.path.insert(0, str(source))
sys.path.insert(0, str(source / 'lvdm'))
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler

try:
    state = torch.load(str(checkpoint), map_location='cpu', mmap=True)
except TypeError:
    state = torch.load(str(checkpoint), map_location='cpu')
assert isinstance(state, dict) and state, 'ToonCrafter checkpoint is empty or invalid'
del state
print('ToonCrafter runtime imports: ok')
print('ToonCrafter checkpoint CPU load: ok')
"""
    run([python, "-c", code])


def main() -> None:
    print("Installing ToonCrafter generative cartoon interpolation")
    ensure_repo(REPOSITORY, SOURCE)
    python = ensure_venv(VENV, choose_python())
    cuda_variant, expected_cuda = install_requirements(python)
    checkpoint = download_model(python)
    validate_runtime(python, checkpoint)

    print("\nToonCrafter setup complete.")
    print(f"Python:     {python}")
    print(f"Source:     {SOURCE}")
    print(f"Checkpoint: {checkpoint}")
    print(f"CUDA:       {cuda_variant} / PyTorch runtime {expected_cuda}")
    print("Runtime preflight: CUDA, imports and checkpoint CPU load all passed.")
    print("The official model targets 512x320 and is very memory hungry. The WebP worker uses FP16, component offload and reduced-resolution retries, but 4 GB GPUs may still be below the practical floor.")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
