#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from setup_engine_common import (
    choose_torch_cuda_variant,
    ensure_repo,
    ensure_venv,
    install_optional_xformers,
    installed_torch_cuda,
    run,
    validate_cuda_runtime,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "MoG-VFI"
VENV = ROOT / ".mog-venv"
REPOSITORY = "https://github.com/MCG-NJU/MoG-VFI.git"
MODEL_REPOSITORY = "MCG-NJU/MoG"
FLOW_FOLDER = "https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o"

CUDA_VARIANTS = {
    "cu118": {
        "index": "https://download.pytorch.org/whl/cu118",
        "runtime": "11.8",
        "cupy": "cupy-cuda11x>=12,<14",
    },
    "cu121": {
        "index": "https://download.pytorch.org/whl/cu121",
        "runtime": "12.1",
        "cupy": "cupy-cuda12x>=12,<14",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Install Motion-Aware Generative Frame Interpolation")
    parser.add_argument("--variant", choices=("ani", "real", "both"), default="ani")
    return parser.parse_args()


def choose_python() -> str:
    explicit = os.environ.get("MOG_BOOTSTRAP_PYTHON")
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
    raise RuntimeError("Could not find Python 3.8-3.10 for the MoG environment.")


def _filtered_requirements() -> Path:
    source_requirements = (SOURCE / "requirements.txt").read_text(encoding="utf-8")
    destination = VENV / "mog-upstream-requirements.txt"
    kept: list[str] = []
    for raw in source_requirements.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        # CUDA/PyTorch packages are selected by this installer so upstream cannot
        # silently replace them with a stack the installed NVIDIA driver cannot use.
        if lower.startswith("torch") or lower.startswith("torchvision"):
            continue
        if lower.startswith("xformers") or lower.startswith("cupy-cuda"):
            continue
        if lower.startswith("deepspeed"):
            continue
        if lower.startswith("accelerate[deepspeed]"):
            continue
        if "microsoft/lora" in lower:
            continue
        kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return destination


def install_requirements(python: Path) -> tuple[str, str]:
    variant, driver = choose_torch_cuda_variant("MOG_CUDA_VARIANT", cuda12_variant="cu121")
    metadata = CUDA_VARIANTS[variant]
    index = os.environ.get("MOG_TORCH_INDEX", str(metadata["index"]))
    expected_cuda = str(metadata["runtime"])
    cupy_package = os.environ.get("MOG_CUPY_PACKAGE", str(metadata["cupy"]))
    driver_text = f"{driver[0]}.{driver[1]}" if driver else "unknown"
    print(f"MoG CUDA stack: {variant} (driver reports CUDA {driver_text})")
    print(f"PyTorch index: {index}")
    print(f"CuPy package: {cupy_package}")

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

    run([python, "-m", "pip", "uninstall", "-y", "cupy-cuda11x", "cupy-cuda12x"], check=False)
    run([python, "-m", "pip", "install", cupy_package])
    run([python, "-m", "pip", "install", "-r", _filtered_requirements()])
    run([python, "-m", "pip", "install", "gdown", "huggingface_hub>=0.25,<1"])
    validate_cuda_runtime(python, expected_cuda, require_cupy=True)
    install_optional_xformers(python, "0.0.22.post7")
    run([python, "-m", "pip", "check"])
    return variant, expected_cuda


def download_model(python: Path, variant: str) -> Path:
    checkpoint_dir = SOURCE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"{variant}.ckpt"
    if destination.is_file() and destination.stat().st_size > 1024 * 1024:
        print(f"Already present: {destination}")
        return destination
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={MODEL_REPOSITORY!r}, filename={variant + '.ckpt'!r}, "
        f"local_dir={str(checkpoint_dir)!r}))"
    )
    run([python, "-c", code])
    if not destination.is_file():
        raise RuntimeError(f"MoG checkpoint download did not create {destination}")
    return destination


def download_flow_checkpoint(python: Path) -> Path:
    checkpoint_dir = SOURCE / "emavfi" / "ckpt"
    upstream_destination = checkpoint_dir / "ours_t.pkl"
    compatibility_destination = checkpoint_dir / "ours_t.ckpt"

    # Upstream's README calls this file ours_t.ckpt, but the Google Drive folder
    # actually contains ours_t.pkl and emavfi/Trainer.py loads ours_t.pkl. Keep the
    # real upstream filename and a compatibility copy for our worker path.
    if upstream_destination.is_file() and upstream_destination.stat().st_size > 1024 * 1024:
        if (
            not compatibility_destination.is_file()
            or compatibility_destination.stat().st_size != upstream_destination.stat().st_size
        ):
            shutil.copy2(upstream_destination, compatibility_destination)
        print(f"Already present: {upstream_destination}")
        return upstream_destination

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mog_flow_") as temp:
        temp_dir = Path(temp)
        run([python, "-m", "gdown", "--folder", FLOW_FOLDER, "-O", temp_dir])
        candidates = list(temp_dir.rglob("ours_t.pkl"))
        if not candidates:
            candidates = list(temp_dir.rglob("ours_t.ckpt"))
        if not candidates:
            raise RuntimeError(
                "MoG flow checkpoint download completed but neither ours_t.pkl nor "
                "ours_t.ckpt was found. Set MOG_FLOW_CHECKPOINT to an existing copy "
                "if Google Drive blocks the folder download."
            )
        shutil.copy2(candidates[0], upstream_destination)
        shutil.copy2(candidates[0], compatibility_destination)
    return upstream_destination


def validate_runtime(python: Path, checkpoints: list[Path], flow_checkpoint: Path) -> None:
    """Validate imports and both checkpoint formats without allocating the full diffusion model on VRAM."""
    code = f"""
import os
import sys
from pathlib import Path
import torch
import cupy
import decord
import einops
import omegaconf
import pytorch_lightning
import transformers

source = Path({str(SOURCE)!r})
os.chdir(source)
sys.path.insert(0, str(source))
import scripts.evaluation.inference
from emavfi.Trainer import Model

flow = Path({str(flow_checkpoint)!r})
state = torch.load(str(flow), map_location='cpu')
assert isinstance(state, dict) and state, 'MoG flow checkpoint is empty or invalid'
del state
for raw in {repr([str(path) for path in checkpoints])}:
    path = Path(raw)
    try:
        value = torch.load(str(path), map_location='cpu', mmap=True)
    except TypeError:
        value = torch.load(str(path), map_location='cpu')
    assert isinstance(value, dict) and value, f'MoG checkpoint is empty or invalid: {{path}}'
    del value
print('MoG runtime imports: ok')
print('MoG flow checkpoint CPU load: ok')
print('MoG diffusion checkpoint CPU load: ok')
print('MoG CuPy devices:', cupy.cuda.runtime.getDeviceCount())
try:
    import xformers
    print('MoG attention: xFormers enabled')
except Exception:
    print('MoG attention: PyTorch fallback')
"""
    run([python, "-c", code])


def main() -> None:
    args = parse_args()
    print("Installing Motion-Aware Generative Frame Interpolation")
    ensure_repo(REPOSITORY, SOURCE)
    python = ensure_venv(VENV, choose_python())
    cuda_variant, expected_cuda = install_requirements(python)

    variants = ("ani", "real") if args.variant == "both" else (args.variant,)
    checkpoints = [download_model(python, variant) for variant in variants]
    flow_checkpoint = download_flow_checkpoint(python)
    validate_runtime(python, checkpoints, flow_checkpoint)

    print("\nMoG setup complete.")
    print(f"Python: {python}")
    print(f"Source: {SOURCE}")
    print(f"Installed variant(s): {', '.join(variants)}")
    print(f"CUDA: {cuda_variant} / PyTorch runtime {expected_cuda}")
    print("Runtime preflight: CUDA, CuPy, imports and checkpoint CPU loads all passed.")
    print("The released MoG checkpoints are very large. 4 GB GPUs may still fail during full model placement; the worker will report that as a hardware memory limit rather than an installation error.")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
