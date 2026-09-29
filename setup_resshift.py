#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setup_engine_common import ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "Multi-Input-Resshift-Diffusion-VFI"
VENV = ROOT / ".resshift-venv"
MODEL_DIR = SOURCE / "_webp_model"
REPOSITORY = "https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI.git"
MODEL_REPOSITORY = "vfontech/Multiple-Input-Resshift-VFI"

CUDA_VARIANTS = {
    "cu118": {
        "torch_index": "https://download.pytorch.org/whl/cu118",
        "torch_cuda": "11.8",
        "cupy": "cupy-cuda11x>=12.0.0",
    },
    "cu124": {
        "torch_index": "https://download.pytorch.org/whl/cu124",
        "torch_cuda": "12.4",
        "cupy": "cupy-cuda12x>=12.0.0",
    },
}

RUNTIME_REQUIREMENTS = (
    "safetensors>=0.4.3",
)


def choose_python() -> str:
    explicit = os.environ.get("RESSHIFT_BOOTSTRAP_PYTHON")
    candidates = [explicit] if explicit else []
    candidates += ["python3.12", "python3.11", "python3.10", sys.executable]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).is_file():
            return str(resolved)
    raise RuntimeError("Could not find Python 3.10+ for the ResShift environment.")


def _driver_cuda_version() -> tuple[int, int] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", f"{result.stdout}\n{result.stderr}")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _variant_from_index(index: str) -> str | None:
    text = str(index).lower()
    if "cu118" in text or "cuda11" in text:
        return "cu118"
    if "cu124" in text or "cuda12" in text:
        return "cu124"
    return None


def choose_cuda_variant() -> tuple[str, str, str]:
    requested = os.environ.get("RESSHIFT_CUDA_VARIANT", "").strip().lower()
    explicit_index = os.environ.get("RESSHIFT_TORCH_INDEX", "").strip()

    if requested:
        if requested not in CUDA_VARIANTS:
            valid = ", ".join(sorted(CUDA_VARIANTS))
            raise RuntimeError(f"Unknown RESSHIFT_CUDA_VARIANT={requested!r}; choose one of: {valid}.")
        variant = requested
    elif explicit_index:
        variant = _variant_from_index(explicit_index) or "cu118"
    else:
        driver = _driver_cuda_version()
        variant = "cu118" if driver and driver[0] < 12 else "cu124"
        if driver is None:
            variant = "cu118"

    metadata = CUDA_VARIANTS[variant]
    index = explicit_index or str(metadata["torch_index"])
    cupy = os.environ.get("RESSHIFT_CUPY_PACKAGE", str(metadata["cupy"])).strip()
    return variant, index, cupy


def _installed_torch_cuda(python: Path) -> str | None:
    result = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _filtered_requirements() -> Path:
    source = SOURCE / "requirements.txt"
    destination = VENV / "resshift-upstream-requirements.txt"
    lines: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip().lower()
        package = stripped.split("#", 1)[0].strip()
        if package.startswith("torch") or package.startswith("torchvision") or package.startswith("cupy-cuda"):
            continue
        lines.append(raw)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def _validate_cuda_stack(python: Path, expected_cuda: str) -> None:
    code = (
        "import torch, cupy; "
        "print('torch', torch.__version__); "
        "print('torch CUDA', torch.version.cuda); "
        "print('cuda available', torch.cuda.is_available()); "
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        "print('cupy devices', cupy.cuda.runtime.getDeviceCount() if torch.cuda.is_available() else 0); "
        f"assert str(torch.version.cuda or '').startswith({expected_cuda!r}), 'wrong PyTorch CUDA runtime'; "
        "assert torch.cuda.is_available(), 'PyTorch cannot initialise CUDA with the installed NVIDIA driver'"
    )
    run([python, "-c", code])


def _download_model(python: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={MODEL_REPOSITORY!r}, local_dir={str(MODEL_DIR)!r})"
    )
    run([python, "-c", code])


def _validate_runtime(python: Path) -> None:
    """Exercise the actual runtime dependency graph before setup reports success."""
    code = f"""
import gc
import os
import sys
from pathlib import Path

import cupy
import kornia
import numpy
import safetensors
import torch
from PIL import Image

source = Path({str(SOURCE)!r})
model_dir = Path({str(MODEL_DIR)!r})
os.chdir(source)
sys.path.insert(0, str(source))

from model.hub import MultiInputResShiftHub
from modules.cupy_module.nedt import NEDT

print('ResShift runtime imports: ok')
model = MultiInputResShiftHub.from_pretrained(str(model_dir))
print('ResShift checkpoint load: ok')

model = model.to('cuda').eval().requires_grad_(False)
torch.cuda.synchronize()
print('ResShift CUDA model placement: ok')

probe = torch.zeros((1, 3, 16, 16), device='cuda', dtype=torch.float32)
with torch.inference_mode():
    result = NEDT().to('cuda')(probe)
torch.cuda.synchronize()
print('ResShift CuPy/NVRTC kernel: ok', tuple(result.shape))

del result, probe, model
gc.collect()
torch.cuda.empty_cache()
"""
    run([python, "-c", code])


def main() -> None:
    print("Installing Multi-Input ResShift Diffusion VFI")
    ensure_repo(REPOSITORY, SOURCE)
    python = ensure_venv(VENV, choose_python())

    variant, torch_index, cupy_package = choose_cuda_variant()
    expected_cuda = str(CUDA_VARIANTS[variant]["torch_cuda"])
    driver = _driver_cuda_version()
    driver_text = f"{driver[0]}.{driver[1]}" if driver else "unknown"
    print(f"ResShift CUDA stack: {variant} (driver reports CUDA {driver_text})")
    print(f"PyTorch index: {torch_index}")
    print(f"CuPy package: {cupy_package}")

    current_cuda = _installed_torch_cuda(python)
    torch_command = [python, "-m", "pip", "install"]
    if current_cuda != expected_cuda:
        torch_command.append("--force-reinstall")
    torch_command += [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "--index-url",
        torch_index,
    ]
    run(torch_command)

    run([
        python,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "cupy-cuda11x",
        "cupy-cuda12x",
    ], check=False)
    run([python, "-m", "pip", "install", cupy_package])
    run([python, "-m", "pip", "install", "-r", _filtered_requirements()])
    run([python, "-m", "pip", "install", *RUNTIME_REQUIREMENTS])
    run([python, "-m", "pip", "check"])

    _validate_cuda_stack(python, expected_cuda)
    _download_model(python)
    _validate_runtime(python)

    print("\nResShift setup complete.")
    print(f"Python: {python}")
    print(f"Source: {SOURCE}")
    print(f"Model:  {MODEL_DIR}")
    print(f"CUDA:   {variant} / PyTorch runtime {expected_cuda}")
    print("Runtime preflight: imports, checkpoint load, CUDA placement and CuPy kernel all passed.")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
