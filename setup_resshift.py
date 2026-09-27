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
        # CUDA 11.x drivers are the important compatibility case for Pascal-era
        # laptops. PyTorch 2.6 still publishes cu118 wheels and upstream explicitly
        # supports cupy-cuda11x, so prefer that stack instead of forcing CUDA 12.4.
        variant = "cu118" if driver and driver[0] < 12 else "cu124"
        if driver is None:
            # cu118 is the broadest default for the GPUs this project supports.
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

    # A same-version cu124 torch installation satisfies `torch==2.6.0`, so pip
    # will not necessarily replace it with cu118 unless we explicitly force the
    # reinstall when the selected CUDA runtime changes.
    current_cuda = _installed_torch_cuda(python)
    torch_command = [
        python,
        "-m",
        "pip",
        "install",
    ]
    if current_cuda != expected_cuda:
        torch_command.append("--force-reinstall")
    torch_command += [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "--index-url",
        torch_index,
    ]
    run(torch_command)

    # Upstream pins cupy-cuda12x in requirements.txt but explicitly documents
    # cupy-cuda11x for CUDA 11.x. Install the matching wheel ourselves and prevent
    # the later requirements pass from silently reintroducing the wrong runtime.
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

    _validate_cuda_stack(python, expected_cuda)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={MODEL_REPOSITORY!r}, local_dir={str(MODEL_DIR)!r})"
    )
    run([python, "-c", code])

    print("\nResShift setup complete.")
    print(f"Python: {python}")
    print(f"Source: {SOURCE}")
    print(f"Model:  {MODEL_DIR}")
    print(f"CUDA:   {variant} / PyTorch runtime {expected_cuda}")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
