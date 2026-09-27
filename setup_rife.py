#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from setup_engine_common import (
    choose_torch_cuda_variant,
    installed_torch_cuda,
    run,
    validate_cuda_runtime,
)

ROOT = Path(__file__).resolve().parent
RIFE_DIR = ROOT / "third_party" / "Practical-RIFE"
VENV_DIR = ROOT / ".rife-venv"
MODEL_DIR = RIFE_DIR / "train_log"
MODEL_DRIVE_ID = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"

CUDA_VARIANTS = {
    "cu118": {
        "index": "https://download.pytorch.org/whl/cu118",
        "runtime": "11.8",
    },
    "cu124": {
        "index": "https://download.pytorch.org/whl/cu124",
        "runtime": "12.4",
    },
}


def python_version(executable: str):
    result = subprocess.run(
        [executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = result.stdout.strip().split(".")
    return int(major), int(minor)


def choose_python():
    explicit = os.environ.get("RIFE_BOOTSTRAP_PYTHON")
    candidates = [explicit] if explicit else []
    candidates += ["python3.11", "python3.10", sys.executable]

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).exists():
            return str(resolved)

    raise RuntimeError("Could not find a Python interpreter for the RIFE environment.")


def venv_python() -> Path:
    candidates = (
        VENV_DIR / "Scripts" / "python.exe",
        VENV_DIR / "Scripts" / "python",
        VENV_DIR / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[-1]


def install_repo():
    RIFE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if (RIFE_DIR / ".git").is_dir():
        run(["git", "pull", "--ff-only"], cwd=RIFE_DIR)
    else:
        run([
            "git", "clone", "--depth", "1",
            "https://github.com/hzwer/Practical-RIFE.git",
            str(RIFE_DIR),
        ])


def create_environment(bootstrap_python: str):
    version = python_version(bootstrap_python)
    if version > (3, 11):
        print(
            f"WARNING: Practical-RIFE currently documents Python <= 3.11; "
            f"using Python {version[0]}.{version[1]} because no older interpreter was found.\n"
            "If installation fails, install Python 3.11 and rerun with:\n"
            "  RIFE_BOOTSTRAP_PYTHON=python3.11 python setup_rife.py"
        )

    python = venv_python()
    if not python.is_file():
        run([bootstrap_python, "-m", "venv", str(VENV_DIR)])
        python = venv_python()
    if not python.is_file():
        raise RuntimeError(f"RIFE virtual environment was created but Python was not found under {VENV_DIR}")

    run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    variant, driver = choose_torch_cuda_variant("RIFE_CUDA_VARIANT", cuda12_variant="cu124")
    metadata = CUDA_VARIANTS[variant]
    index = os.environ.get("RIFE_TORCH_INDEX", str(metadata["index"]))
    expected_cuda = str(metadata["runtime"])
    driver_text = f"{driver[0]}.{driver[1]}" if driver else "unknown"
    print(f"RIFE CUDA stack: {variant} (driver reports CUDA {driver_text})")

    command = [python, "-m", "pip", "install"]
    if installed_torch_cuda(python) != expected_cuda:
        command.append("--force-reinstall")
    command += [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "--index-url",
        index,
    ]
    run(command)
    run([
        python,
        "-m",
        "pip",
        "install",
        "opencv-python-headless",
        "Pillow",
        "numpy<2.3",
        "gdown",
    ])
    run([python, "-m", "pip", "check"])
    validate_cuda_runtime(python, expected_cuda)
    return python, variant, expected_cuda


def install_model(python: Path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (MODEL_DIR / "flownet.pkl").is_file():
        print("RIFE 4.25 model weights already present.")
        return

    with tempfile.TemporaryDirectory(prefix="rife425_") as temp:
        temp = Path(temp)
        archive = temp / "RIFEv4.25.zip"
        extracted = temp / "extracted"
        extracted.mkdir()

        run([
            python, "-m", "gdown",
            "--id", MODEL_DRIVE_ID,
            "-O", str(archive),
        ])

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)

        weights = list(extracted.rglob("flownet.pkl"))
        if not weights:
            raise RuntimeError("The downloaded RIFE 4.25 archive did not contain flownet.pkl.")

        source_dir = weights[0].parent
        for item in source_dir.iterdir():
            if item.is_file() and item.suffix.lower() in {".py", ".pkl", ".txt"}:
                shutil.copy2(item, MODEL_DIR / item.name)

    if not (MODEL_DIR / "flownet.pkl").is_file():
        raise RuntimeError("RIFE model installation did not produce train_log/flownet.pkl.")


def validate_runtime(python: Path):
    code = f"""
import os
import sys
from pathlib import Path
import torch
source = Path({str(RIFE_DIR)!r})
model_dir = Path({str(MODEL_DIR)!r})
os.chdir(source)
sys.path.insert(0, str(source))
assert (model_dir / 'flownet.pkl').is_file()
from model.RIFE import Model
print('RIFE runtime import: ok')
"""
    run([python, "-c", code])


def main():
    print("Installing Practical-RIFE 4.25 for WebP Animator")
    install_repo()
    bootstrap_python = choose_python()
    print(f"RIFE bootstrap Python: {bootstrap_python}")
    rife_python, cuda_variant, expected_cuda = create_environment(bootstrap_python)
    install_model(rife_python)
    validate_runtime(rife_python)

    print("\nRIFE setup complete.")
    print(f"RIFE Python: {rife_python}")
    print(f"RIFE source: {RIFE_DIR}")
    print(f"RIFE model:  {MODEL_DIR}")
    print(f"CUDA:        {cuda_variant} / PyTorch runtime {expected_cuda}")
    print("Runtime preflight: CUDA and RIFE import passed.")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
