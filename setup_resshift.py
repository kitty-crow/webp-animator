#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setup_engine_common import ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "Multi-Input-Resshift-Diffusion-VFI"
VENV = ROOT / ".resshift-venv"
MODEL_DIR = SOURCE / "_webp_model"
REPOSITORY = "https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI.git"
MODEL_REPOSITORY = "vfontech/Multiple-Input-Resshift-VFI"


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


def main() -> None:
    print("Installing Multi-Input ResShift Diffusion VFI")
    ensure_repo(REPOSITORY, SOURCE)
    python = ensure_venv(VENV, choose_python())

    # Install a CUDA-enabled PyTorch first. The upstream requirements accept newer
    # compatible builds, so the following requirements pass keeps the isolated
    # environment reproducible without replacing the selected CUDA wheel.
    run([
        python,
        "-m",
        "pip",
        "install",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "--index-url",
        os.environ.get("RESSHIFT_TORCH_INDEX", "https://download.pytorch.org/whl/cu124"),
    ])
    run([python, "-m", "pip", "install", "-r", SOURCE / "requirements.txt"])

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
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
