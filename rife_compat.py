from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _venv_python(venv: Path) -> Path:
    candidates = (
        venv / "Scripts" / "python.exe",
        venv / "Scripts" / "python",
        venv / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[-1]


def rife_paths():
    explicit_python = os.environ.get("RIFE_PYTHON")
    rife_python = (
        Path(explicit_python).expanduser()
        if explicit_python
        else _venv_python(ROOT / ".rife-venv")
    )
    rife_dir = Path(
        os.environ.get("RIFE_DIR", ROOT / "third_party" / "Practical-RIFE")
    ).expanduser()
    model_dir = Path(
        os.environ.get("RIFE_MODEL_DIR", rife_dir / "train_log")
    ).expanduser()

    ready = (
        rife_python.is_file()
        and rife_dir.is_dir()
        and model_dir.is_dir()
        and (model_dir / "flownet.pkl").is_file()
    )
    return ready, rife_python, rife_dir, model_dir


def install(legacy) -> None:
    """Replace the legacy POSIX-only RIFE path detector with a cross-platform one."""
    legacy.rife_paths = rife_paths
