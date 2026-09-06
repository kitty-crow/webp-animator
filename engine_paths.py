from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def venv_python(venv: Path) -> Path:
    """Return the expected Python executable for a venv on Windows or POSIX."""
    candidates = (
        venv / "Scripts" / "python.exe",
        venv / "Scripts" / "python",
        venv / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[-1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def amt_paths():
    python = _path_from_env("AMT_PYTHON", venv_python(ROOT / ".amt-venv"))
    source = _path_from_env("AMT_DIR", ROOT / "third_party" / "AMT")
    config = _path_from_env("AMT_CONFIG", source / "cfgs" / "AMT-S.yaml")
    checkpoint = _path_from_env("AMT_CHECKPOINT", source / "pretrained" / "amt-s.pth")
    ready = python.is_file() and source.is_dir() and config.is_file() and checkpoint.is_file()
    return ready, python, source, config, checkpoint


def eden_paths():
    python = _path_from_env("EDEN_PYTHON", venv_python(ROOT / ".eden-venv"))
    source = _path_from_env("EDEN_DIR", ROOT / "third_party" / "EDEN")
    config = _path_from_env("EDEN_CONFIG", source / "configs" / "eval_eden.yaml")
    checkpoint = _path_from_env("EDEN_CHECKPOINT", source / "checkpoints" / "eden.pt")
    ready = python.is_file() and source.is_dir() and config.is_file() and checkpoint.is_file()
    return ready, python, source, config, checkpoint


def speed_paths():
    python = _path_from_env("SPEED_PYTHON", venv_python(ROOT / ".speed-venv"))
    source = _path_from_env("SPEED_DIR", ROOT / "third_party" / "SPEED")
    config = _path_from_env("SPEED_CONFIG", source / "configs" / "eval_config.yaml")
    checkpoint = _path_from_env("SPEED_CHECKPOINT", source / "ckpts" / "speed.pt")
    ready = python.is_file() and source.is_dir() and config.is_file() and checkpoint.is_file()
    return ready, python, source, config, checkpoint


def acceleration_status() -> dict:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda else None
        return {
            "torch": True,
            "cuda": cuda,
            "device": device_name,
            "version": str(torch.__version__),
        }
    except Exception as exc:
        return {
            "torch": False,
            "cuda": False,
            "device": None,
            "version": None,
            "error": str(exc),
        }


def engine_status(legacy=None) -> dict:
    result = {"acceleration": acceleration_status()}

    if legacy is not None:
        try:
            ready, python, source, model = legacy.rife_paths()
            result["rife"] = {
                "ready": bool(ready),
                "python": str(python),
                "source": str(source),
                "checkpoint": str(model),
            }
        except Exception as exc:
            result["rife"] = {"ready": False, "error": str(exc)}

    ready, python, source, config, checkpoint = amt_paths()
    result["amt"] = {
        "ready": bool(ready),
        "python": str(python),
        "source": str(source),
        "config": str(config),
        "checkpoint": str(checkpoint),
    }

    ready, python, source, config, checkpoint = eden_paths()
    result["eden"] = {
        "ready": bool(ready),
        "python": str(python),
        "source": str(source),
        "config": str(config),
        "checkpoint": str(checkpoint),
    }

    ready, python, source, config, checkpoint = speed_paths()
    result["speed"] = {
        "ready": bool(ready),
        "python": str(python),
        "source": str(source),
        "config": str(config),
        "checkpoint": str(checkpoint),
    }
    return result
