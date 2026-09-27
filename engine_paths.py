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


def resshift_paths():
    python = _path_from_env("RESSHIFT_PYTHON", venv_python(ROOT / ".resshift-venv"))
    source = _path_from_env(
        "RESSHIFT_DIR",
        ROOT / "third_party" / "Multi-Input-Resshift-Diffusion-VFI",
    )
    model = _path_from_env("RESSHIFT_MODEL_DIR", source / "_webp_model")
    ready = (
        python.is_file()
        and source.is_dir()
        and (source / "model" / "hub.py").is_file()
        and model.is_dir()
        and (
            (model / "model.safetensors").is_file()
            or (model / "pytorch_model.bin").is_file()
            or any(model.glob("*.safetensors"))
        )
    )
    return ready, python, source, model


def mog_paths(variant: str):
    variant = str(variant).strip().lower()
    if variant not in {"ani", "real"}:
        raise ValueError(f"Unknown MoG variant: {variant}")
    python = _path_from_env("MOG_PYTHON", venv_python(ROOT / ".mog-venv"))
    source = _path_from_env("MOG_DIR", ROOT / "third_party" / "MoG-VFI")
    checkpoint = _path_from_env(
        f"MOG_{variant.upper()}_CHECKPOINT",
        source / "checkpoints" / f"{variant}.ckpt",
    )
    flow_checkpoint = _path_from_env(
        "MOG_FLOW_CHECKPOINT",
        source / "emavfi" / "ckpt" / "ours_t.ckpt",
    )
    config = _path_from_env(
        f"MOG_{variant.upper()}_CONFIG",
        source / "configs" / f"{variant}.yaml",
    )
    ready = (
        python.is_file()
        and source.is_dir()
        and checkpoint.is_file()
        and flow_checkpoint.is_file()
        and config.is_file()
        and (source / "scripts" / "evaluation" / "inference.py").is_file()
    )
    return ready, python, source, config, checkpoint, flow_checkpoint


def acceleration_status() -> dict:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda else None
        device_count = int(torch.cuda.device_count()) if cuda else 0
        devices = [torch.cuda.get_device_name(index) for index in range(device_count)] if cuda else []
        total_vram = []
        if cuda:
            for index in range(device_count):
                try:
                    total_vram.append(int(torch.cuda.get_device_properties(index).total_memory))
                except Exception:
                    total_vram.append(None)
        return {
            "torch": True,
            "cuda": cuda,
            "device": device_name,
            "device_count": device_count,
            "devices": devices,
            "total_vram": total_vram,
            "version": str(torch.__version__),
        }
    except Exception as exc:
        return {
            "torch": False,
            "cuda": False,
            "device": None,
            "device_count": 0,
            "devices": [],
            "total_vram": [],
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

    ready, python, source, model = resshift_paths()
    result["resshift"] = {
        "ready": bool(ready),
        "python": str(python),
        "source": str(source),
        "checkpoint": str(model),
        "advanced": True,
        "setup": "python setup_resshift.py",
    }

    for variant, key in (("ani", "mog_ani"), ("real", "mog_real")):
        ready, python, source, config, checkpoint, flow_checkpoint = mog_paths(variant)
        result[key] = {
            "ready": bool(ready),
            "python": str(python),
            "source": str(source),
            "config": str(config),
            "checkpoint": str(checkpoint),
            "flow_checkpoint": str(flow_checkpoint),
            "advanced": True,
            "setup": f"python setup_mog.py --variant {variant}",
        }
    return result
