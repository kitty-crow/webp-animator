from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def run(command, *, cwd=None, check=True):
    command = [str(item) for item in command]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=check)


def choose_python(env_name: str):
    explicit = os.environ.get(env_name)
    candidates = [explicit] if explicit else []
    candidates += ["python3.10", "python3.11", sys.executable]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).is_file():
            return str(resolved)
    raise RuntimeError("Could not find a Python interpreter. Python 3.10 is recommended.")


def venv_python(venv: Path):
    candidates = [
        venv / "Scripts" / "python.exe",
        venv / "Scripts" / "python",
        venv / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[-1]


def ensure_venv(venv: Path, bootstrap_python: str):
    python = venv_python(venv)
    if not python.is_file():
        run([bootstrap_python, "-m", "venv", venv])
    python = venv_python(venv)
    run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    return python


def ensure_repo(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / ".git").is_dir():
        run(["git", "pull", "--ff-only"], cwd=destination)
    else:
        run(["git", "clone", "--depth", "1", url, destination])


def download(url: str, destination: Path):
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}\n        -> {destination}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    os.replace(temporary, destination)


def nvidia_driver_cuda_version() -> tuple[int, int] | None:
    """Return the CUDA compatibility level reported by the installed NVIDIA driver."""
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


def choose_torch_cuda_variant(
    env_name: str,
    *,
    cuda12_variant: str = "cu121",
    default_without_driver: str = "cu118",
) -> tuple[str, tuple[int, int] | None]:
    """Choose a PyTorch CUDA wheel family that the installed driver can initialise.

    Research-model installers must not blindly install CUDA-12 wheels: older but
    otherwise supported Pascal/Turing machines can expose only CUDA 11.x through
    their current driver. An explicit environment override remains available for
    machines where nvidia-smi is unavailable or intentionally masked.
    """
    allowed = {"cu118", cuda12_variant}
    requested = os.environ.get(env_name, "").strip().lower()
    if requested:
        if requested not in allowed:
            raise RuntimeError(
                f"Unknown {env_name}={requested!r}; choose one of: {', '.join(sorted(allowed))}."
            )
        return requested, nvidia_driver_cuda_version()

    driver = nvidia_driver_cuda_version()
    if driver is None:
        variant = default_without_driver
    else:
        variant = "cu118" if driver[0] < 12 else cuda12_variant
    return variant, driver


def installed_torch_cuda(python: Path) -> str | None:
    result = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def validate_cuda_runtime(python: Path, expected_prefix: str, *, require_cupy: bool = False) -> None:
    imports = "import torch"
    cupy_probe = ""
    if require_cupy:
        imports += ", cupy"
        cupy_probe = (
            "; print('cupy devices', cupy.cuda.runtime.getDeviceCount())"
            "; assert cupy.cuda.runtime.getDeviceCount() > 0, 'CuPy cannot see a CUDA device'"
        )
    code = (
        f"{imports}; "
        "print('torch', torch.__version__); "
        "print('torch CUDA', torch.version.cuda); "
        "print('cuda available', torch.cuda.is_available()); "
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); "
        f"assert str(torch.version.cuda or '').startswith({expected_prefix!r}), 'wrong PyTorch CUDA runtime'; "
        "assert torch.cuda.is_available(), 'PyTorch cannot initialise CUDA with the installed NVIDIA driver'"
        + cupy_probe
    )
    run([python, "-c", code])
