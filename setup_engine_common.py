from __future__ import annotations

import os
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
    candidates = [venv / "Scripts" / "python.exe", venv / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[1]


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
