#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command, *, cwd=None):
    print("+", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_repo(url: str, directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    if (directory / ".git").is_dir():
        run(["git", "pull", "--ff-only"], cwd=directory)
    else:
        run(["git", "clone", "--depth", "1", url, str(directory)])


def ensure_venv(directory: Path) -> Path:
    python = venv_python(directory)
    if not python.is_file():
        run([sys.executable, "-m", "venv", str(directory)])
    run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    return python


def pip_install(python: Path, *args: str) -> None:
    run([python, "-m", "pip", "install", *args])


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 1024:
        print(f"Checkpoint already present: {destination}")
        return
    print(f"Downloading {url}\n       -> {destination}")
    temp = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temp)
        temp.replace(destination)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
