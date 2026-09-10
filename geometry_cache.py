from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image


CACHE_VERSION = "geometry-cache-v1"
_LOCK = threading.RLock()
_TLS = threading.local()
_GEOMETRY_KEYS = (
    "geometry_mode",
    "axis",
    "max_shift",
    "sigma",
    "alpha_threshold",
)


def geometry_settings(settings: dict | None) -> dict:
    value = settings if isinstance(settings, dict) else {}
    return {key: value.get(key) for key in _GEOMETRY_KEYS}


def cache_key(paths: Sequence[Path], settings: dict | None) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            geometry_settings(settings),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    for ordinal, path in enumerate(paths):
        digest.update(str(ordinal).encode("ascii"))
        digest.update(b":")
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def cache_root_for_paths(paths: Sequence[Path]) -> Path | None:
    """Return the durable job-local cache root for source or analysis snapshots."""
    if not paths:
        return None
    try:
        resolved = Path(paths[0]).resolve()
    except Exception:
        return None
    for parent in (resolved.parent, *resolved.parents):
        try:
            if parent.parent.name == ".webp-jobs":
                return parent / "geometry-cache"
        except Exception:
            continue
    return None


def _entry(cache_root: Path, key: str) -> Path:
    return Path(cache_root) / key


def load(cache_root: Path, key: str) -> list[Image.Image] | None:
    root = _entry(cache_root, key)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("version") != CACHE_VERSION or value.get("key") != key:
            return None
        count = int(value.get("frame_count", -1))
        if count < 0:
            return None
        frames: list[Image.Image] = []
        for index in range(count):
            path = root / f"{index:06d}.png"
            if not path.is_file():
                return None
            with Image.open(path) as image:
                frames.append(image.convert("RGBA").copy())
        return frames
    except Exception:
        return None


def save(cache_root: Path, key: str, frames: Sequence[Image.Image], settings: dict | None) -> Path:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = _entry(cache_root, key)
    with _LOCK:
        if (target / "manifest.json").is_file():
            return target
        staging = cache_root / f".{key}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for index, image in enumerate(frames):
                image.convert("RGBA").save(staging / f"{index:06d}.png", format="PNG")
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": CACHE_VERSION,
                        "key": key,
                        "frame_count": len(frames),
                        "geometry": geometry_settings(settings),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                os.replace(staging, target)
            except (FileExistsError, PermissionError, OSError):
                if not (target / "manifest.json").is_file():
                    raise
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def normalise_with_cache(
    normalise_fn: Callable,
    legacy,
    images: list[Image.Image],
    source_paths: Sequence[Path],
    settings: dict,
    cache_root: Path | None,
    progress=None,
    *,
    cache_label: str = "gap analysis",
) -> tuple[list[Image.Image], dict]:
    """Normalise geometry once and optionally persist/reuse its exact output."""
    if cache_root is None:
        return normalise_fn(legacy, images, settings, progress), {"hit": False, "key": None}

    key = cache_key(source_paths, settings)
    cached = load(cache_root, key)
    if cached is not None and len(cached) == len(images):
        if progress:
            progress(40, f"Reusing frame matching from {cache_label}")
        return cached, {"hit": True, "key": key}

    frames = normalise_fn(legacy, images, settings, progress)
    save(cache_root, key, frames, settings)
    return frames, {"hit": False, "key": key}


def _push_context(paths: Sequence[Path], label: str):
    previous = (
        getattr(_TLS, "paths", None),
        getattr(_TLS, "cache_root", None),
        getattr(_TLS, "label", None),
    )
    _TLS.paths = [Path(path) for path in paths]
    _TLS.cache_root = cache_root_for_paths(paths)
    _TLS.label = label
    return previous


def _pop_context(previous) -> None:
    for name, value in zip(("paths", "cache_root", "label"), previous):
        if value is None:
            try:
                delattr(_TLS, name)
            except AttributeError:
                pass
        else:
            setattr(_TLS, name, value)


def install(advanced_pipeline_module, temporal_v2_module) -> None:
    """Persist analysis geometry and transparently reuse it during generation."""
    if getattr(advanced_pipeline_module, "_geometry_cache_installed", False):
        return
    advanced_pipeline_module._geometry_cache_installed = True

    original_normalise = advanced_pipeline_module._normalise_geometry
    original_analyse = temporal_v2_module.analyse_paths
    original_process = temporal_v2_module.process_job

    def cached_normalise(legacy, images, settings, progress=None):
        paths = getattr(_TLS, "paths", None)
        cache_root = getattr(_TLS, "cache_root", None)
        label = getattr(_TLS, "label", "gap analysis")
        if not paths or len(paths) != len(images):
            return original_normalise(legacy, images, settings, progress)
        frames, state = normalise_with_cache(
            original_normalise,
            legacy,
            images,
            paths,
            settings,
            cache_root,
            progress,
            cache_label=label,
        )
        _TLS.last_cache_state = state
        return frames

    def analyse_paths(legacy, paths, settings, *args, **kwargs):
        previous = _push_context(paths, "gap analysis")
        try:
            return original_analyse(legacy, paths, settings, *args, **kwargs)
        finally:
            _pop_context(previous)

    def process_job(legacy, job_id, paths, settings):
        previous = _push_context(paths, "gap analysis")
        try:
            return original_process(legacy, job_id, paths, settings)
        finally:
            _pop_context(previous)

    advanced_pipeline_module._normalise_geometry = cached_normalise
    temporal_v2_module.analyse_paths = analyse_paths
    temporal_v2_module.process_job = process_job
