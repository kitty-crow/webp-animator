from __future__ import annotations

import ctypes
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_GIB = 1024 ** 3


def model_nbytes(model) -> int:
    """Return resident parameter+buffer bytes without materialising new tensors."""
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        try:
            pointer = int(tensor.data_ptr())
        except Exception:
            pointer = id(tensor)
        if pointer in seen:
            continue
        seen.add(pointer)
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def available_ram_bytes() -> int:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass

    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 0


def _normalise_mode(value: object) -> str:
    mode = str(value or "auto").strip().lower()
    aliases = {
        "ram": "cpu", "host": "cpu",
        "ssd": "disk", "hdd": "disk",
        "resident": "gpu", "cuda": "gpu",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "gpu", "cpu", "disk"}:
        raise RuntimeError(
            f"Unknown model offload mode {mode!r}; choose auto, gpu, cpu/ram, or disk."
        )
    return mode


def choose_offload_mode(model, device, *, env_var: str) -> tuple[str, dict[str, Any]]:
    import torch

    requested = _normalise_mode(os.environ.get(env_var, "auto"))
    weight_bytes = model_nbytes(model)
    total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    try:
        free_gpu, _ = torch.cuda.mem_get_info(device)
        free_gpu = int(free_gpu)
    except Exception:
        free_gpu = total_gpu
    ram_bytes = available_ram_bytes()

    gpu_margin = max(768 * 1024**2, int(weight_bytes * 0.25))
    gpu_resident_ok = weight_bytes + gpu_margin <= free_gpu
    ram_margin = max(2 * _GIB, int(weight_bytes * 0.35))
    cpu_offload_ok = ram_bytes <= 0 or weight_bytes + ram_margin <= ram_bytes

    if requested == "auto":
        mode = "gpu" if gpu_resident_ok else ("cpu" if cpu_offload_ok else "disk")
    else:
        mode = requested

    return mode, {
        "weights": weight_bytes,
        "gpu": total_gpu,
        "gpu_free": free_gpu,
        "ram_available": ram_bytes,
        "gpu_resident_ok": gpu_resident_ok,
        "cpu_offload_ok": cpu_offload_ok,
        "requested": requested,
    }


def _sync_execution_device(model, device) -> None:
    try:
        modules = model.modules()
    except Exception:
        modules = (model,)
    for module in modules:
        if hasattr(module, "_device"):
            try:
                module._device = device
            except Exception:
                pass


def _stage_buffers(model, device) -> int:
    moved = 0
    for module in model.modules():
        for name, value in list(module._buffers.items()):
            if value is None:
                continue
            try:
                replacement = value.to(device)
            except Exception:
                continue
            module._buffers[name] = replacement
            moved += int(replacement.numel()) * int(replacement.element_size())
    return moved


def _ensure_accelerate():
    try:
        from accelerate import cpu_offload, disk_offload
        return cpu_offload, disk_offload
    except Exception:
        print("MODEL_OFFLOAD installing Hugging Face Accelerate support once", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "accelerate==0.25.0"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "RAM/disk model offload requires Hugging Face Accelerate. Automatic "
                "installation failed; run this engine's setup again or install "
                "accelerate==0.25.0 in its isolated environment."
            )
        from accelerate import cpu_offload, disk_offload
        return cpu_offload, disk_offload


def _disk_offload(model, device, *, source: Path, checkpoint: Path, namespace: str, stats: dict[str, Any]):
    _cpu_offload, disk_offload = _ensure_accelerate()

    explicit = os.environ.get(f"{namespace.upper()}_OFFLOAD_DIR", "").strip()
    offload_root = Path(explicit).expanduser().resolve() if explicit else source / "_webp_offload" / checkpoint.stem
    offload_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(offload_root).free
    required = max(int(stats["weights"] * 1.15), stats["weights"] + 512 * 1024**2)
    if free < required:
        raise RuntimeError(
            f"{namespace} needs about {required / _GIB:.1f} GB free for disk offload, "
            f"but only {free / _GIB:.1f} GB is available at {offload_root}."
        )
    model = disk_offload(
        model,
        offload_dir=offload_root,
        execution_device=device,
        offload_buffers=False,
    )
    setattr(model, "_webp_offload_dir", str(offload_root))
    return model


def apply_model_offload(
    model,
    device,
    *,
    source: Path,
    checkpoint: Path,
    env_var: str,
    namespace: str,
):
    mode, stats = choose_offload_mode(model, device, env_var=env_var)
    weight_gb = stats["weights"] / _GIB
    gpu_gb = stats["gpu"] / _GIB
    free_gpu_gb = stats["gpu_free"] / _GIB
    ram_gb = stats["ram_available"] / _GIB if stats["ram_available"] else 0.0

    if mode == "gpu":
        model = model.to(device)
        buffer_bytes = 0
    else:
        cpu_offload, _disk = _ensure_accelerate()

        if mode == "cpu":
            try:
                model = cpu_offload(model, execution_device=device, offload_buffers=False)
            except (MemoryError, OSError, RuntimeError) as exc:
                if stats.get("requested") != "auto":
                    raise
                print(
                    f"{namespace.upper()}_OFFLOAD RAM path failed ({exc}); falling back to disk",
                    flush=True,
                )
                gc.collect()
                model = _disk_offload(
                    model, device, source=source, checkpoint=checkpoint,
                    namespace=namespace, stats=stats,
                )
                mode = "disk"
        else:
            model = _disk_offload(
                model, device, source=source, checkpoint=checkpoint,
                namespace=namespace, stats=stats,
            )
        buffer_bytes = _stage_buffers(model, device)

    _sync_execution_device(model, device)
    setattr(model, "_webp_execution_device", device)
    setattr(model, "_webp_offload_mode", mode)
    setattr(model, "_webp_model_weight_bytes", int(stats["weights"]))
    print(
        f"{namespace.upper()}_OFFLOAD mode={mode} weights={weight_gb:.2f}GB "
        f"gpu={gpu_gb:.2f}GB free={free_gpu_gb:.2f}GB "
        f"ram_available={ram_gb:.2f}GB cuda_buffers={buffer_bytes / 1024**2:.1f}MB",
        flush=True,
    )
    return model, mode, stats
