from __future__ import annotations

import ctypes
import gc
import os
import shutil
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
    """Best-effort available physical RAM without adding a psutil dependency."""
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
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * size
    except Exception:
        return 0


def _normalise_mode(value: object) -> str:
    mode = str(value or "auto").strip().lower()
    aliases = {
        "ram": "cpu",
        "host": "cpu",
        "ssd": "disk",
        "hdd": "disk",
        "resident": "gpu",
        "cuda": "gpu",
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
    gpu_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    ram_bytes = available_ram_bytes()

    gpu_margin = max(768 * 1024**2, int(weight_bytes * 0.25))
    gpu_resident_ok = weight_bytes + gpu_margin <= gpu_bytes

    ram_margin = max(2 * _GIB, int(weight_bytes * 0.35))
    cpu_offload_ok = ram_bytes <= 0 or weight_bytes + ram_margin <= ram_bytes

    if requested == "auto":
        if gpu_resident_ok:
            mode = "gpu"
        elif cpu_offload_ok:
            mode = "cpu"
        else:
            mode = "disk"
    else:
        mode = requested

    return mode, {
        "weights": weight_bytes,
        "gpu": gpu_bytes,
        "ram_available": ram_bytes,
        "gpu_resident_ok": gpu_resident_ok,
        "cpu_offload_ok": cpu_offload_ok,
        "requested": requested,
    }


def _sync_execution_device(model, device) -> None:
    """Keep Lightning's model.device on CUDA while Accelerate stores weights elsewhere."""
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


def _disk_offload(model, device, *, source: Path, checkpoint: Path, namespace: str, stats: dict[str, Any]):
    from accelerate import disk_offload

    explicit = os.environ.get(f"{namespace.upper()}_OFFLOAD_DIR", "").strip()
    offload_root = (
        Path(explicit).expanduser().resolve()
        if explicit
        else source / "_webp_offload" / checkpoint.stem
    )
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
        offload_buffers=True,
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
    """Place a model on GPU, stream it from RAM, or memory-map it from disk.

    CPU/disk modes use Accelerate hooks: each submodule's parameters are copied to the
    execution GPU only for the forward that needs them and are removed afterwards.
    Disk mode stores the host weights as memory-mapped files so steady-state weights
    need not fit in physical RAM either. The largest individual layer plus its
    activations still has to fit on the GPU.
    """
    import torch

    mode, stats = choose_offload_mode(model, device, env_var=env_var)
    weight_gb = stats["weights"] / _GIB
    gpu_gb = stats["gpu"] / _GIB
    ram_gb = stats["ram_available"] / _GIB if stats["ram_available"] else 0.0

    if mode == "gpu":
        model = model.to(device)
    else:
        try:
            from accelerate import cpu_offload
            from accelerate import disk_offload  # noqa: F401 - validates both APIs exist
        except Exception as exc:
            raise RuntimeError(
                f"{namespace} selected {mode} offload but Hugging Face Accelerate is missing. "
                f"Rerun this engine's setup script."
            ) from exc

        if mode == "cpu":
            try:
                model = cpu_offload(
                    model,
                    execution_device=device,
                    offload_buffers=True,
                )
            except (MemoryError, OSError, RuntimeError) as exc:
                # In auto mode RAM was only an estimate. If host allocation still
                # fails, fall through to memory-mapped disk rather than aborting.
                if stats.get("requested") != "auto":
                    raise
                print(
                    f"{namespace.upper()}_OFFLOAD RAM path failed ({exc}); falling back to disk",
                    flush=True,
                )
                gc.collect()
                model = _disk_offload(
                    model,
                    device,
                    source=source,
                    checkpoint=checkpoint,
                    namespace=namespace,
                    stats=stats,
                )
                mode = "disk"
        else:
            model = _disk_offload(
                model,
                device,
                source=source,
                checkpoint=checkpoint,
                namespace=namespace,
                stats=stats,
            )

    _sync_execution_device(model, device)
    setattr(model, "_webp_execution_device", device)
    setattr(model, "_webp_offload_mode", mode)
    setattr(model, "_webp_model_weight_bytes", int(stats["weights"]))
    print(
        f"{namespace.upper()}_OFFLOAD mode={mode} weights={weight_gb:.2f}GB "
        f"gpu={gpu_gb:.2f}GB ram_available={ram_gb:.2f}GB",
        flush=True,
    )
    return model, mode, stats
