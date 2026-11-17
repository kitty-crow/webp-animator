"""Native GPU scheduling and peer-backed model weight storage."""
from __future__ import annotations

import os


_PEER_ENGINES = frozenset({"mog_ani", "mog_real", "tooncrafter"})


def worker_device_masks(engine, tokens, task_count, worker_limit=0, strategy="auto"):
    """Keep independent tasks on independent GPUs, or expose peers to one worker."""
    strategy = str(strategy).strip().lower()
    if strategy not in {"auto", "parallel", "peer"}:
        raise ValueError("GENERATIVE_VFI_GPU_STRATEGY must be auto, parallel or peer")
    devices = list(tokens)
    if not devices:
        return [None]
    peer_job = engine in _PEER_ENGINES and (
        strategy == "peer" or (strategy == "auto" and task_count == 1)
    )
    if peer_job and len(devices) > 1:
        return [",".join(devices)]
    if worker_limit > 0:
        devices = devices[:worker_limit]
    return devices[:max(1, task_count)]


def configured_worker_masks(engine, tokens, task_count):
    return worker_device_masks(
        engine, tokens, task_count,
        int(os.environ.get("GENERATIVE_VFI_GPU_WORKERS", "0") or 0),
        os.environ.get("GENERATIVE_VFI_GPU_STRATEGY", "auto"),
    )


def peer_memory(torch, device):
    """Discover mutually accessible CUDA peers and their currently free bytes."""
    if torch.cuda.device_count() < 2:
        return {}
    primary = device.index if device.index is not None else torch.cuda.current_device()
    result = {}
    for index in range(torch.cuda.device_count()):
        if index == primary:
            continue
        try:
            if not (torch.cuda.can_device_access_peer(primary, index)
                    and torch.cuda.can_device_access_peer(index, primary)):
                continue
            free, _total = torch.cuda.mem_get_info(index)
            if free > 0:
                result[index] = int(free)
        except (RuntimeError, AttributeError, NotImplementedError):
            continue
    return result


def peer_weight_state(torch, model, device, is_oom):
    """Stage parameters on accessible peers; leave weights that do not fit in RAM.

    The returned mapping feeds Accelerate's existing layer-staging hooks.
    CUDA selects the physical peer transfer path. This is weight offload, not
    a single pooled CUDA allocation or distributed activation computation.
    """
    budgets = peer_memory(torch, device)
    if not budgets:
        return None, 0
    state = model.state_dict()
    parameters = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    moved = 0
    for name in sorted(parameters.intersection(state),
                       key=lambda key: state[key].numel() * state[key].element_size(),
                       reverse=True):
        value = state[name]
        size = int(value.numel()) * int(value.element_size())
        if not size:
            continue
        for index in sorted(budgets, key=budgets.get, reverse=True):
            if size > budgets[index]:
                continue
            try:
                state[name] = value.to(torch.device("cuda", index))
            except RuntimeError as exc:
                if not is_oom(torch, exc):
                    raise
                # Another allocation may have consumed the free memory.
                try:
                    budgets[index] = min(budgets[index] - size, int(torch.cuda.mem_get_info(index)[0]))
                except RuntimeError:
                    budgets[index] = 0
                continue
            budgets[index] -= size
            moved += size
            break
    if not moved:
        return None, 0
    # Non-peer weights remain accessible through the same hook-backed mapping.
    for name, value in state.items():
        if torch.is_tensor(value) and value.device.type != "cpu":
            if name not in parameters or value.device.index not in budgets:
                state[name] = value.to("cpu")
    return state, moved
