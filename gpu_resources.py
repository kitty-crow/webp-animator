"""Hardware-adaptive worker assignment and peer weight storage."""
from __future__ import annotations

import os


PEER_ENGINES = frozenset({"mog_ani", "mog_real", "tooncrafter"})


def worker_device_groups(tokens, task_count, requested=0, *, allow_peer=False):
    """Assign disjoint visible-device groups, retaining an explicit worker cap."""
    if task_count <= 0:
        return []
    tokens = list(dict.fromkeys(tokens))
    if requested > 0:
        tokens = tokens[:requested]
    if not tokens:
        return [None]
    groups = [[token] for token in tokens[:task_count]]
    if allow_peer and peer_offload_enabled():
        for index, token in enumerate(tokens[len(groups):]):
            groups[index % len(groups)].append(token)
    return [",".join(group) for group in groups]


def peer_offload_enabled():
    return os.environ.get("WEBP_PEER_OFFLOAD", "auto").strip().lower() not in {
        "0", "no", "off", "false",
    }


def choose_peer(torch, execution_device, weight_bytes):
    """Find a mutually accessible GPU with room for the complete backing store.

    CUDA determines the transport. Peer capability does not alone prove NVLink
    transport, and aggregate free memory is not one contiguous allocation.
    """
    if not peer_offload_enabled():
        return None
    try:
        primary = torch.device(execution_device).index
        if primary is None:
            primary = torch.cuda.current_device()
        count = torch.cuda.device_count()
    except (AttributeError, RuntimeError, TypeError):
        return None

    candidates = []
    reserve = max(256 * 1024**2, weight_bytes // 10)
    for index in range(count):
        if index == primary:
            continue
        try:
            if not (torch.cuda.can_device_access_peer(primary, index)
                    and torch.cuda.can_device_access_peer(index, primary)):
                continue
            free, _total = torch.cuda.mem_get_info(index)
            if int(free) >= weight_bytes + reserve:
                candidates.append((int(free), index))
        except (AttributeError, RuntimeError, TypeError):
            continue
    return max(candidates)[1] if candidates else None


def diffusion_target_ladder(total_vram):
    """Prefer the highest existing aligned model target on capable devices.

    Keep the original conservative starting points for small GPUs and the
    descending OOM fallback ladder for every GPU.
    """
    gib = total_vram / 1024**3
    if gib <= 4.5:
        return [(256, 128), (192, 128), (128, 128)]
    if gib <= 8.0:
        return [(320, 192), (256, 128), (192, 128), (128, 128)]
    return [(512, 320), (448, 256), (384, 256), (320, 192),
            (256, 128), (192, 128), (128, 128)]
