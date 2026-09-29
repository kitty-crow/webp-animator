from __future__ import annotations


def probe_xformers_attention(torch) -> tuple[bool, str]:
    """Return whether xFormers can execute its actual CUDA attention kernel."""
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    try:
        import xformers.ops
    except Exception as exc:
        return False, f"xFormers unavailable: {exc.__class__.__name__}"

    q = None
    out = None
    try:
        with torch.inference_mode():
            q = torch.randn((2, 16, 32), device="cuda", dtype=torch.float16)
            out = xformers.ops.memory_efficient_attention(q, q, q)
            torch.cuda.synchronize()
        return True, "kernel probe passed"
    except Exception as exc:
        return False, f"kernel probe failed: {exc.__class__.__name__}: {exc}"
    finally:
        del out, q
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def set_first_stage_attention(model_config, *, use_xformers: bool) -> str:
    """Force the VAE to a backend that really exists on the current runtime.

    MoG and ToonCrafter upstream default their dual-reference VAE to
    ``vanilla-xformers`` even when importing xFormers failed.  Their code logs a
    fallback message but still instantiates MemoryEfficientAttnBlock, which later
    raises ``NameError: xformers is not defined`` during decode.  PyTorch >=2 ships
    scaled-dot-product attention, so ``vanilla`` is the correct fallback and keeps
    identical q/k/v/proj parameter names for strict checkpoint loading.
    """
    try:
        ddconfig = model_config["params"]["first_stage_config"]["params"]["ddconfig"]
    except Exception as exc:
        raise RuntimeError(
            "Video diffusion config is missing first_stage_config.params.ddconfig; "
            "cannot select a safe VAE attention backend."
        ) from exc

    selected = "vanilla-xformers" if use_xformers else "vanilla"
    ddconfig["attn_type"] = selected
    return selected


def configure_first_stage_attention(torch, model_config, *, label: str) -> str:
    usable, reason = probe_xformers_attention(torch)
    selected = set_first_stage_attention(model_config, use_xformers=usable)
    backend = "xFormers" if usable else "PyTorch SDPA"
    print(f"{label} VAE attention: {backend} ({reason})", flush=True)
    return selected


def validate_first_stage_attention(model, *, selected: str, label: str) -> None:
    """Fail during model load if upstream ignored our fallback selection."""
    first_stage = getattr(model, "first_stage_model", None)
    if first_stage is None:
        raise RuntimeError(f"{label} model is missing first_stage_model after construction.")

    memory_efficient = [
        name or "<root>"
        for name, module in first_stage.named_modules()
        if module.__class__.__name__ == "MemoryEfficientAttnBlock"
    ]
    if selected == "vanilla" and memory_efficient:
        preview = ", ".join(memory_efficient[:6])
        raise RuntimeError(
            f"{label} requested the PyTorch VAE attention fallback, but upstream still "
            f"constructed MemoryEfficientAttnBlock at: {preview}."
        )
