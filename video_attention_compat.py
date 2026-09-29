from __future__ import annotations

import os
from types import MethodType


SDPA_QUERY_CHUNK = max(1, int(os.environ.get("WEBP_VAE_SDPA_QUERY_CHUNK", "512")))


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
    """Force ordinary VAE self-attention onto a backend that really exists.

    MoG and ToonCrafter default their dual-reference VAE to ``vanilla-xformers``
    even when importing xFormers failed. Their code logs a fallback message but
    still instantiates MemoryEfficientAttnBlock, which later raises during decode.
    PyTorch >=2 ships scaled-dot-product attention, so ``vanilla`` is the safe
    fallback and keeps identical q/k/v/proj parameter names for checkpoint loading.

    Their *reference-fusion* attention is a separate hard-coded xFormers path and is
    repaired after model construction by ``validate_first_stage_attention`` below.
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


def _torch_fusion_forward(torch, module, x, context=None, mask=None):
    """PyTorch SDPA equivalent of upstream's hard-coded xFormers fusion block.

    The upstream block forms one query sequence per output frame and gives each
    query access to the spatial tokens from both endpoint-reference features. The
    q/k/v projections and output projection remain the original checkpoint-backed
    modules; only the attention kernel is replaced. Query chunking bounds the math
    backend's peak temporary allocation on Pascal/low-VRAM cards without changing
    which keys each query attends to.
    """
    if context is None:
        raise RuntimeError("Dual-reference VAE fusion attention requires reference context.")
    if mask is not None:
        raise NotImplementedError("Dual-reference VAE fusion masks are not supported upstream.")
    if x.ndim != 4 or context.ndim != 5:
        raise RuntimeError(
            "Unexpected dual-reference VAE shapes: "
            f"x={tuple(x.shape)}, context={tuple(context.shape)}"
        )

    residual = x
    temporal_batch, channels, height, width = x.shape
    ref_batch, ref_channels, reference_count, ref_height, ref_width = context.shape
    if reference_count < 2:
        raise RuntimeError(
            f"Dual-reference VAE expected at least two references, got {reference_count}."
        )
    if ref_batch <= 0 or temporal_batch % ref_batch:
        raise RuntimeError(
            "Dual-reference VAE temporal/reference batch mismatch: "
            f"temporal={temporal_batch}, reference={ref_batch}."
        )

    hidden = module.norm(x)
    hidden = hidden.permute(0, 2, 3, 1).reshape(temporal_batch, height * width, channels)
    q = module.to_q(hidden)

    ref_tokens = ref_height * ref_width
    refs = (
        context.permute(0, 2, 3, 4, 1)
        .contiguous()
        .reshape(ref_batch * reference_count, ref_tokens, ref_channels)
    )
    k = module.to_k(refs).reshape(ref_batch, reference_count, ref_tokens, -1)
    v = module.to_v(refs).reshape(ref_batch, reference_count, ref_tokens, -1)

    repeats = temporal_batch // ref_batch
    key_width = k.shape[-1]
    value_width = v.shape[-1]
    k = torch.cat(
        (
            k[:, 0:1].expand(ref_batch, repeats, ref_tokens, key_width),
            k[:, 1:2].expand(ref_batch, repeats, ref_tokens, key_width),
        ),
        dim=2,
    ).reshape(temporal_batch, ref_tokens * 2, key_width)
    v = torch.cat(
        (
            v[:, 0:1].expand(ref_batch, repeats, ref_tokens, value_width),
            v[:, 1:2].expand(ref_batch, repeats, ref_tokens, value_width),
        ),
        dim=2,
    ).reshape(temporal_batch, ref_tokens * 2, value_width)

    heads = int(module.heads)
    dim_head = int(module.dim_head)

    def split_heads(tensor):
        if tensor.shape[-1] != heads * dim_head:
            raise RuntimeError(
                "Dual-reference VAE attention projection width mismatch: "
                f"got {tensor.shape[-1]}, expected {heads * dim_head}."
            )
        return (
            tensor.reshape(temporal_batch, tensor.shape[1], heads, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    pieces = []
    query_tokens = q.shape[2]
    for start in range(0, query_tokens, SDPA_QUERY_CHUNK):
        stop = min(query_tokens, start + SDPA_QUERY_CHUNK)
        pieces.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:stop],
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        )
    out = torch.cat(pieces, dim=2)
    out = (
        out.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(temporal_batch, query_tokens, heads * dim_head)
    )
    out = module.to_out(out)
    out = out.transpose(1, 2).reshape(temporal_batch, channels, height, width)
    return residual + out


def _install_fusion_sdpa_fallback(torch, first_stage, *, label: str) -> int:
    """Patch upstream's second, hard-coded xFormers VAE attention path in place."""
    patched = 0
    for _name, module in first_stage.named_modules():
        if module.__class__.__name__ != "MemoryEfficientCrossAttentionWrapperFusion":
            continue
        if getattr(module, "__webp_torch_sdpa__", False):
            patched += 1
            continue

        def replacement(self, x, context=None, mask=None, __torch=torch):
            return _torch_fusion_forward(__torch, self, x, context=context, mask=mask)

        module._forward = MethodType(replacement, module)
        module.__webp_torch_sdpa__ = True
        patched += 1

    if patched:
        print(
            f"{label} VAE reference fusion: PyTorch SDPA fallback "
            f"({patched} block(s), query chunk {SDPA_QUERY_CHUNK})",
            flush=True,
        )
    return patched


def validate_first_stage_attention(model, *, selected: str, label: str) -> None:
    """Repair and then validate every VAE attention path selected for this runtime."""
    first_stage = getattr(model, "first_stage_model", None)
    if first_stage is None:
        raise RuntimeError(f"{label} model is missing first_stage_model after construction.")

    if selected == "vanilla":
        # The ordinary midpoint attention obeys ddconfig.attn_type, but upstream's
        # dual-reference refinement is hard-coded to memory-efficient-cross-attn-
        # fusion and calls xFormers regardless of XFORMERS_IS_AVAILABLE. Patch the
        # existing instances so checkpoint parameter names/layout remain untouched.
        import torch

        _install_fusion_sdpa_fallback(torch, first_stage, label=label)

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

    unpatched_fusion = [
        name or "<root>"
        for name, module in first_stage.named_modules()
        if module.__class__.__name__ == "MemoryEfficientCrossAttentionWrapperFusion"
        and selected == "vanilla"
        and not getattr(module, "__webp_torch_sdpa__", False)
    ]
    if unpatched_fusion:
        preview = ", ".join(unpatched_fusion[:6])
        raise RuntimeError(
            f"{label} still has xFormers-only VAE reference-fusion block(s): {preview}."
        )
