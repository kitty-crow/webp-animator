from __future__ import annotations

import io
import math
import os
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops


@dataclass
class DeltaFrame:
    image: Image.Image
    x: int
    y: int
    duration: int
    blend: bool


def _u24(value: int) -> bytes:
    value = max(0, min(0xFFFFFF, int(value)))
    return bytes((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF))


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    if len(fourcc) != 4:
        raise ValueError("RIFF chunk id must be four bytes")
    result = fourcc + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        result += b"\x00"
    return result


def _iter_webp_chunks(data: bytes):
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("Pillow did not return a WebP RIFF stream")
    offset = 12
    while offset + 8 <= len(data):
        fourcc = data[offset:offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(data):
            raise ValueError("Truncated WebP chunk")
        payload = data[start:end]
        yield fourcc, payload
        offset = end + (size & 1)


def _still_payload(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="WEBP",
        lossless=True,
        quality=max(0, min(100, int(quality))),
        method=6,
        exact=True,
    )
    parts = []
    for fourcc, payload in _iter_webp_chunks(buffer.getvalue()):
        if fourcc in {b"ALPH", b"VP8 ", b"VP8L"}:
            parts.append(_chunk(fourcc, payload))
    if not any(part[:4] in {b"VP8 ", b"VP8L"} for part in parts):
        raise ValueError("Still WebP did not contain a VP8/VP8L image payload")
    return b"".join(parts)


def _changed_info_cpu(first: Image.Image, second: Image.Image):
    a = np.asarray(first.convert("RGBA"), dtype=np.uint8)
    b = np.asarray(second.convert("RGBA"), dtype=np.uint8)
    changed = np.any(a != b, axis=2)
    ys, xs = np.nonzero(changed)
    if not xs.size:
        return None, False
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    # Source-over with an opaque new pixel always reproduces that target pixel
    # exactly. This is the safe case where unchanged pixels inside the rectangle
    # can be made transparent and inherited from the prior canvas.
    blend_safe = bool(np.all(b[..., 3][changed] == 255))
    return bbox, blend_safe


def _changed_infos(frames: Sequence[Image.Image]):
    if len(frames) < 2:
        return []

    try:
        import torch

        if torch.cuda.is_available():
            height, width = frames[0].height, frames[0].width
            if all(frame.size == (width, height) for frame in frames):
                target_bytes = int(os.environ.get("WEBP_DELTA_GPU_BYTES", 192 * 1024 * 1024))
                bytes_per_pair = max(1, height * width * 4 * 2)
                batch_size = max(1, min(32, target_bytes // bytes_per_pair))
                device = torch.device("cuda")
                result = []
                try:
                    for start in range(0, len(frames) - 1, batch_size):
                        count = min(batch_size, len(frames) - 1 - start)
                        a_np = np.stack([
                            np.asarray(frames[start + i].convert("RGBA"), dtype=np.uint8)
                            for i in range(count)
                        ])
                        b_np = np.stack([
                            np.asarray(frames[start + i + 1].convert("RGBA"), dtype=np.uint8)
                            for i in range(count)
                        ])
                        a = torch.from_numpy(a_np).to(device=device)
                        b = torch.from_numpy(b_np).to(device=device)
                        changed = torch.any(a != b, dim=3)
                        for row in range(count):
                            coords = torch.nonzero(changed[row], as_tuple=False)
                            if coords.numel() == 0:
                                result.append((None, False))
                                continue
                            y0 = int(coords[:, 0].min().item())
                            y1 = int(coords[:, 0].max().item()) + 1
                            x0 = int(coords[:, 1].min().item())
                            x1 = int(coords[:, 1].max().item()) + 1
                            alpha = b[row, :, :, 3][changed[row]]
                            blend_safe = bool(torch.all(alpha == 255).item())
                            result.append(((x0, y0, x1, y1), blend_safe))
                        del changed, b, a
                        torch.cuda.empty_cache()
                    return result
                except Exception:
                    torch.cuda.empty_cache()
    except Exception:
        pass

    pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    workers = max(1, min(len(pairs), int(os.environ.get("WEBP_ANALYSIS_THREADS", os.cpu_count() or 1))))
    if workers <= 1:
        return [_changed_info_cpu(*pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="webp-delta") as pool:
        return list(pool.map(lambda pair: _changed_info_cpu(*pair), pairs))


def _align_bbox_even(bbox, width: int, height: int):
    left, top, right, bottom = bbox
    # ANMF stores x/y in units of two pixels. Expanding left/top by one pixel is
    # lossless and keeps the frame location representable.
    if left & 1:
        left -= 1
    if top & 1:
        top -= 1
    left = max(0, left)
    top = max(0, top)
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return left, top, right, bottom


def _normalise_durations(count: int, durations) -> list[int]:
    if isinstance(durations, (int, float)):
        return [max(1, int(round(durations)))] * count
    values = [max(1, int(round(value))) for value in durations]
    if len(values) != count:
        raise ValueError("Duration list must match the frame count")
    return values


def _prepare_delta_frames(frames: Sequence[Image.Image], durations: list[int]) -> list[DeltaFrame]:
    if not frames:
        raise ValueError("No frames to encode")
    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("All WebP frames must share one canvas")

    infos = _changed_infos(frames)
    prepared = [DeltaFrame(frames[0].convert("RGBA").copy(), 0, 0, durations[0], False)]

    for index in range(1, len(frames)):
        bbox, blend_safe = infos[index - 1]
        if bbox is None:
            prepared[-1].duration += durations[index]
            continue

        left, top, right, bottom = _align_bbox_even(bbox, width, height)
        target = frames[index].convert("RGBA")
        crop = target.crop((left, top, right, bottom))

        if blend_safe:
            previous = frames[index - 1].convert("RGBA").crop((left, top, right, bottom))
            a = np.asarray(previous, dtype=np.uint8)
            b = np.asarray(crop, dtype=np.uint8).copy()
            unchanged = np.all(a == b, axis=2)
            b[unchanged] = (0, 0, 0, 0)
            crop = Image.fromarray(b, "RGBA")

        prepared.append(
            DeltaFrame(
                image=crop,
                x=left,
                y=top,
                duration=durations[index],
                blend=blend_safe,
            )
        )
    return prepared


def _mux_animation(frames: Sequence[DeltaFrame], canvas: tuple[int, int], loop: int, quality: int) -> bytes:
    width, height = canvas
    workers_default = max(1, min(len(frames), os.cpu_count() or 1))
    workers = max(1, int(os.environ.get("WEBP_ENCODE_THREADS", workers_default)))

    if workers <= 1 or len(frames) <= 1:
        payloads = [_still_payload(frame.image, quality) for frame in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="webp-encode") as pool:
            payloads = list(pool.map(lambda frame: _still_payload(frame.image, quality), frames))

    # VP8X: alpha + animation. Canvas dimensions are stored minus one as u24.
    vp8x = bytes((0x12, 0, 0, 0)) + _u24(width - 1) + _u24(height - 1)
    anim = b"\x00\x00\x00\x00" + struct.pack("<H", max(0, min(0xFFFF, int(loop))))
    body = _chunk(b"VP8X", vp8x) + _chunk(b"ANIM", anim)

    for frame, image_payload in zip(frames, payloads):
        flags = 0x00 if frame.blend else 0x02  # bit 1: 1 means do not blend
        header = (
            _u24(frame.x // 2)
            + _u24(frame.y // 2)
            + _u24(frame.image.width - 1)
            + _u24(frame.image.height - 1)
            + _u24(frame.duration)
            + bytes((flags,))
        )
        body += _chunk(b"ANMF", header + image_payload)

    riff_payload = b"WEBP" + body
    return b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload


def _save_pillow(
    frames: Sequence[Image.Image],
    output: Path,
    durations: list[int],
    *,
    loop: int,
    lossless: bool,
    quality: int,
):
    frames = [frame.convert("RGBA") for frame in frames]
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=list(frames[1:]),
        duration=durations if len(frames) > 1 else durations[0],
        loop=loop,
        lossless=lossless,
        quality=quality,
        method=6,
        minimize_size=True,
        exact=bool(lossless),
    )


def save_webp_fast(
    frames: Sequence[Image.Image],
    output: Path,
    *,
    durations,
    loop: int = 0,
    lossless: bool = True,
    quality: int = 90,
) -> None:
    """Encode an animated WebP with parallel lossless delta-frame preparation.

    Lossless mode constructs standards-compliant ANMF rectangles directly. Opaque
    changes use alpha blending so unchanged pixels inside the rectangle can inherit
    the previous canvas. Translucent/clearing changes use exact no-blend rectangles.
    If the specialised path ever fails, callers still get the established Pillow
    encoder rather than a failed render.
    """
    if not frames:
        raise ValueError("No frames to save")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalised = _normalise_durations(len(frames), durations)

    if not lossless or len(frames) == 1:
        _save_pillow(frames, output, normalised, loop=loop, lossless=lossless, quality=quality)
        return

    try:
        prepared = _prepare_delta_frames(frames, normalised)
        encoded = _mux_animation(prepared, frames[0].size, loop, quality)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".webp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        try:
            # A cheap structural validation catches mux bugs without decoding every
            # frame on every render. Full pixel-equivalence is covered by tests.
            with Image.open(temporary) as check:
                if check.format != "WEBP" or getattr(check, "n_frames", 1) != len(prepared):
                    raise ValueError("Delta WebP validation failed")
            os.replace(temporary, output)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        _save_pillow(frames, output, normalised, loop=loop, lossless=lossless, quality=quality)
