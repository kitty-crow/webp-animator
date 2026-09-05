#!/usr/bin/env python3
"""
anim_align_webp.py

Align a sequence of raster animation frames by translation, then encode them
as an animated WebP.

The registration step compares each frame to the previous frame and searches
for the X/Y translation that maximises pixel similarity.

Key properties:
- No resizing.
- No source pixels are cropped by default.
- The output canvas expands to the union of all translated source frames.
- Alignment can be horizontal only, vertical only, both axes, or disabled.
- Transparent pixels can be ignored for foreground-oriented sprite alignment.
- Uses only Pillow + NumPy.

Example:
    python anim_align_webp.py frame_*.png -o walk.webp --axis xy --max-shift 48

Horizontal only:
    python anim_align_webp.py frame_*.png -o walk.webp --axis x --max-shift 80

Vertical only:
    python anim_align_webp.py frame_*.png -o walk.webp --axis y --max-shift 40
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, features


@dataclass(frozen=True)
class ShiftResult:
    dx: int
    dy: int
    score: float


def load_rgba(path: Path) -> Image.Image:
    """Load one raster file as a single RGBA frame."""
    with Image.open(path) as im:
        return im.convert("RGBA").copy()


def _active_pixel_count(arr: np.ndarray, alpha_threshold: int) -> int:
    """
    Number of pixels considered visually meaningful.

    For opaque images this is simply width * height.
    For images with transparency, fully/mostly transparent pixels do not
    dominate the registration score.
    """
    alpha = arr[..., 3]
    active = int(np.count_nonzero(alpha > alpha_threshold))
    return active if active > 0 else arr.shape[0] * arr.shape[1]


def _overlap_slices(
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
    dx: int,
    dy: int,
):
    """
    A lives at world origin (0, 0).
    B lives at world position (dx, dy).

    Positive dx moves B right. Positive dy moves B down.
    """
    ha, wa = a_shape
    hb, wb = b_shape

    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(wa, dx + wb)
    y1 = min(ha, dy + hb)

    if x1 <= x0 or y1 <= y0:
        return None

    a_ys = slice(y0, y1)
    a_xs = slice(x0, x1)
    b_ys = slice(y0 - dy, y1 - dy)
    b_xs = slice(x0 - dx, x1 - dx)
    return a_ys, a_xs, b_ys, b_xs


def translation_score(
    a: np.ndarray,
    b: np.ndarray,
    dx: int,
    dy: int,
    *,
    sigma: float,
    alpha_threshold: int,
    normaliser: int | None = None,
) -> float:
    """
    Score B placed at (dx, dy) relative to A.

    Each overlapping pixel receives a soft similarity score:

        exp(-(colour_distance / sigma)^2)

    The total is divided by the amount of meaningful source content, rather
    than by overlap size. That prevents tiny overlaps from receiving an
    artificially excellent score.

    RGB and alpha are both considered. Pixels where both frames are effectively
    transparent are ignored.
    """
    sl = _overlap_slices(
        (a.shape[0], a.shape[1]),
        (b.shape[0], b.shape[1]),
        dx,
        dy,
    )
    if sl is None:
        return -math.inf

    a_ys, a_xs, b_ys, b_xs = sl
    aa = a[a_ys, a_xs].astype(np.float32, copy=False)
    bb = b[b_ys, b_xs].astype(np.float32, copy=False)

    active = (aa[..., 3] > alpha_threshold) | (bb[..., 3] > alpha_threshold)
    if not np.any(active):
        return 0.0

    # Mean absolute RGBA difference per pixel, in the 0..255 range.
    dist = np.mean(np.abs(aa - bb), axis=2)

    # Smoothly rewards exact and near-equal pixel values.
    similarity = np.exp(-np.square(dist / sigma))
    similarity = np.where(active, similarity, 0.0)

    if normaliser is None:
        normaliser = max(
            _active_pixel_count(a, alpha_threshold),
            _active_pixel_count(b, alpha_threshold),
            1,
        )

    return float(similarity.sum(dtype=np.float64) / normaliser)


def _resize_proxy(arr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale for coarse search. Returns (proxy, proxy/original scale)."""
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr, 1.0

    scale = max_side / longest
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    proxy = np.asarray(
        Image.fromarray(arr, "RGBA").resize((nw, nh), Image.Resampling.BILINEAR)
    )
    return proxy, scale


def _axis_ranges(axis: str, max_x: int, max_y: int, step: int):
    if axis == "x":
        return range(-max_x, max_x + 1, step), (0,)
    if axis == "y":
        return (0,), range(-max_y, max_y + 1, step)
    if axis == "none":
        return (0,), (0,)
    return (
        range(-max_x, max_x + 1, step),
        range(-max_y, max_y + 1, step),
    )


def _search_grid(
    a: np.ndarray,
    b: np.ndarray,
    xs: Iterable[int],
    ys: Iterable[int],
    *,
    sigma: float,
    alpha_threshold: int,
) -> ShiftResult:
    normaliser = max(
        _active_pixel_count(a, alpha_threshold),
        _active_pixel_count(b, alpha_threshold),
        1,
    )

    best = ShiftResult(0, 0, -math.inf)

    for dy in ys:
        for dx in xs:
            score = translation_score(
                a,
                b,
                dx,
                dy,
                sigma=sigma,
                alpha_threshold=alpha_threshold,
                normaliser=normaliser,
            )
            if score > best.score:
                best = ShiftResult(dx, dy, score)

    return best


def find_best_translation(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    axis: str,
    max_shift_x: int,
    max_shift_y: int,
    sigma: float,
    alpha_threshold: int,
    proxy_max_side: int,
) -> ShiftResult:
    """
    Coarse-to-fine translational registration.

    A proxy image is used for the global scan, then the winning region is
    refined at full resolution to exact integer-pixel coordinates.
    """
    if axis == "none":
        a = np.asarray(a_img)
        b = np.asarray(b_img)
        return ShiftResult(
            0,
            0,
            translation_score(
                a, b, 0, 0,
                sigma=sigma,
                alpha_threshold=alpha_threshold,
            ),
        )

    a = np.asarray(a_img)
    b = np.asarray(b_img)

    pa, sa = _resize_proxy(a, proxy_max_side)
    pb, sb = _resize_proxy(b, proxy_max_side)
    scale = min(sa, sb)

    # If the frames differ in dimensions and ended up with slightly different
    # resize scales, put both on a common proxy scale.
    if abs(sa - scale) > 1e-9:
        h, w = a.shape[:2]
        pa = np.asarray(
            a_img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.BILINEAR,
            )
        )
    if abs(sb - scale) > 1e-9:
        h, w = b.shape[:2]
        pb = np.asarray(
            b_img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.BILINEAR,
            )
        )

    pmx = max(0, round(max_shift_x * scale))
    pmy = max(0, round(max_shift_y * scale))

    # Keep the global search bounded for speed while still sampling the full
    # requested search range.
    largest = max(pmx if axis in ("x", "xy") else 0,
                  pmy if axis in ("y", "xy") else 0)
    coarse_step = max(1, math.ceil(largest / 40))

    xs, ys = _axis_ranges(axis, pmx, pmy, coarse_step)
    coarse = _search_grid(
        pa, pb, xs, ys,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
    )

    # Refine on proxy one pixel around the coarse winner.
    if coarse_step > 1:
        rx0 = max(-pmx, coarse.dx - coarse_step)
        rx1 = min(pmx, coarse.dx + coarse_step)
        ry0 = max(-pmy, coarse.dy - coarse_step)
        ry1 = min(pmy, coarse.dy + coarse_step)

        if axis == "x":
            xs2, ys2 = range(rx0, rx1 + 1), (0,)
        elif axis == "y":
            xs2, ys2 = (0,), range(ry0, ry1 + 1)
        else:
            xs2, ys2 = range(rx0, rx1 + 1), range(ry0, ry1 + 1)

        coarse = _search_grid(
            pa, pb, xs2, ys2,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
        )

    # Convert proxy displacement to full-resolution pixels.
    guess_x = round(coarse.dx / scale) if scale else 0
    guess_y = round(coarse.dy / scale) if scale else 0

    # Refine the integer answer at full resolution. A ±3 pixel neighbourhood
    # is enough after exact one-pixel proxy refinement.
    radius = max(3, math.ceil(1 / max(scale, 1e-9)))

    if axis == "x":
        xs3 = range(
            max(-max_shift_x, guess_x - radius),
            min(max_shift_x, guess_x + radius) + 1,
        )
        ys3 = (0,)
    elif axis == "y":
        xs3 = (0,)
        ys3 = range(
            max(-max_shift_y, guess_y - radius),
            min(max_shift_y, guess_y + radius) + 1,
        )
    else:
        xs3 = range(
            max(-max_shift_x, guess_x - radius),
            min(max_shift_x, guess_x + radius) + 1,
        )
        ys3 = range(
            max(-max_shift_y, guess_y - radius),
            min(max_shift_y, guess_y + radius) + 1,
        )

    return _search_grid(
        a, b, xs3, ys3,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
    )


def register_sequence(
    images: Sequence[Image.Image],
    *,
    axis: str,
    max_shift_x: int,
    max_shift_y: int,
    sigma: float,
    alpha_threshold: int,
    proxy_max_side: int,
) -> tuple[list[tuple[int, int]], list[ShiftResult]]:
    """
    Register each frame against the immediately preceding frame.

    positions[i] is the cumulative world-space top-left position of frame i.
    """
    if not images:
        return [], []

    positions: list[tuple[int, int]] = [(0, 0)]
    pairwise: list[ShiftResult] = []

    for i in range(1, len(images)):
        result = find_best_translation(
            images[i - 1],
            images[i],
            axis=axis,
            max_shift_x=max_shift_x,
            max_shift_y=max_shift_y,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            proxy_max_side=proxy_max_side,
        )
        pairwise.append(result)

        px, py = positions[-1]
        positions.append((px + result.dx, py + result.dy))

    return positions, pairwise


def render_union_canvas(
    images: Sequence[Image.Image],
    positions: Sequence[tuple[int, int]],
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int]]:
    """
    Render every translated source frame on one common union canvas.

    No source image is resized and no source pixel is cropped.
    """
    if not images:
        raise ValueError("No frames")

    min_x = min(x for x, _ in positions)
    min_y = min(y for _, y in positions)
    max_x = max(x + im.width for im, (x, y) in zip(images, positions))
    max_y = max(y + im.height for im, (x, y) in zip(images, positions))

    width = max_x - min_x
    height = max_y - min_y

    origin = (-min_x, -min_y)
    rendered: list[Image.Image] = []

    for im, (x, y) in zip(images, positions):
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.alpha_composite(im, (x + origin[0], y + origin[1]))
        rendered.append(canvas)

    return rendered, (width, height), origin


def save_webp(
    frames: Sequence[Image.Image],
    output: Path,
    *,
    duration_ms: int,
    loop: int,
    lossless: bool,
    quality: int,
) -> None:
    if not frames:
        raise ValueError("No frames to save")

    output.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=loop,
        lossless=lossless,
        quality=quality,
        method=6,
    )


def save_aligned_pngs(frames: Sequence[Image.Image], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(out_dir / f"aligned_{i:04d}.png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Register raster animation frames by translation and encode an "
            "animated WebP without resizing or cropping source pixels."
        )
    )
    p.add_argument(
        "frames",
        nargs="+",
        type=Path,
        help="Input raster frames in animation order.",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output animated .webp file.",
    )
    p.add_argument(
        "--axis",
        choices=("x", "y", "xy", "none"),
        default="xy",
        help="Allowed registration direction. Default: xy.",
    )
    p.add_argument(
        "--max-shift",
        type=int,
        default=64,
        help="Maximum shift in pixels on each enabled axis. Default: 64.",
    )
    p.add_argument(
        "--max-shift-x",
        type=int,
        default=None,
        help="Override horizontal maximum shift.",
    )
    p.add_argument(
        "--max-shift-y",
        type=int,
        default=None,
        help="Override vertical maximum shift.",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Frame duration in milliseconds. Default: 100.",
    )
    p.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Animation loop count. 0 means forever. Default: 0.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=24.0,
        help=(
            "Colour similarity softness in 0..255 units. Lower values demand "
            "closer pixel matches. Default: 24."
        ),
    )
    p.add_argument(
        "--alpha-threshold",
        type=int,
        default=8,
        help=(
            "Alpha <= this value is treated as transparent background for "
            "registration. Default: 8."
        ),
    )
    p.add_argument(
        "--proxy-max-side",
        type=int,
        default=320,
        help="Maximum side of coarse-search proxy. Default: 320.",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=90,
        help="WebP quality 0..100 when not lossless. Default: 90.",
    )
    p.add_argument(
        "--lossy",
        action="store_true",
        help="Use lossy WebP. Default is lossless.",
    )
    p.add_argument(
        "--save-aligned-dir",
        type=Path,
        default=None,
        help="Optional directory in which to save aligned PNG frames.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame registration report.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if len(args.frames) < 1:
        print("At least one input frame is required.", file=sys.stderr)
        return 2

    missing = [str(p) for p in args.frames if not p.is_file()]
    if missing:
        print("Missing input file(s):", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    if not 0 <= args.quality <= 100:
        print("--quality must be between 0 and 100", file=sys.stderr)
        return 2

    if args.max_shift < 0:
        print("--max-shift must be >= 0", file=sys.stderr)
        return 2

    max_x = args.max_shift if args.max_shift_x is None else args.max_shift_x
    max_y = args.max_shift if args.max_shift_y is None else args.max_shift_y

    if max_x < 0 or max_y < 0:
        print("--max-shift-x/--max-shift-y must be >= 0", file=sys.stderr)
        return 2

    if args.sigma <= 0:
        print("--sigma must be > 0", file=sys.stderr)
        return 2

    if args.duration <= 0:
        print("--duration must be > 0", file=sys.stderr)
        return 2

    if not features.check("webp"):
        print(
            "This Pillow build does not have WebP support.",
            file=sys.stderr,
        )
        return 1

    images = [load_rgba(path) for path in args.frames]

    positions, pairwise = register_sequence(
        images,
        axis=args.axis,
        max_shift_x=max_x,
        max_shift_y=max_y,
        sigma=args.sigma,
        alpha_threshold=args.alpha_threshold,
        proxy_max_side=args.proxy_max_side,
    )

    if not args.quiet:
        print("Pairwise registration:")
        print("  frame 0000: position=(+0,+0)")
        for i, result in enumerate(pairwise, start=1):
            x, y = positions[i]
            print(
                f"  frame {i:04d}: "
                f"pair_shift=({result.dx:+d},{result.dy:+d}) "
                f"score={result.score:.6f} "
                f"position=({x:+d},{y:+d})"
            )

    rendered, canvas_size, origin = render_union_canvas(images, positions)

    if args.save_aligned_dir:
        save_aligned_pngs(rendered, args.save_aligned_dir)

    save_webp(
        rendered,
        args.output,
        duration_ms=args.duration,
        loop=args.loop,
        lossless=not args.lossy,
        quality=args.quality,
    )

    if not args.quiet:
        print(f"Canvas: {canvas_size[0]} x {canvas_size[1]}")
        print(f"World origin offset in output: ({origin[0]}, {origin[1]})")
        print(f"Wrote: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
