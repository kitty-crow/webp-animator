#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from engine_backends import run_worker


def _numeric_pngs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


def _write_frames(frames, directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(directory / f"{index:06d}.png")


def process_job(legacy, job_id: str, paths: list[Path], settings: dict) -> None:
    job = legacy.get_job(job_id)
    if not job:
        return
    job_dir = Path(job["job_dir"])

    try:
        legacy.set_job(job_id, progress=12, message="Loading frames")
        images = [legacy.load_rgba(path) for path in paths]
        geometry_mode = settings.get("geometry_mode", "sequential")

        if geometry_mode == "none":
            legacy.set_job(job_id, progress=30, message="Keeping source geometry unchanged")
            positions = [(0, 0) for _ in images]
        elif geometry_mode == "fit_previous":
            legacy.set_job(job_id, progress=18, message="Fitting oversized frames")
            images, scales = legacy.shrink_larger_frames_to_previous(images)
            resized_count = sum(1 for scale in scales if scale < 1.0)
            if resized_count:
                legacy.set_job(job_id, progress=22, message=f"Fitted {resized_count} oversized frame{'s' if resized_count != 1 else ''}")
            positions, _ = legacy.register_sequence(
                images,
                axis=settings["axis"],
                max_shift_x=settings["max_shift"],
                max_shift_y=settings["max_shift"],
                sigma=settings["sigma"],
                alpha_threshold=settings["alpha_threshold"],
                proxy_max_side=320,
            )
        elif geometry_mode == "fix_first":
            def fix_progress(current, total):
                fraction = current / max(1, total)
                legacy.set_job(job_id, progress=18 + round(fraction * 20), message=f"Matching frame {current + 1}/{total + 1} to frame 1")

            images, positions, _ = legacy.fix_frames_to_first(
                images,
                axis=settings["axis"],
                max_shift=settings["max_shift"],
                sigma=settings["sigma"],
                alpha_threshold=settings["alpha_threshold"],
                progress_callback=fix_progress,
            )
        else:
            legacy.set_job(job_id, progress=26, message="Registering frame positions")
            positions, _ = legacy.register_sequence(
                images,
                axis=settings["axis"],
                max_shift_x=settings["max_shift"],
                max_shift_y=settings["max_shift"],
                sigma=settings["sigma"],
                alpha_threshold=settings["alpha_threshold"],
                proxy_max_side=320,
            )

        legacy.set_job(job_id, progress=40, message="Rendering common canvas")
        frames, _, _ = legacy.render_union_canvas(images, positions)
        duration = settings["duration"]
        aligned_dir = job_dir / "aligned"
        _write_frames(frames, aligned_dir)
        current_dir = aligned_dir

        generator = settings.get("frame_generator", "none")
        if generator in {"eden", "speed"} and len(frames) > 1:
            generated_dir = job_dir / "generated"
            shutil.rmtree(generated_dir, ignore_errors=True)
            legacy.set_job(job_id, progress=44, message=f"Starting {generator.upper()} frame generation")

            def generator_progress(current, total):
                fraction = current / max(1, total)
                legacy.set_job(job_id, progress=44 + round(fraction * 20), message=f"{generator.upper()} midpoint generation {current}/{total}")

            run_worker(
                name=generator,
                worker=f"{generator}_worker.py",
                input_dir=current_dir,
                output_dir=generated_dir,
                progress=generator_progress,
            )
            if not _numeric_pngs(generated_dir):
                raise RuntimeError(f"{generator.upper()} produced no output frames.")
            current_dir = generated_dir
            duration = max(1, round(duration / 2))

        multiplier = int(settings.get("rife_multiplier", 1))
        raw_interpolator = settings.get("interpolator")
        if raw_interpolator in {"none", "rife", "amt"}:
            interpolator = raw_interpolator
        else:
            # Persisted jobs created before the selector split only have the old
            # RIFE multiplier. Preserve their restart/resume behaviour exactly.
            interpolator = "rife" if multiplier > 1 else "none"

        if interpolator in {"rife", "amt"} and multiplier > 1:
            interpolated_dir = job_dir / "interpolated"
            shutil.rmtree(interpolated_dir, ignore_errors=True)
            legacy.set_job(job_id, progress=66, message=f"Starting {interpolator.upper()} interpolation")
            if interpolator == "rife":
                legacy.run_rife(job_id, current_dir, interpolated_dir, multiplier)
            else:
                def interpolation_progress(current, total):
                    fraction = current / max(1, total)
                    legacy.set_job(job_id, progress=66 + round(fraction * 20), message=f"AMT interpolation {current}/{total}")

                run_worker(
                    name="amt",
                    worker="amt_worker.py",
                    input_dir=current_dir,
                    output_dir=interpolated_dir,
                    extra_args=["--multi", str(multiplier)],
                    progress=interpolation_progress,
                )
            if not _numeric_pngs(interpolated_dir):
                raise RuntimeError(f"{interpolator.upper()} produced no output frames.")
            current_dir = interpolated_dir
            duration = max(1, round(duration / multiplier))

        if current_dir == aligned_dir:
            final_frames = frames
        else:
            final_frames = [legacy.load_rgba(path) for path in _numeric_pngs(current_dir)]

        legacy.set_job(job_id, progress=90, message="Encoding animated WebP")
        output_path = job_dir / "animation.webp"
        legacy.save_webp(
            final_frames,
            output_path,
            duration_ms=duration,
            loop=0,
            lossless=not settings["lossy"],
            quality=settings["quality"],
        )
        legacy.set_job(
            job_id,
            progress=100,
            status="done",
            message=f"Ready: {len(final_frames)} frames at {duration} ms/frame",
            output_path=str(output_path),
        )
    except Exception as exc:
        legacy.set_job(job_id, status="error", message=str(exc), error=str(exc))
