from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from engine_paths import amt_paths, eden_paths, speed_paths
from smart_frames import gap_scores, non_adjacent_removals, reduction_scores
from webp_fast import save_webp_fast

ROOT = Path(__file__).resolve().parent


@dataclass
class FrameRecord:
    image: Image.Image
    duration: float
    source_indices: set[int] = field(default_factory=set)
    generated: bool = False
    engine: str | None = None
    between_source_frames: tuple[int, int] | None = None
    recursive_depth: int = 0

    def clone(self):
        return FrameRecord(
            image=self.image,
            duration=self.duration,
            source_indices=set(self.source_indices),
            generated=self.generated,
            engine=self.engine,
            between_source_frames=self.between_source_frames,
            recursive_depth=self.recursive_depth,
        )


def parse_target_gaps(value) -> set[int]:
    """Parse 1-based left-frame gap indexes into zero-based source indexes."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")
    result = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if "-" in text:
            text = text.split("-", 1)[0].strip()
        try:
            index = int(text) - 1
        except ValueError:
            continue
        if index >= 0:
            result.add(index)
    return result


def _normalise_geometry(legacy, images: list[Image.Image], settings: dict, progress=None):
    geometry_mode = settings.get("geometry_mode", "sequential")
    axis = settings.get("axis", "xy")
    max_shift = int(settings.get("max_shift", 64))
    sigma = float(settings.get("sigma", 24.0))
    alpha_threshold = int(settings.get("alpha_threshold", 8))

    if geometry_mode == "none":
        if progress:
            progress(26, "Frame geometry disabled")
        positions = [(0, 0)] * len(images)
    elif geometry_mode == "fit_previous":
        if progress:
            progress(18, "Fitting oversized frames")
        images, _ = legacy.shrink_larger_frames_to_previous(images)
        if progress:
            progress(26, "Matching frame positions")
        positions, _ = legacy.register_sequence(
            images,
            axis=axis,
            max_shift_x=max_shift,
            max_shift_y=max_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            proxy_max_side=320,
        )
    elif geometry_mode == "fix_first":
        if progress:
            progress(18, "Optimising scale and pan against frame 1")

        def fix_progress(current, total):
            if progress:
                fraction = current / max(1, total)
                progress(18 + round(fraction * 20), f"Matching frame {current + 1}/{total + 1} to frame 1")

        images, positions, _ = legacy.fix_frames_to_first(
            images,
            axis=axis,
            max_shift=max_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            progress_callback=fix_progress,
        )
    else:
        if progress:
            progress(26, "Matching frame positions")
        positions, _ = legacy.register_sequence(
            images,
            axis=axis,
            max_shift_x=max_shift,
            max_shift_y=max_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            proxy_max_side=320,
        )

    if progress:
        progress(40, "Rendering common animation canvas")
    frames, _, _ = legacy.render_union_canvas(images, positions)
    return frames


def _protected_sources(target_gaps: set[int]) -> set[int]:
    protected = set()
    for left in target_gaps:
        protected.add(left)
        protected.add(left + 1)
    return protected


def _smart_reduce(
    records: list[FrameRecord],
    *,
    threshold: float,
    alpha_threshold: int,
    protected_sources: set[int],
    max_rounds: int = 64,
):
    removed_sources: set[int] = set()
    rounds = 0
    while len(records) >= 3 and rounds < max_rounds:
        rounds += 1
        scores = reduction_scores([record.image for record in records], alpha_threshold=alpha_threshold)
        protected_indexes = {
            index
            for index, record in enumerate(records)
            if record.source_indices & protected_sources
        }
        remove = non_adjacent_removals(scores, threshold, protected_indexes=protected_indexes)
        if not remove:
            break
        for index in reversed(remove):
            record = records[index]
            removed_sources.update(record.source_indices)
            records[index - 1].duration += record.duration
            records[index - 1].source_indices.update(record.source_indices)
            del records[index]
    return sorted(removed_sources), rounds


def _source_gap_record_index(records: Sequence[FrameRecord], left_source: int):
    for index in range(len(records) - 1):
        if left_source in records[index].source_indices and left_source + 1 in records[index + 1].source_indices:
            return index
    return None


def _eligible_gap_indexes(records: Sequence[FrameRecord], target_gaps: set[int]) -> list[int]:
    if not target_gaps:
        return list(range(max(0, len(records) - 1)))
    result = []
    for source_gap in sorted(target_gaps):
        index = _source_gap_record_index(records, source_gap)
        if index is not None:
            result.append(index)
    return sorted(set(result))


def _smart_filter_gaps(
    records: Sequence[FrameRecord],
    indexes: Sequence[int],
    *,
    enabled: bool,
    threshold: float,
    alpha_threshold: int,
):
    indexes = list(indexes)
    if not enabled or not indexes:
        return indexes, {}
    scores = gap_scores([record.image for record in records], alpha_threshold=alpha_threshold)
    score_map = {index: scores[index] for index in indexes if index < len(scores)}
    return [index for index in indexes if score_map.get(index, 0.0) > threshold], score_map


def _save_records(records: Sequence[FrameRecord], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, record in enumerate(records):
        path = directory / f"{index:06d}.png"
        record.image.save(path)
        paths.append(path)
    return paths


def _run_process(command, *, cwd: Path, progress_callback=None, label: str):
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    tail = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if line.startswith("PROGRESS "):
            try:
                _, current, total = line.split()
                if progress_callback:
                    progress_callback(int(current), max(1, int(total)))
            except Exception:
                pass
        elif line:
            tail.append(line)
            tail = tail[-30:]
    code = process.wait()
    if code != 0:
        details = "\n".join(tail[-12:]) or "No worker output."
        raise RuntimeError(f"{label} failed:\n{details}")


def _run_generator(legacy, engine: str, tasks: list[dict], stage_dir: Path, progress=None):
    if not tasks:
        return {}
    worker_dir = stage_dir / f"{engine}-generator"
    output_dir = worker_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = worker_dir / "manifest.json"
    result_path = worker_dir / "result.json"
    manifest_path.write_text(json.dumps({"output_dir": str(output_dir), "tasks": tasks, "seed": 0}, indent=2), encoding="utf-8")

    if engine == "eden":
        ready, python, source, config, checkpoint = eden_paths()
        if not ready:
            raise RuntimeError("EDEN is selected but is not installed. Run `python setup_eden.py`.")
        command = [python, ROOT / "eden_worker.py", "--eden-dir", source, "--config", config, "--checkpoint", checkpoint, "--manifest", manifest_path, "--result", result_path]
    elif engine == "speed":
        ready, python, source, config, checkpoint = speed_paths()
        if not ready:
            raise RuntimeError("SPEED is selected but is not installed. Run `python setup_speed.py`.")
        command = [python, ROOT / "speed_worker.py", "--speed-dir", source, "--config", config, "--checkpoint", checkpoint, "--manifest", manifest_path, "--result", result_path]
    else:
        raise ValueError(f"Unknown frame generator: {engine}")

    _run_process(
        command,
        cwd=source,
        label=engine.upper(),
        progress_callback=(lambda current, total: progress(current / total)) if progress else None,
    )
    value = json.loads(result_path.read_text(encoding="utf-8"))
    return {str(task["id"]): task["frame"] for task in value.get("tasks", [])}


def _run_interpolator(legacy, engine: str, tasks: list[dict], stage_dir: Path, progress=None):
    if not tasks:
        return {}
    worker_dir = stage_dir / f"{engine}-interpolator"
    output_dir = worker_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = worker_dir / "manifest.json"
    result_path = worker_dir / "result.json"
    manifest_path.write_text(json.dumps({"output_dir": str(output_dir), "tasks": tasks}, indent=2), encoding="utf-8")

    if engine == "rife":
        ready, python, source, model = legacy.rife_paths()
        if not ready:
            raise RuntimeError("RIFE is selected but is not installed. Run `python setup_rife.py`.")
        command = [python, ROOT / "rife_selective_worker.py", "--rife-dir", source, "--model-dir", model, "--manifest", manifest_path, "--result", result_path]
    elif engine == "amt":
        ready, python, source, config, checkpoint = amt_paths()
        if not ready:
            raise RuntimeError("AMT is selected but is not installed. Run `python setup_amt.py`.")
        command = [python, ROOT / "amt_worker.py", "--amt-dir", source, "--config", config, "--checkpoint", checkpoint, "--manifest", manifest_path, "--result", result_path]
    else:
        raise ValueError(f"Unknown interpolator: {engine}")

    _run_process(
        command,
        cwd=source,
        label=engine.upper(),
        progress_callback=(lambda current, total: progress(current / total)) if progress else None,
    )
    value = json.loads(result_path.read_text(encoding="utf-8"))
    return {str(task["id"]): list(task.get("frames", [])) for task in value.get("tasks", [])}


def _integer_durations(records: Sequence[FrameRecord]) -> list[int]:
    values = [max(1.0, float(record.duration)) for record in records]
    target = max(len(values), int(round(sum(values))))
    floors = [max(1, int(math.floor(value))) for value in values]
    delta = target - sum(floors)
    if delta > 0:
        order = sorted(range(len(values)), key=lambda i: values[i] - math.floor(values[i]), reverse=True)
        for offset in range(delta):
            floors[order[offset % len(order)]] += 1
    elif delta < 0:
        order = sorted(range(len(values)), key=lambda i: values[i] - math.floor(values[i]))
        remaining = -delta
        for index in order:
            take = min(remaining, max(0, floors[index] - 1))
            floors[index] -= take
            remaining -= take
            if remaining <= 0:
                break
    return floors


def _insert_generator_results(
    records: list[FrameRecord],
    root_gaps: list[tuple[int, int]],
    results: dict[str, str],
    engine: str,
):
    inserted_groups = []
    for ordinal, (record_index, source_gap) in reversed(list(enumerate(root_gaps))):
        task_id = f"gen-{ordinal}"
        path = results.get(task_id)
        if not path:
            continue
        with Image.open(path) as image:
            middle = image.convert("RGBA").copy()
        left = records[record_index]
        old_duration = left.duration
        left.duration = old_duration / 2.0
        new_record = FrameRecord(
            image=middle,
            duration=old_duration / 2.0,
            source_indices=set(),
            generated=True,
            engine=engine,
            between_source_frames=(source_gap, source_gap + 1),
            recursive_depth=1,
        )
        records.insert(record_index + 1, new_record)
        inserted_groups.append((source_gap, source_gap + 1))
    return inserted_groups


def _segments_for_source_gap(records: Sequence[FrameRecord], left_source: int):
    left_index = None
    right_index = None
    for index, record in enumerate(records):
        if left_source in record.source_indices:
            left_index = index
        if left_source + 1 in record.source_indices:
            right_index = index
            break
    if left_index is None or right_index is None or right_index <= left_index:
        return []
    return list(range(left_index, right_index))


def _insert_interpolator_results(
    records: list[FrameRecord],
    task_specs: list[tuple[str, int, int, int]],
    results: dict[str, list[str]],
    engine: str,
):
    for task_id, record_index, left_source, depth in reversed(task_specs):
        paths = results.get(task_id, [])
        if not paths:
            continue
        left_record = records[record_index]
        old_duration = left_record.duration
        piece_count = len(paths) + 1
        left_record.duration = old_duration / piece_count
        inserted = []
        for path in paths:
            with Image.open(path) as image:
                middle = image.convert("RGBA").copy()
            inserted.append(
                FrameRecord(
                    image=middle,
                    duration=old_duration / piece_count,
                    source_indices=set(),
                    generated=True,
                    engine=engine,
                    between_source_frames=(left_source, left_source + 1),
                    recursive_depth=depth,
                )
            )
        records[record_index + 1:record_index + 1] = inserted


def process_job(legacy, job_id: str, paths: list[Path], settings: dict):
    job = legacy.get_job(job_id)
    if not job:
        return
    job_dir = Path(job["job_dir"])

    def progress(value, message):
        legacy.set_job(job_id, progress=int(value), message=message)

    try:
        progress(12, "Loading frames")
        images = [legacy.load_rgba(path) for path in paths]
        frames = _normalise_geometry(legacy, images, settings, progress)
        duration = float(settings.get("duration", 100))
        records = [
            FrameRecord(image=frame, duration=duration, source_indices={index})
            for index, frame in enumerate(frames)
        ]

        target_gaps = parse_target_gaps(settings.get("target_gaps", ""))
        protected_sources = _protected_sources(target_gaps)
        removed_sources = []
        if settings.get("smart_reduction", False):
            progress(42, "Smart frame reduction analysis")
            removed_sources, _ = _smart_reduce(
                records,
                threshold=float(settings.get("reduction_threshold", 2.0)),
                alpha_threshold=int(settings.get("alpha_threshold", 8)),
                protected_sources=protected_sources,
            )

        stage_dir = job_dir / "advanced"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        aligned_paths = _save_records(records, stage_dir / "normalised")

        eligible = _eligible_gap_indexes(records, target_gaps)
        eligible, gap_map = _smart_filter_gaps(
            records,
            eligible,
            enabled=bool(settings.get("smart_missing", False)),
            threshold=float(settings.get("missing_threshold", 12.0)),
            alpha_threshold=int(settings.get("alpha_threshold", 8)),
        )

        # Record source-gap identity before any insertion changes list indexes.
        root_gaps = []
        for record_index in eligible:
            left_sources = records[record_index].source_indices
            right_sources = records[record_index + 1].source_indices
            if target_gaps:
                source_gap = next((gap for gap in target_gaps if gap in left_sources and gap + 1 in right_sources), None)
            else:
                source_gap = max(left_sources) if left_sources else record_index
            if source_gap is None:
                continue
            root_gaps.append((record_index, int(source_gap)))

        generator = str(settings.get("frame_generator", "none")).lower()
        interpolator = str(settings.get("interpolator", "none")).lower()
        multiplier = int(settings.get("rife_multiplier", settings.get("interpolation_multiplier", 1)) or 1)
        if multiplier not in {1, 2, 4, 8}:
            multiplier = 1
        max_depth = int(round(math.log2(multiplier))) if multiplier > 1 else 0

        generated_root_sources = set()
        if generator in {"eden", "speed"} and root_gaps:
            current_paths = _save_records(records, stage_dir / "generator-input")
            tasks = []
            for ordinal, (record_index, source_gap) in enumerate(root_gaps):
                tasks.append({
                    "id": f"gen-{ordinal}",
                    "left": str(current_paths[record_index]),
                    "right": str(current_paths[record_index + 1]),
                })
            progress(48, f"Starting {generator.upper()} frame generation")
            results = _run_generator(
                legacy,
                generator,
                tasks,
                stage_dir,
                progress=lambda fraction: progress(48 + round(fraction * 17), f"{generator.upper()} frame generation"),
            )
            groups = _insert_generator_results(records, root_gaps, results, generator)
            generated_root_sources = {left for left, _ in groups}

        remaining_depth = max_depth
        if generator in {"eden", "speed"} and generated_root_sources:
            remaining_depth = max(0, max_depth - 1)

        if interpolator in {"rife", "amt"} and remaining_depth > 0 and root_gaps:
            current_paths = _save_records(records, stage_dir / "interpolator-input")
            tasks = []
            task_specs = []
            ordinal = 0
            for _, source_gap in root_gaps:
                segment_indexes = _segments_for_source_gap(records, source_gap)
                depth = remaining_depth if source_gap in generated_root_sources else max_depth
                if depth <= 0:
                    continue
                for record_index in segment_indexes:
                    task_id = f"interp-{ordinal}"
                    ordinal += 1
                    task = {
                        "id": task_id,
                        "left": str(current_paths[record_index]),
                        "right": str(current_paths[record_index + 1]),
                        "depth": depth,
                        "alpha_threshold": int(settings.get("alpha_threshold", 8)),
                    }
                    if settings.get("smart_missing", False):
                        task["threshold"] = float(settings.get("missing_threshold", 12.0))
                    tasks.append(task)
                    task_specs.append((task_id, record_index, source_gap, depth))
            if tasks:
                progress(66, f"Starting {interpolator.upper()} interpolation")
                results = _run_interpolator(
                    legacy,
                    interpolator,
                    tasks,
                    stage_dir,
                    progress=lambda fraction: progress(66 + round(fraction * 19), f"{interpolator.upper()} interpolation"),
                )
                _insert_interpolator_results(records, task_specs, results, interpolator)

        progress(88, "Preparing optimised WebP deltas")
        output_path = job_dir / "animation.webp"
        durations = _integer_durations(records)
        save_webp_fast(
            [record.image for record in records],
            output_path,
            durations=durations,
            loop=0,
            lossless=not bool(settings.get("lossy", False)),
            quality=int(settings.get("quality", 90)),
        )

        generated_meta = [
            {
                "index": index,
                "engine": record.engine,
                "between_source_frames": record.between_source_frames,
                "recursive_depth": record.recursive_depth,
                "generated": record.generated,
            }
            for index, record in enumerate(records)
            if record.generated
        ]
        legacy.set_job(
            job_id,
            progress=100,
            status="done",
            message=(
                f"Ready: {len(records)} frames; "
                f"smart reduction removed {len(removed_sources)} source frame"
                f"{'s' if len(removed_sources) != 1 else ''}"
            ),
            output_path=str(output_path),
            generated_frames=generated_meta,
            smart_reduction_removed=[index + 1 for index in removed_sources],
            smart_gap_scores={str(index + 1): round(score, 4) for index, score in gap_map.items()},
        )
    except Exception as exc:
        legacy.set_job(job_id, status="error", message=str(exc), error=str(exc))


def analyse_paths(legacy, paths: list[Path], settings: dict) -> dict:
    images = [legacy.load_rgba(path) for path in paths]
    frames = _normalise_geometry(legacy, images, settings, None)
    records = [
        FrameRecord(image=frame, duration=float(settings.get("duration", 100)), source_indices={index})
        for index, frame in enumerate(frames)
    ]
    target_gaps = parse_target_gaps(settings.get("target_gaps", ""))
    removed = []
    if settings.get("smart_reduction", False):
        removed, rounds = _smart_reduce(
            records,
            threshold=float(settings.get("reduction_threshold", 2.0)),
            alpha_threshold=int(settings.get("alpha_threshold", 8)),
            protected_sources=_protected_sources(target_gaps),
        )
    else:
        rounds = 0

    gaps = gap_scores([record.image for record in records], alpha_threshold=int(settings.get("alpha_threshold", 8)))
    missing_threshold = float(settings.get("missing_threshold", 12.0))
    missing = []
    for index, score in enumerate(gaps):
        if score <= missing_threshold:
            continue
        left = sorted(records[index].source_indices)
        right = sorted(records[index + 1].source_indices)
        missing.append({
            "gap_index": index + 1,
            "score": round(score, 4),
            "left_sources": [value + 1 for value in left],
            "right_sources": [value + 1 for value in right],
        })

    return {
        "source_count": len(frames),
        "suggested_count": len(records),
        "reduction_removed": [index + 1 for index in removed],
        "reduction_rounds": rounds,
        "missing_gaps": missing,
        "gap_scores": [round(score, 4) for score in gaps],
    }
