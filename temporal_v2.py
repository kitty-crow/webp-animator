from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

import advanced_pipeline as base
from engine_paths import amt_paths, eden_paths, speed_paths
from smart_frames import gap_scores, non_adjacent_removals, pair_scores, reduction_scores
from webp_fast import save_webp_fast

ROOT = Path(__file__).resolve().parent
AUTO_FILL_MAX_FRAMES = max(1, int(os.environ.get("FRAME_AUTO_FILL_MAX", "31")))


@dataclass
class GapPlan:
    kind: str  # normal | loop
    record_index: int | None
    source_gap: int | None
    manual: bool
    score: float | None = None

    @property
    def key(self) -> str:
        if self.kind == "loop":
            return "loop"
        if self.source_gap is not None:
            return str(self.source_gap + 1)
        return str((self.record_index or 0) + 1)


def parse_target_gap_spec(value) -> tuple[set[int], bool]:
    """Return zero-based ordinary source gaps plus the cyclic loop selection."""
    if value is None:
        return set(), False
    if isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")

    normal: set[int] = set()
    loop = False
    for part in parts:
        text = str(part).strip().lower()
        if not text:
            continue
        if text in {"loop", "last-first", "last_to_first", "lasttofirst"}:
            loop = True
            continue
        if "-" in text:
            text = text.split("-", 1)[0].strip()
        try:
            index = int(text) - 1
        except ValueError:
            continue
        if index >= 0:
            normal.add(index)
    return normal, loop


def _protected_sources(normal: set[int], loop: bool, source_count: int) -> set[int]:
    protected: set[int] = set()
    for left in normal:
        protected.add(left)
        protected.add(left + 1)
    if loop and source_count:
        protected.add(0)
        protected.add(source_count - 1)
    return protected


def _smart_reduce(
    records: list[base.FrameRecord],
    *,
    threshold: float,
    alpha_threshold: int,
    protected_sources: set[int],
    progress: Callable[[int, str], None] | None = None,
    max_rounds: int = 64,
):
    removed_sources: set[int] = set()
    rounds = 0
    while len(records) >= 3 and rounds < max_rounds:
        rounds += 1
        if progress:
            progress(min(68, 45 + rounds), f"Smart reduction pass {rounds}")
        scores = reduction_scores(
            [record.image for record in records],
            alpha_threshold=alpha_threshold,
        )
        protected_indexes = {
            index
            for index, record in enumerate(records)
            if record.source_indices & protected_sources
        }
        remove = non_adjacent_removals(
            scores,
            threshold,
            protected_indexes=protected_indexes,
        )
        if not remove:
            break
        for index in reversed(remove):
            record = records[index]
            removed_sources.update(record.source_indices)
            records[index - 1].duration += record.duration
            records[index - 1].source_indices.update(record.source_indices)
            del records[index]
    return sorted(removed_sources), rounds


def _source_gap_index(records: Sequence[base.FrameRecord], source_gap: int):
    return base._source_gap_record_index(records, source_gap)


def _normal_gap_source(records: Sequence[base.FrameRecord], index: int) -> int:
    left = records[index].source_indices
    right = records[index + 1].source_indices
    if left and right:
        return max(left)
    return index


def _auto_gap_scores(
    records: Sequence[base.FrameRecord],
    *,
    alpha_threshold: int,
    loop_mode: str,
) -> tuple[dict[int, float], float | None]:
    frames = [record.image for record in records]
    if len(frames) < 2:
        return {}, None

    pairs: list[tuple[int, int]] = []
    keys: list[int | str] = []
    if loop_mode != "only":
        for index in range(len(frames) - 1):
            pairs.append((index, index + 1))
            keys.append(index)
    if loop_mode in {"alongside", "only"}:
        pairs.append((len(frames) - 1, 0))
        keys.append("loop")

    scores = pair_scores(frames, pairs, alpha_threshold=alpha_threshold)
    normal: dict[int, float] = {}
    loop_score = None
    for key, score in zip(keys, scores):
        if key == "loop":
            loop_score = float(score)
        else:
            normal[int(key)] = float(score)
    return normal, loop_score


def _plans_for_job(records: Sequence[base.FrameRecord], settings: dict, source_count: int):
    manual_normal, manual_loop = parse_target_gap_spec(settings.get("target_gaps", ""))
    smart_missing = bool(settings.get("smart_missing", False))
    threshold = float(settings.get("missing_threshold", 12.0))
    loop_mode = str(settings.get("loop_analysis", "off")).strip().lower()
    if loop_mode not in {"off", "alongside", "only"}:
        loop_mode = "off"

    plans: dict[tuple[str, int], GapPlan] = {}

    # Manual selections are explicit instructions and are never filtered away by
    # Smart Missing. They are also allowed to coexist with automatic discovery.
    for source_gap in sorted(manual_normal):
        index = _source_gap_index(records, source_gap)
        if index is not None:
            plans[("normal", index)] = GapPlan(
                kind="normal",
                record_index=index,
                source_gap=source_gap,
                manual=True,
            )
    if manual_loop and len(records) >= 2:
        plans[("loop", -1)] = GapPlan(
            kind="loop",
            record_index=None,
            source_gap=max(0, source_count - 1),
            manual=True,
        )

    auto_normal: dict[int, float] = {}
    loop_score = None
    if smart_missing:
        auto_normal, loop_score = _auto_gap_scores(
            records,
            alpha_threshold=int(settings.get("alpha_threshold", 8)),
            loop_mode=loop_mode,
        )
        for index, score in auto_normal.items():
            if score <= threshold:
                continue
            source_gap = _normal_gap_source(records, index)
            key = ("normal", index)
            if key in plans:
                plans[key].score = score
            else:
                plans[key] = GapPlan(
                    kind="normal",
                    record_index=index,
                    source_gap=source_gap,
                    manual=False,
                    score=score,
                )
        if loop_score is not None and loop_score > threshold and len(records) >= 2:
            key = ("loop", -1)
            if key in plans:
                plans[key].score = loop_score
            else:
                plans[key] = GapPlan(
                    kind="loop",
                    record_index=None,
                    source_gap=max(0, source_count - 1),
                    manual=False,
                    score=loop_score,
                )

    # Preserve the established conventional interpolation behaviour when neither
    # Smart Missing nor manual targeting is being used.
    if not smart_missing and not manual_normal and not manual_loop:
        for index in range(max(0, len(records) - 1)):
            plans[("normal", index)] = GapPlan(
                kind="normal",
                record_index=index,
                source_gap=_normal_gap_source(records, index),
                manual=False,
            )

    return sorted(
        plans.values(),
        key=lambda plan: (plan.kind == "loop", plan.record_index if plan.record_index is not None else 10**9),
    ), auto_normal, loop_score


def _task_refs(task_result: dict) -> list[dict]:
    refs = list(task_result.get("frames", []))
    if not refs and task_result.get("frame"):
        refs = [{"path": task_result["frame"], "t": 0.5}]
    normalised = []
    for offset, ref in enumerate(refs):
        if isinstance(ref, dict):
            path = ref.get("path")
            t = ref.get("t", (offset + 1) / (len(refs) + 1))
        else:
            path = ref
            t = (offset + 1) / (len(refs) + 1)
        if not path:
            continue
        normalised.append({"path": str(path), "t": max(0.0, min(1.0, float(t)))})
    normalised.sort(key=lambda item: item["t"])
    return normalised


def _run_generator(engine: str, tasks: list[dict], stage_dir: Path, progress=None):
    if not tasks:
        return {}
    worker_dir = stage_dir / f"{engine}-generator-v2"
    output_dir = worker_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = worker_dir / "manifest.json"
    result_path = worker_dir / "result.json"
    manifest_path.write_text(
        json.dumps({"output_dir": str(output_dir), "tasks": tasks, "seed": 0}, indent=2),
        encoding="utf-8",
    )

    if engine == "eden":
        ready, python, source, config, checkpoint = eden_paths()
        if not ready:
            raise RuntimeError("EDEN is selected but is not installed. Run `python setup_eden.py`.")
        command = [
            python, ROOT / "eden_worker.py", "--eden-dir", source,
            "--config", config, "--checkpoint", checkpoint,
            "--manifest", manifest_path, "--result", result_path,
        ]
    elif engine == "speed":
        ready, python, source, config, checkpoint = speed_paths()
        if not ready:
            raise RuntimeError("SPEED is selected but is not installed. Run `python setup_speed.py`.")
        command = [
            python, ROOT / "speed_worker.py", "--speed-dir", source,
            "--config", config, "--checkpoint", checkpoint,
            "--manifest", manifest_path, "--result", result_path,
        ]
    else:
        raise ValueError(f"Unknown frame generator: {engine}")

    base._run_process(
        command,
        cwd=source,
        label=engine.upper(),
        progress_callback=(lambda current, total: progress(current / total)) if progress else None,
    )
    value = json.loads(result_path.read_text(encoding="utf-8"))
    return {str(item.get("id")): dict(item) for item in value.get("tasks", [])}


def _run_interpolator(legacy, engine: str, tasks: list[dict], stage_dir: Path, progress=None):
    if not tasks:
        return {}
    worker_dir = stage_dir / f"{engine}-interpolator-v2"
    output_dir = worker_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = worker_dir / "manifest.json"
    result_path = worker_dir / "result.json"
    manifest_path.write_text(
        json.dumps({"output_dir": str(output_dir), "tasks": tasks}, indent=2),
        encoding="utf-8",
    )

    if engine == "rife":
        ready, python, source, model = legacy.rife_paths()
        if not ready:
            raise RuntimeError("RIFE is selected but is not installed. Run `python setup_rife.py`.")
        command = [
            python, ROOT / "rife_selective_worker.py", "--rife-dir", source,
            "--model-dir", model, "--manifest", manifest_path, "--result", result_path,
        ]
    elif engine == "amt":
        ready, python, source, config, checkpoint = amt_paths()
        if not ready:
            raise RuntimeError("AMT is selected but is not installed. Run `python setup_amt.py`.")
        command = [
            python, ROOT / "amt_worker.py", "--amt-dir", source,
            "--config", config, "--checkpoint", checkpoint,
            "--manifest", manifest_path, "--result", result_path,
        ]
    else:
        raise ValueError(f"Unknown interpolator: {engine}")

    base._run_process(
        command,
        cwd=source,
        label=engine.upper(),
        progress_callback=(lambda current, total: progress(current / total)) if progress else None,
    )
    value = json.loads(result_path.read_text(encoding="utf-8"))
    return {str(item.get("id")): dict(item) for item in value.get("tasks", [])}


def _load_ref_image(ref: dict) -> Image.Image:
    with Image.open(ref["path"]) as image:
        return image.convert("RGBA").copy()


def _insert_gap_frames(
    records: list[base.FrameRecord],
    plan: GapPlan,
    refs: list[dict],
    *,
    engine_label: str,
    source_count: int,
):
    refs = sorted(refs, key=lambda item: float(item.get("t", 0.5)))
    if not refs:
        return

    if plan.kind == "loop":
        left_record = records[-1]
        old_duration = float(left_record.duration)
        times = [0.0] + [float(ref["t"]) for ref in refs] + [1.0]
        intervals = [old_duration * max(0.0, times[i + 1] - times[i]) for i in range(len(times) - 1)]
        left_record.duration = intervals[0]
        for offset, ref in enumerate(refs):
            records.append(
                base.FrameRecord(
                    image=_load_ref_image(ref),
                    duration=intervals[offset + 1],
                    source_indices=set(),
                    generated=True,
                    engine=engine_label,
                    between_source_frames=(max(0, source_count - 1), 0),
                    recursive_depth=1,
                )
            )
        return

    if plan.record_index is None:
        return
    index = int(plan.record_index)
    left_record = records[index]
    old_duration = float(left_record.duration)
    times = [0.0] + [float(ref["t"]) for ref in refs] + [1.0]
    intervals = [old_duration * max(0.0, times[i + 1] - times[i]) for i in range(len(times) - 1)]
    left_record.duration = intervals[0]
    source_gap = int(plan.source_gap if plan.source_gap is not None else index)
    inserted = []
    for offset, ref in enumerate(refs):
        inserted.append(
            base.FrameRecord(
                image=_load_ref_image(ref),
                duration=intervals[offset + 1],
                source_indices=set(),
                generated=True,
                engine=engine_label,
                between_source_frames=(source_gap, source_gap + 1),
                recursive_depth=1,
            )
        )
    records[index + 1:index + 1] = inserted


def _uniform_count_for_legacy(settings: dict) -> int:
    multiplier = int(settings.get("rife_multiplier", settings.get("interpolation_multiplier", 1)) or 1)
    return max(0, multiplier - 1) if multiplier in {1, 2, 4, 8} else 0


def _requested_fill_count(settings: dict, using_gap_fill: bool) -> int:
    if using_gap_fill:
        try:
            return max(0, int(settings.get("frames_to_fill", 1)))
        except (TypeError, ValueError):
            return 1
    return _uniform_count_for_legacy(settings)


def _prepare_engine_results(
    legacy,
    plans: list[GapPlan],
    records: list[base.FrameRecord],
    stage_dir: Path,
    settings: dict,
    progress: Callable[[int, str], None],
):
    if not plans:
        return {}, []

    source_paths = base._save_records(records, stage_dir / "temporal-input")
    generator = str(settings.get("frame_generator", "none")).lower()
    interpolator = str(settings.get("interpolator", "none")).lower()
    manual_normal, manual_loop = parse_target_gap_spec(settings.get("target_gaps", ""))
    using_gap_fill = bool(settings.get("smart_missing", False) or manual_normal or manual_loop)
    fill_count = _requested_fill_count(settings, using_gap_fill)
    threshold = float(settings.get("missing_threshold", 12.0))
    alpha_threshold = int(settings.get("alpha_threshold", 8))

    endpoint_paths: dict[str, tuple[Path, Path]] = {}
    for ordinal, plan in enumerate(plans):
        key = f"gap-{ordinal}"
        if plan.kind == "loop":
            endpoint_paths[key] = (source_paths[-1], source_paths[0])
        else:
            assert plan.record_index is not None
            endpoint_paths[key] = (
                source_paths[plan.record_index],
                source_paths[plan.record_index + 1],
            )

    # In automatic count mode, a manually selected gap may already satisfy the
    # threshold. Honour 0 = engine decides, which may legitimately mean 0 frames.
    active_keys = set(endpoint_paths)
    if fill_count == 0:
        frames = [record.image for record in records]
        for ordinal, plan in enumerate(plans):
            key = f"gap-{ordinal}"
            if plan.score is not None:
                score = plan.score
            elif plan.kind == "loop":
                score = pair_scores(frames, [(len(frames) - 1, 0)], alpha_threshold=alpha_threshold)[0]
            else:
                assert plan.record_index is not None
                score = pair_scores(frames, [(plan.record_index, plan.record_index + 1)], alpha_threshold=alpha_threshold)[0]
            if score <= threshold:
                active_keys.discard(key)

    final_refs: dict[str, list[dict]] = {key: [] for key in endpoint_paths}
    diagnostics: list[dict] = []
    if not active_keys or (generator not in {"eden", "speed"} and interpolator not in {"rife", "amt"}):
        return final_refs, diagnostics

    # One engine selected: let it own the whole requested count/adaptive fill.
    if generator in {"eden", "speed"} and interpolator not in {"rife", "amt"}:
        tasks = []
        for key in sorted(active_keys):
            left, right = endpoint_paths[key]
            task = {
                "id": key,
                "left": str(left),
                "right": str(right),
                "count": fill_count,
                "threshold": threshold,
                "max_frames": AUTO_FILL_MAX_FRAMES,
                "alpha_threshold": alpha_threshold,
            }
            tasks.append(task)
        progress(52, f"Starting {generator.upper()} gap filling")
        result = _run_generator(
            generator,
            tasks,
            stage_dir,
            progress=lambda fraction: progress(52 + round(fraction * 27), f"{generator.upper()} gap filling"),
        )
        for key, item in result.items():
            final_refs[key] = _task_refs(item)
            if item.get("limit_reached"):
                diagnostics.append({"gap": key, "engine": generator, "limit_reached": True, "max_score": item.get("max_score")})
        return final_refs, diagnostics

    if interpolator in {"rife", "amt"} and generator not in {"eden", "speed"}:
        tasks = []
        for key in sorted(active_keys):
            left, right = endpoint_paths[key]
            tasks.append({
                "id": key,
                "left": str(left),
                "right": str(right),
                "count": fill_count,
                "threshold": threshold,
                "max_frames": AUTO_FILL_MAX_FRAMES,
                "alpha_threshold": alpha_threshold,
            })
        progress(56, f"Starting {interpolator.upper()} gap filling")
        result = _run_interpolator(
            legacy,
            interpolator,
            tasks,
            stage_dir,
            progress=lambda fraction: progress(56 + round(fraction * 29), f"{interpolator.upper()} gap filling"),
        )
        for key, item in result.items():
            final_refs[key] = _task_refs(item)
            if item.get("limit_reached"):
                diagnostics.append({"gap": key, "engine": interpolator, "limit_reached": True, "max_score": item.get("max_score")})
        return final_refs, diagnostics

    # Generator + interpolator: the generator supplies a structural midpoint anchor.
    generator_tasks = []
    for key in sorted(active_keys):
        left, right = endpoint_paths[key]
        generator_tasks.append({
            "id": key,
            "left": str(left),
            "right": str(right),
            "count": 1,
            "alpha_threshold": alpha_threshold,
        })
    progress(48, f"Starting {generator.upper()} structural midpoint generation")
    generated = _run_generator(
        generator,
        generator_tasks,
        stage_dir,
        progress=lambda fraction: progress(48 + round(fraction * 17), f"{generator.upper()} structural midpoint generation"),
    )

    interp_tasks = []
    mapping: dict[str, tuple[str, str, float, float]] = {}
    for key in sorted(active_keys):
        anchor_refs = _task_refs(generated.get(key, {}))
        if not anchor_refs:
            continue
        anchor = min(anchor_refs, key=lambda ref: abs(float(ref["t"]) - 0.5))
        anchor["t"] = 0.5
        left, right = endpoint_paths[key]

        if fill_count > 0:
            remaining = max(0, fill_count - 1)
            left_count = (remaining + 1) // 2
            right_count = remaining - left_count
        else:
            left_count = right_count = 0

        keep_anchor = fill_count == 0 or fill_count % 2 == 1
        if keep_anchor:
            final_refs[key].append(anchor)

        if fill_count > 0 and not keep_anchor:
            # Even fixed counts cannot include the midpoint and remain uniformly
            # cardinal. Use the midpoint as an internal structural anchor but do
            # not include it as one of the final frames.
            left_count = fill_count // 2
            right_count = fill_count // 2

        if left_count > 0 or fill_count == 0:
            task_id = f"{key}-left"
            interp_tasks.append({
                "id": task_id,
                "left": str(left),
                "right": anchor["path"],
                "count": left_count if fill_count > 0 else 0,
                "threshold": threshold,
                "max_frames": max(1, AUTO_FILL_MAX_FRAMES // 2),
                "alpha_threshold": alpha_threshold,
            })
            mapping[task_id] = (key, "left", 0.0, 0.5)
        if right_count > 0 or fill_count == 0:
            task_id = f"{key}-right"
            interp_tasks.append({
                "id": task_id,
                "left": anchor["path"],
                "right": str(right),
                "count": right_count if fill_count > 0 else 0,
                "threshold": threshold,
                "max_frames": max(1, AUTO_FILL_MAX_FRAMES // 2),
                "alpha_threshold": alpha_threshold,
            })
            mapping[task_id] = (key, "right", 0.5, 1.0)

    if interp_tasks:
        progress(66, f"Starting {interpolator.upper()} anchored interpolation")
        interpolated = _run_interpolator(
            legacy,
            interpolator,
            interp_tasks,
            stage_dir,
            progress=lambda fraction: progress(66 + round(fraction * 19), f"{interpolator.upper()} anchored interpolation"),
        )
        for task_id, item in interpolated.items():
            if task_id not in mapping:
                continue
            key, _, start, end = mapping[task_id]
            for ref in _task_refs(item):
                local_t = float(ref["t"])
                final_refs[key].append({
                    "path": ref["path"],
                    "t": start + (end - start) * local_t,
                })
            if item.get("limit_reached"):
                diagnostics.append({"gap": key, "engine": interpolator, "limit_reached": True, "max_score": item.get("max_score")})

    for key in final_refs:
        final_refs[key].sort(key=lambda ref: float(ref["t"]))
        if fill_count > 0 and len(final_refs[key]) > fill_count:
            final_refs[key] = final_refs[key][:fill_count]
    return final_refs, diagnostics


def process_job(legacy, job_id: str, paths: list[Path], settings: dict):
    job = legacy.get_job(job_id)
    if not job:
        return
    job_dir = Path(job["job_dir"])

    def progress(value, message):
        legacy.set_job(job_id, progress=int(value), message=message)

    try:
        progress(10, "Loading frames")
        images = [legacy.load_rgba(path) for path in paths]
        frames = base._normalise_geometry(legacy, images, settings, progress)
        duration = float(settings.get("duration", 100))
        records = [
            base.FrameRecord(image=frame, duration=duration, source_indices={index})
            for index, frame in enumerate(frames)
        ]
        source_count = len(records)

        manual_normal, manual_loop = parse_target_gap_spec(settings.get("target_gaps", ""))
        removed_sources = []
        if settings.get("smart_reduction", False):
            progress(42, "Smart frame reduction analysis")
            removed_sources, _ = _smart_reduce(
                records,
                threshold=float(settings.get("reduction_threshold", 2.0)),
                alpha_threshold=int(settings.get("alpha_threshold", 8)),
                protected_sources=_protected_sources(manual_normal, manual_loop, source_count),
            )

        stage_dir = job_dir / "advanced"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        plans, auto_scores, loop_score = _plans_for_job(records, settings, source_count)
        refs_by_key, diagnostics = _prepare_engine_results(
            legacy,
            plans,
            records,
            stage_dir,
            settings,
            progress,
        )

        # Insert ordinary gaps in reverse list order so their original record indexes
        # remain valid. The cyclic loop bridge is appended only after those inserts.
        ordinary = [(ordinal, plan) for ordinal, plan in enumerate(plans) if plan.kind == "normal"]
        for ordinal, plan in sorted(ordinary, key=lambda item: item[1].record_index or 0, reverse=True):
            _insert_gap_frames(
                records,
                plan,
                refs_by_key.get(f"gap-{ordinal}", []),
                engine_label="+".join(
                    name for name in (
                        str(settings.get("frame_generator", "none")),
                        str(settings.get("interpolator", "none")),
                    ) if name != "none"
                ) or "none",
                source_count=source_count,
            )

        for ordinal, plan in enumerate(plans):
            if plan.kind != "loop":
                continue
            _insert_gap_frames(
                records,
                plan,
                refs_by_key.get(f"gap-{ordinal}", []),
                engine_label="+".join(
                    name for name in (
                        str(settings.get("frame_generator", "none")),
                        str(settings.get("interpolator", "none")),
                    ) if name != "none"
                ) or "none",
                source_count=source_count,
            )

        progress(88, "Preparing optimised WebP deltas")
        output_path = job_dir / "animation.webp"
        durations = base._integer_durations(records)
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
                "loop_bridge": bool(record.generated and record.between_source_frames == (max(0, source_count - 1), 0)),
            }
            for index, record in enumerate(records)
            if record.generated
        ]
        limit_notes = [item for item in diagnostics if item.get("limit_reached")]
        message = (
            f"Ready: {len(records)} frames; smart reduction removed {len(removed_sources)} source frame"
            f"{'s' if len(removed_sources) != 1 else ''}"
        )
        if limit_notes:
            message += f"; {len(limit_notes)} automatic gap fill(s) reached the safety ceiling"

        legacy.set_job(
            job_id,
            progress=100,
            status="done",
            message=message,
            output_path=str(output_path),
            generated_frames=generated_meta,
            smart_reduction_removed=[index + 1 for index in removed_sources],
            smart_gap_scores={str(index + 1): round(score, 4) for index, score in auto_scores.items()},
            loop_gap_score=None if loop_score is None else round(loop_score, 4),
            auto_fill_diagnostics=diagnostics,
        )
    except Exception as exc:
        legacy.set_job(job_id, status="error", message=str(exc), error=str(exc))


def _loop_dwell_diagnostic(records: Sequence[base.FrameRecord], loop_score: float | None, alpha_threshold: int):
    if loop_score is None or len(records) < 3:
        return None
    frames = [record.image for record in records]
    neighbours = pair_scores(
        frames,
        [(len(frames) - 2, len(frames) - 1), (0, 1)],
        alpha_threshold=alpha_threshold,
    )
    typical = sum(neighbours) / len(neighbours)
    duplicate_like = loop_score <= max(0.75, typical * 0.18)
    if not duplicate_like or typical <= 1.0:
        return None
    return {
        "detected": True,
        "loop_score": round(float(loop_score), 4),
        "neighbour_step": round(float(typical), 4),
        "message": "Last and first frames are nearly duplicates while neighbouring motion is larger; consider collapsing the redundant loop hold instead of generating bridge frames.",
    }


def analyse_paths(
    legacy,
    paths: list[Path],
    settings: dict,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    def report(value, message):
        if progress:
            progress(int(value), message)

    report(5, "Loading analysis frames")
    images = [legacy.load_rgba(path) for path in paths]
    report(12, "Normalising frame geometry")
    frames = base._normalise_geometry(legacy, images, settings, report)
    records = [
        base.FrameRecord(
            image=frame,
            duration=float(settings.get("duration", 100)),
            source_indices={index},
        )
        for index, frame in enumerate(frames)
    ]
    source_count = len(records)
    manual_normal, manual_loop = parse_target_gap_spec(settings.get("target_gaps", ""))

    removed = []
    if settings.get("smart_reduction", False):
        report(44, "Running GPU-first smart reduction")
        removed, rounds = _smart_reduce(
            records,
            threshold=float(settings.get("reduction_threshold", 2.0)),
            alpha_threshold=int(settings.get("alpha_threshold", 8)),
            protected_sources=_protected_sources(manual_normal, manual_loop, source_count),
            progress=report,
        )
    else:
        rounds = 0

    smart_missing = bool(settings.get("smart_missing", False))
    loop_mode = str(settings.get("loop_analysis", "off")).strip().lower()
    if loop_mode not in {"off", "alongside", "only"}:
        loop_mode = "off"

    auto_normal: dict[int, float] = {}
    loop_score = None
    missing = []
    if smart_missing:
        report(72, "GPU gap scoring")
        auto_normal, loop_score = _auto_gap_scores(
            records,
            alpha_threshold=int(settings.get("alpha_threshold", 8)),
            loop_mode=loop_mode,
        )
        threshold = float(settings.get("missing_threshold", 12.0))
        for index, score in auto_normal.items():
            if score <= threshold:
                continue
            left = sorted(records[index].source_indices)
            right = sorted(records[index + 1].source_indices)
            source_gap = _normal_gap_source(records, index)
            missing.append({
                "gap_key": str(source_gap + 1),
                "gap_index": source_gap + 1,
                "label": f"Frame {source_gap + 1} ↔ Frame {source_gap + 2}",
                "score": round(score, 4),
                "left_sources": [value + 1 for value in left],
                "right_sources": [value + 1 for value in right],
            })
        if loop_score is not None and loop_score > threshold:
            missing.append({
                "gap_key": "loop",
                "gap_index": None,
                "label": f"Loop: Frame {source_count} ↔ Frame 1",
                "score": round(loop_score, 4),
                "left_sources": [source_count],
                "right_sources": [1],
            })

    report(90, "Checking loop continuity")
    dwell = _loop_dwell_diagnostic(
        records,
        loop_score,
        int(settings.get("alpha_threshold", 8)),
    )
    report(100, "Analysis complete")

    ordered_scores = [
        round(auto_normal.get(index, 0.0), 4)
        for index in range(max(0, len(records) - 1))
    ]
    return {
        "source_count": len(frames),
        "suggested_count": len(records),
        "reduction_removed": [index + 1 for index in removed],
        "reduction_rounds": rounds,
        "missing_gaps": missing,
        "gap_scores": ordered_scores,
        "loop_gap_score": None if loop_score is None else round(loop_score, 4),
        "loop_dwell": dwell,
        "manual_gaps": [str(index + 1) for index in sorted(manual_normal)] + (["loop"] if manual_loop else []),
    }
