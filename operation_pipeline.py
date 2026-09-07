from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

import advanced_pipeline as base
import job_control
import temporal_repair

_ALLOWED = {"gap", "repair", "interpolate"}
_ALIASES = {
    "gap": "gap",
    "gap-fill": "gap",
    "gap_fill": "gap",
    "generation": "gap",
    "generator": "gap",
    "repair": "repair",
    "propainter": "repair",
    "pro-painter": "repair",
    "interpolate": "interpolate",
    "interpolation": "interpolate",
    "interp": "interpolate",
}


@dataclass(frozen=True)
class PairSpec:
    left_index: int
    right_index: int
    wrap: bool
    source_gap: int


def parse_pipeline(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace(";", ",").split(",")
    result: list[str] = []
    for part in parts:
        token = _ALIASES.get(str(part).strip().lower())
        if token in _ALLOWED:
            result.append(token)
    return result[:32]


def default_pipeline(settings: dict) -> list[str]:
    result: list[str] = []
    if str(settings.get("frame_generator", "none")).lower() in {"eden", "speed"}:
        result.append("gap")
    if temporal_repair.repair_requested(settings):
        result.append("repair")
    if str(settings.get("interpolator", "none")).lower() in {"rife", "amt"}:
        result.append("interpolate")
    return result


def pipeline_for_settings(settings: dict) -> list[str]:
    raw = str(settings.get("temporal_pipeline", "") or "").strip()
    return parse_pipeline(raw) if raw else default_pipeline(settings)


def _source_record_index(records: Sequence[base.FrameRecord], source: int, *, after: int = -1) -> int | None:
    for index in range(max(0, after + 1), len(records)):
        if source in records[index].source_indices:
            return index
    return None


def pair_specs_for_plan(
    records: Sequence[base.FrameRecord],
    plan,
    source_count: int,
) -> list[PairSpec]:
    if len(records) < 2:
        return []

    if plan.kind == "loop":
        last_source = max(0, source_count - 1)
        start = _source_record_index(records, last_source)
        if start is None:
            return []
        pairs = [
            PairSpec(index, index + 1, False, last_source)
            for index in range(start, len(records) - 1)
        ]
        pairs.append(PairSpec(len(records) - 1, 0, True, last_source))
        return pairs

    source_gap = int(plan.source_gap if plan.source_gap is not None else 0)
    start = _source_record_index(records, source_gap)
    if start is None:
        return []
    end = _source_record_index(records, source_gap + 1, after=start)
    if end is None or end <= start:
        return []
    return [
        PairSpec(index, index + 1, False, source_gap)
        for index in range(start, end)
    ]


def selected_pair_specs(records, plans, source_count: int) -> list[PairSpec]:
    seen: set[tuple[int, int, bool]] = set()
    result: list[PairSpec] = []
    for plan in plans:
        for pair in pair_specs_for_plan(records, plan, source_count):
            key = (pair.left_index, pair.right_index, pair.wrap)
            if key in seen:
                continue
            seen.add(key)
            result.append(pair)
    result.sort(key=lambda item: (item.wrap, item.left_index, item.right_index))
    return result


def _gap_count(settings: dict) -> int:
    try:
        return max(0, int(settings.get("frames_to_fill", 1)))
    except (TypeError, ValueError):
        return 1


def _interpolation_count(settings: dict) -> int:
    try:
        multiplier = int(settings.get("rife_multiplier", settings.get("interpolation_multiplier", 1)) or 1)
    except (TypeError, ValueError):
        return 0
    return max(0, multiplier - 1) if multiplier in {1, 2, 4, 8} else 0


def _insert_pair_refs(
    records: list[base.FrameRecord],
    pair: PairSpec,
    refs: list[dict],
    *,
    engine: str,
    source_count: int,
):
    refs = sorted(refs, key=lambda item: float(item.get("t", 0.5)))
    if not refs:
        return

    if pair.wrap:
        left_record = records[-1]
        right_record = records[0]
        left_index = len(records) - 1
    else:
        if pair.left_index < 0 or pair.left_index + 1 >= len(records):
            return
        left_record = records[pair.left_index]
        right_record = records[pair.left_index + 1]
        left_index = pair.left_index

    old_duration = float(left_record.duration)
    times = [0.0] + [max(0.0, min(1.0, float(ref.get("t", 0.5)))) for ref in refs] + [1.0]
    intervals = [old_duration * max(0.0, times[i + 1] - times[i]) for i in range(len(times) - 1)]
    left_record.duration = intervals[0]
    depth = max(int(left_record.recursive_depth), int(right_record.recursive_depth)) + 1
    between = (
        (max(0, source_count - 1), 0)
        if pair.wrap
        else (pair.source_gap, pair.source_gap + 1)
    )

    inserted: list[base.FrameRecord] = []
    for offset, ref in enumerate(refs):
        with Image.open(ref["path"]) as image:
            middle = image.convert("RGBA").copy()
        inserted.append(
            base.FrameRecord(
                image=middle,
                duration=intervals[offset + 1],
                source_indices=set(),
                generated=True,
                engine=engine,
                between_source_frames=between,
                recursive_depth=depth,
            )
        )

    if pair.wrap:
        records.extend(inserted)
    else:
        records[left_index + 1:left_index + 1] = inserted


def _engine_pass(
    temporal_v2,
    legacy,
    records: list[base.FrameRecord],
    plans,
    source_count: int,
    settings: dict,
    stage_dir: Path,
    *,
    operation: str,
    pass_number: int,
    pass_total: int,
    progress,
):
    pair_specs = selected_pair_specs(records, plans, source_count)
    if not pair_specs:
        progress(1.0, f"Pass {pass_number}/{pass_total} · no selected gaps remain")
        return []

    if operation == "gap":
        engine = str(settings.get("frame_generator", "none")).lower()
        if engine not in {"eden", "speed"}:
            progress(1.0, f"Pass {pass_number}/{pass_total} · gap filling skipped (generator is None)")
            return []
        count = _gap_count(settings)
        runner = lambda tasks, cb: temporal_v2._run_generator(engine, tasks, stage_dir, progress=cb)
        action = "gap filling"
    else:
        engine = str(settings.get("interpolator", "none")).lower()
        if engine not in {"rife", "amt"}:
            progress(1.0, f"Pass {pass_number}/{pass_total} · interpolation skipped (interpolator is None)")
            return []
        count = _interpolation_count(settings)
        if count <= 0:
            progress(1.0, f"Pass {pass_number}/{pass_total} · interpolation skipped (density is 1×)")
            return []
        runner = lambda tasks, cb: temporal_v2._run_interpolator(legacy, engine, tasks, stage_dir, progress=cb)
        action = "interpolation"

    input_paths = base._save_records(records, stage_dir / "input")
    threshold = float(settings.get("missing_threshold", 12.0))
    alpha_threshold = int(settings.get("alpha_threshold", 8))
    tasks: list[dict] = []
    mapping: dict[str, PairSpec] = {}
    for ordinal, pair in enumerate(pair_specs):
        task_id = f"pass-{pass_number}-pair-{ordinal}"
        left = input_paths[pair.left_index]
        right = input_paths[pair.right_index]
        task = {
            "id": task_id,
            "left": str(left),
            "right": str(right),
            "count": count,
            "threshold": threshold,
            "max_frames": temporal_v2.AUTO_FILL_MAX_FRAMES,
            "alpha_threshold": alpha_threshold,
        }
        tasks.append(task)
        mapping[task_id] = pair

    progress(0.0, f"Pass {pass_number}/{pass_total} · starting {engine.upper()} {action}")
    result = runner(
        tasks,
        lambda fraction: progress(
            max(0.0, min(1.0, float(fraction))),
            f"Pass {pass_number}/{pass_total} · {engine.upper()} {action}",
        ),
    )

    diagnostics: list[dict] = []
    normal_results: list[tuple[int, PairSpec, list[dict], str]] = []
    wrap_results: list[tuple[PairSpec, list[dict], str]] = []
    for task_id, item in result.items():
        pair = mapping.get(task_id)
        if pair is None:
            continue
        refs = temporal_v2._task_refs(item)
        if item.get("limit_reached"):
            diagnostics.append(
                {
                    "pass": pass_number,
                    "operation": operation,
                    "engine": engine,
                    "pair": task_id,
                    "limit_reached": True,
                    "max_score": item.get("max_score"),
                }
            )
        if pair.wrap:
            wrap_results.append((pair, refs, engine))
        else:
            normal_results.append((pair.left_index, pair, refs, engine))

    for _, pair, refs, engine_name in sorted(normal_results, key=lambda item: item[0], reverse=True):
        _insert_pair_refs(records, pair, refs, engine=engine_name, source_count=source_count)
    for pair, refs, engine_name in wrap_results:
        current_wrap = PairSpec(len(records) - 1, 0, True, pair.source_gap)
        _insert_pair_refs(records, current_wrap, refs, engine=engine_name, source_count=source_count)

    progress(1.0, f"Pass {pass_number}/{pass_total} · {engine.upper()} {action} complete")
    return diagnostics


def _repair_pass(
    records: list[base.FrameRecord],
    stage_dir: Path,
    *,
    pass_number: int,
    pass_total: int,
    progress,
):
    targets = [index for index, record in enumerate(records) if record.generated]
    if not targets:
        progress(1.0, f"Pass {pass_number}/{pass_total} · ProPainter skipped (no generated frames yet)")
        return None

    progress(0.0, f"Pass {pass_number}/{pass_total} · starting ProPainter audit + repair")
    repaired, stats = temporal_repair.repair_images(
        [record.image for record in records],
        targets,
        stage_dir / "temporal-repair",
        progress=lambda current, total: progress(
            current / max(1, total),
            f"Pass {pass_number}/{pass_total} · ProPainter temporal repair · step {current}/{total}",
        ),
    )
    for index in targets:
        records[index].image = repaired[index]
        prior = str(records[index].engine or "generated")
        if "propainter" not in prior.lower():
            records[index].engine = f"{prior}+propainter"
    progress(1.0, f"Pass {pass_number}/{pass_total} · ProPainter audit + repair complete")
    return stats


def install(temporal_v2):
    if getattr(temporal_v2, "_ordered_operation_pipeline_installed", False):
        return
    temporal_v2._ordered_operation_pipeline_installed = True

    def process_job(legacy, job_id: str, paths: list[Path], settings: dict):
        job = legacy.get_job(job_id)
        if not job:
            return
        job_dir = Path(job["job_dir"])

        def report(value, message):
            job_control.raise_if_cancelled(job_id)
            legacy.set_job(job_id, progress=int(value), message=message)

        try:
            report(10, "Loading frames")
            images = [legacy.load_rgba(path) for path in paths]
            frames = base._normalise_geometry(legacy, images, settings, report)
            duration = float(settings.get("duration", 100))
            records = [
                base.FrameRecord(image=frame, duration=duration, source_indices={index})
                for index, frame in enumerate(frames)
            ]
            source_count = len(records)

            manual_normal, manual_loop = temporal_v2.parse_target_gap_spec(settings.get("target_gaps", ""))
            removed_sources = []
            if settings.get("smart_reduction", False):
                report(42, "Smart frame reduction analysis")
                removed_sources, _ = temporal_v2._smart_reduce(
                    records,
                    threshold=float(settings.get("reduction_threshold", 2.0)),
                    alpha_threshold=int(settings.get("alpha_threshold", 8)),
                    protected_sources=temporal_v2._protected_sources(manual_normal, manual_loop, source_count),
                )

            stage_root = job_dir / "advanced"
            import shutil
            shutil.rmtree(stage_root, ignore_errors=True)
            stage_root.mkdir(parents=True, exist_ok=True)

            plans, auto_scores, loop_score = temporal_v2._plans_for_job(records, settings, source_count)
            pipeline = pipeline_for_settings(settings)
            diagnostics: list[dict] = []
            repair_stats: list[dict] = []
            pass_total = len(pipeline)

            if not pipeline:
                report(86, "No temporal passes selected")
            for index, operation in enumerate(pipeline, start=1):
                job_control.raise_if_cancelled(job_id)
                start = 46.0 + ((index - 1) / max(1, pass_total)) * 40.0
                end = 46.0 + (index / max(1, pass_total)) * 40.0

                def pass_progress(fraction, message, start=start, end=end):
                    report(round(start + (end - start) * max(0.0, min(1.0, float(fraction)))), message)

                pass_dir = stage_root / f"pass-{index:02d}-{operation}"
                if operation in {"gap", "interpolate"}:
                    diagnostics.extend(
                        _engine_pass(
                            temporal_v2,
                            legacy,
                            records,
                            plans,
                            source_count,
                            settings,
                            pass_dir,
                            operation=operation,
                            pass_number=index,
                            pass_total=pass_total,
                            progress=pass_progress,
                        )
                    )
                elif operation == "repair":
                    stats = _repair_pass(
                        records,
                        pass_dir,
                        pass_number=index,
                        pass_total=pass_total,
                        progress=pass_progress,
                    )
                    if stats:
                        repair_stats.append(dict(stats, pass_number=index))

            report(88, "Preparing optimised WebP deltas")
            output_path = job_dir / "animation.webp"
            durations = base._integer_durations(records)
            temporal_v2.save_webp_fast(
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
                    "loop_bridge": bool(
                        record.generated
                        and record.between_source_frames == (max(0, source_count - 1), 0)
                    ),
                }
                for index, record in enumerate(records)
                if record.generated
            ]
            limit_notes = [item for item in diagnostics if item.get("limit_reached")]
            message = (
                f"Ready: {len(records)} frames; smart reduction removed {len(removed_sources)} source frame"
                f"{'s' if len(removed_sources) != 1 else ''}; passes: "
                + (" → ".join(pipeline) if pipeline else "none")
            )
            if repair_stats:
                message += f"; ProPainter repair pass{'es' if len(repair_stats) != 1 else ''}: {len(repair_stats)}"
            if limit_notes:
                message += f"; {len(limit_notes)} adaptive fill(s) reached the safety ceiling"

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
                temporal_pipeline=pipeline,
                temporal_repair_passes=repair_stats,
            )
        except job_control.JobCancelled:
            raise
        except Exception as exc:
            legacy.set_job(job_id, status="error", message=str(exc), error=str(exc))

    temporal_v2.process_job = process_job
