from __future__ import annotations

import json
import os
from pathlib import Path

import advanced_pipeline as base


def _ensure_keys(records) -> None:
    for index, record in enumerate(records):
        if getattr(record, "_live_key", None):
            continue
        sources = sorted(getattr(record, "source_indices", set()) or set())
        if not getattr(record, "generated", False) and sources:
            record._live_key = f"source:{sources[0]}"
        else:
            record._live_key = f"timeline:{index}"


def _stage_for(record) -> str:
    if not getattr(record, "generated", False):
        return "Original"
    engine = str(getattr(record, "engine", "") or "generated")
    lower = engine.lower()
    if "propainter" in lower:
        prior = engine.replace("+propainter", "").replace("propainter+", "").strip("+")
        return f"Repaired · {prior.upper() if prior else 'ProPainter'} + ProPainter"
    if any(name in lower for name in ("rife", "amt")):
        return f"Interpolated · {engine.upper()}"
    if any(name in lower for name in ("eden", "speed")):
        return f"Generated · {engine.upper()}"
    return f"Generated · {engine}"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _base_manifest(records, input_paths, stage_root: Path, stage_dir: Path, *, operation: str, pass_number: int, pair_specs=None, engine="") -> None:
    _ensure_keys(records)
    frames = []
    for index, (record, path) in enumerate(zip(records, input_paths)):
        frames.append(
            {
                "key": str(record._live_key),
                "name": f"frame-{index + 1:04d}.png",
                "rel": _relative(Path(path), stage_root),
                "stage": _stage_for(record),
                "generated": bool(getattr(record, "generated", False)),
                "engine": str(getattr(record, "engine", "") or ""),
                "timeline_index": index,
            }
        )

    pairs = []
    for task_index, pair in enumerate(pair_specs or []):
        pairs.append(
            {
                "task_index": task_index,
                "left_index": int(pair.left_index),
                "right_index": int(pair.right_index),
                "wrap": bool(pair.wrap),
                "source_gap": int(pair.source_gap),
            }
        )

    _atomic_json(
        stage_dir / "live-base.json",
        {
            "pass_number": int(pass_number),
            "operation": str(operation),
            "engine": str(engine),
            "frames": frames,
            "pairs": pairs,
        },
    )


def _publish(records, stage_root: Path, completed_pass: int) -> None:
    _ensure_keys(records)
    live_root = stage_root / "live-timeline"
    revision = live_root / f"rev-{int(completed_pass):03d}"
    revision.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, record in enumerate(records):
        path = revision / f"{index:06d}.png"
        record.image.convert("RGBA").save(path)
        frames.append(
            {
                "key": str(record._live_key),
                "name": f"frame-{index + 1:04d}.png",
                "rel": _relative(path, stage_root),
                "stage": _stage_for(record),
                "generated": bool(getattr(record, "generated", False)),
                "engine": str(getattr(record, "engine", "") or ""),
                "timeline_index": index,
            }
        )
    _atomic_json(
        live_root / "manifest.json",
        {
            "completed_pass": int(completed_pass),
            "frames": frames,
        },
    )


def install(operation_pipeline_module) -> None:
    if getattr(operation_pipeline_module, "_live_timeline_pipeline_installed", False):
        return
    operation_pipeline_module._live_timeline_pipeline_installed = True

    original_engine_pass = operation_pipeline_module._engine_pass
    original_repair_pass = operation_pipeline_module._repair_pass

    def engine_pass(temporal_v2, legacy, records, plans, source_count, settings, stage_dir, *, operation, pass_number, pass_total, progress):
        stage_root = Path(stage_dir).parent
        _ensure_keys(records)
        input_paths = base._save_records(records, Path(stage_dir) / "input")
        pair_specs = operation_pipeline_module.selected_pair_specs(records, plans, source_count)
        if operation == "gap":
            engine = str(settings.get("frame_generator", "none")).lower()
        else:
            engine = str(settings.get("interpolator", "none")).lower()
        _base_manifest(
            records,
            input_paths,
            stage_root,
            Path(stage_dir),
            operation=operation,
            pass_number=pass_number,
            pair_specs=pair_specs,
            engine=engine,
        )
        result = original_engine_pass(
            temporal_v2,
            legacy,
            records,
            plans,
            source_count,
            settings,
            stage_dir,
            operation=operation,
            pass_number=pass_number,
            pass_total=pass_total,
            progress=progress,
        )
        generated_counter = 0
        for record in records:
            if getattr(record, "_live_key", None):
                continue
            record._live_key = f"pass:{pass_number}:generated:{generated_counter}"
            generated_counter += 1
        _publish(records, stage_root, pass_number)
        return result

    def repair_pass(records, stage_dir, *, pass_number, pass_total, progress):
        stage_root = Path(stage_dir).parent
        _ensure_keys(records)
        input_paths = base._save_records(records, Path(stage_dir) / "input")
        _base_manifest(
            records,
            input_paths,
            stage_root,
            Path(stage_dir),
            operation="repair",
            pass_number=pass_number,
            pair_specs=[],
            engine="propainter",
        )
        result = original_repair_pass(
            records,
            stage_dir,
            pass_number=pass_number,
            pass_total=pass_total,
            progress=progress,
        )
        _publish(records, stage_root, pass_number)
        return result

    operation_pipeline_module._engine_pass = engine_pass
    operation_pipeline_module._repair_pass = repair_pass
