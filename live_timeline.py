from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import engine_catalog


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _item(root: Path, job_id: str, *, key: str, name: str, stage: str, rel: str, generated=False, engine=""):
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
        stat = path.stat()
    except Exception:
        return None
    if not path.is_file() or stat.st_size <= 0:
        return None
    url = f"/live-frame?id={quote(job_id)}&rel={quote(rel)}&v={stat.st_mtime_ns}"
    return {
        "key": str(key),
        "name": str(name or path.name),
        "stage": str(stage or "Frame"),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "url": url,
        "generated": bool(generated),
        "engine": str(engine or ""),
    }


def _canonical(root: Path, job_id: str):
    manifest = _read_json(root / "live-timeline" / "manifest.json")
    if not manifest:
        return [], -1
    items = []
    for frame in manifest.get("frames", []):
        if not isinstance(frame, dict):
            continue
        item = _item(
            root,
            job_id,
            key=frame.get("key", frame.get("rel", "")),
            name=frame.get("name", "frame.png"),
            stage=frame.get("stage", "Frame"),
            rel=str(frame.get("rel", "")),
            generated=frame.get("generated", False),
            engine=frame.get("engine", ""),
        )
        if item:
            items.append(item)
    return items, int(manifest.get("completed_pass", -1))


def _latest_live_base(root: Path, completed_pass: int):
    candidates = []
    for path in root.glob("pass-*-*/live-base.json"):
        value = _read_json(path)
        if not value:
            continue
        number = int(value.get("pass_number", -1))
        if number > completed_pass:
            candidates.append((number, path.parent, value))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _base_items(root: Path, job_id: str, base: dict):
    items = []
    for frame in base.get("frames", []):
        if not isinstance(frame, dict):
            continue
        item = _item(
            root,
            job_id,
            key=frame.get("key", frame.get("rel", "")),
            name=frame.get("name", "frame.png"),
            stage=frame.get("stage", "Frame"),
            rel=str(frame.get("rel", "")),
            generated=frame.get("generated", False),
            engine=frame.get("engine", ""),
        )
        if item:
            items.append(item)
    return items


def _engine_outputs(root: Path, job_id: str, pass_dir: Path, base: dict):
    operation = str(base.get("operation", ""))
    engine_id = str(base.get("engine", "") or "").strip().lower()
    engine_label = engine_catalog.engine_label(engine_id)
    if operation == "gap":
        stage = f"Generated · {engine_label if engine_id else 'generator'}"
        folders = list(pass_dir.glob("*-generator-v2/frames")) + list(pass_dir.glob("*-generator/frames"))
    else:
        stage = f"Interpolated · {engine_label if engine_id else 'interpolator'}"
        folders = list(pass_dir.glob("*-interpolator-v2/frames")) + list(pass_dir.glob("*-interpolator/frames"))

    outputs: dict[int, list[dict]] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.png")):
            parts = path.stem.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                task_index = int(parts[0])
                counter = int(parts[1])
            except ValueError:
                continue
            rel = path.relative_to(root).as_posix()
            item = _item(
                root,
                job_id,
                key=f"pass:{base.get('pass_number', 0)}:task:{task_index}:frame:{counter}",
                name=path.name,
                stage=stage,
                rel=rel,
                generated=True,
                engine=engine_id,
            )
            if item:
                outputs.setdefault(task_index, []).append((counter, item))

    return {index: [item for _, item in sorted(values, key=lambda pair: pair[0])] for index, values in outputs.items()}


def _dynamic_engine(root: Path, job_id: str, pass_dir: Path, base: dict):
    base_items = _base_items(root, job_id, base)
    if not base_items:
        return []
    output_by_task = _engine_outputs(root, job_id, pass_dir, base)
    insert_after: dict[int, list[dict]] = {}
    wrap_items: list[dict] = []
    for pair in base.get("pairs", []):
        if not isinstance(pair, dict):
            continue
        task_index = int(pair.get("task_index", -1))
        outputs = output_by_task.get(task_index, [])
        if not outputs:
            continue
        if bool(pair.get("wrap", False)):
            wrap_items.extend(outputs)
        else:
            insert_after.setdefault(int(pair.get("left_index", -1)), []).extend(outputs)

    timeline = []
    for index, item in enumerate(base_items):
        timeline.append(item)
        timeline.extend(insert_after.get(index, []))
    timeline.extend(wrap_items)
    return timeline


def _dynamic_repair(root: Path, job_id: str, pass_dir: Path, base: dict):
    timeline = _base_items(root, job_id, base)
    output_root = pass_dir / "temporal-repair" / "output"
    if not output_root.is_dir():
        return timeline

    paths = list((output_root / "frames").glob("*.png")) if (output_root / "frames").is_dir() else []
    paths.extend(path for path in output_root.glob("*.png") if path not in paths)
    for path in paths:
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if index < 0 or index >= len(timeline):
            continue
        rel = path.relative_to(root).as_posix()
        previous = timeline[index]
        prior_stage = str(previous.get("stage", "Frame"))
        item = _item(
            root,
            job_id,
            key=previous.get("key", f"timeline:{index}"),
            name=previous.get("name", path.name),
            stage=f"Repaired · ProPainter · {prior_stage}",
            rel=rel,
            generated=previous.get("generated", False),
            engine="propainter",
        )
        if item:
            timeline[index] = item
    return timeline


def build_timeline(root: Path, job_id: str):
    canonical, completed_pass = _canonical(root, job_id)
    current = _latest_live_base(root, completed_pass)
    if current is None:
        return canonical
    _, pass_dir, base = current
    operation = str(base.get("operation", ""))
    if operation == "repair":
        return _dynamic_repair(root, job_id, pass_dir, base)
    if operation in {"gap", "interpolate"}:
        return _dynamic_engine(root, job_id, pass_dir, base)
    return _base_items(root, job_id, base) or canonical


def allowed_live_frame(relative: Path) -> bool:
    parts = [part.lower() for part in relative.parts]
    if not parts or relative.suffix.lower() != ".png":
        return False
    first = parts[0]
    if first == "live-timeline" or first.startswith("pass-"):
        return True
    if "generator" in first or "interpolator" in first:
        return True
    return first == "temporal-repair" and len(parts) >= 2 and parts[1] == "output"


def install(job_control_module) -> None:
    if getattr(job_control_module, "_live_timeline_installed", False):
        return
    job_control_module._live_timeline_installed = True
    legacy_live_frames = job_control_module._live_frames

    def live_frames(job_id: str):
        root, job = job_control_module._live_root(job_id)
        if root is None or not root.is_dir():
            return [], job
        timeline = build_timeline(root, job_id)
        if timeline:
            return timeline, job
        return legacy_live_frames(job_id)

    job_control_module._candidate_live_frame = allowed_live_frame
    job_control_module._live_frames = live_frames
